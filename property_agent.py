"""
Property Enquiry Agent
======================
Flow: login → load_preferences → chat → parse_intent
      → [update_preferences] + [nl_to_sql → query_listings] + [start_booking] → format_reply

Booking sub-flow (once a FINALIZE is detected): start_booking (ask for email)
      → booking_flow (collect email, then visit time, then send confirmation
        email + calendar RSVP) -- runs on later turns, bypassing chat/parse_intent.

Run:
    python property_agent.py
"""

import uuid
import json
import re
import time
import logging
import logging.handlers
import smtplib
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import psycopg
from psycopg.rows import dict_row
from typing import TypedDict, Optional
from langchain_mistralai import ChatMistralAI
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from typing import Annotated
import os
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
# Everything (LLM calls, generated SQL, DB errors, call lifecycle) goes through
# this logger so it's all visible in one place: console (stdout, captured by
# uvicorn/systemd/docker) and a rotating file next to this module.
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "agent.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3),
    ],
    force=True,  # override any handlers uvicorn/another import may have already attached to root
)
logger = logging.getLogger("property_agent")


def _ctx(state: dict) -> str:
    """Short 'phone=.. thread=..' prefix so every log line is traceable to a caller/session."""
    phone = state.get("phone_number", "?")
    thread = str(state.get("thread_id", "?"))[:8]
    return f"[phone={phone} thread={thread}]"


# ── Config ────────────────────────────────────────────────────────────────────

DB_URI = os.getenv("DB_URI")
# DB_URI = "postgresql://postgres:1234@localhost:5432/postgres?sslmode=prefer&connect_timeout=10"

# print(API_KEY)
API_KEY = os.getenv("MISTRAL_API_KEY")

# print(API_KEY)


# ChatMistralAI defaults to a 120s per-attempt timeout -- fine for a batch
# script, disastrous for a live voice call (observed in production: single
# calls hanging 69s/104s/177s while the caller sat in silence). A much
# shorter timeout means a stuck primary attempt fails over to a retry/the
# fallback model in seconds instead of minutes.
LLM_TIMEOUT_SEC = 15

primary_llm = ChatMistralAI(model="mistral-large-latest", max_retries=3, timeout=LLM_TIMEOUT_SEC, temperature=0.1, api_key=API_KEY)
fallback_llm = ChatMistralAI(model="open-mistral-7b", max_retries=3, timeout=LLM_TIMEOUT_SEC, temperature=0.1, api_key=API_KEY)

llm = primary_llm.with_fallbacks([fallback_llm])
logger.info(f"LLM configured | primary=mistral-large-latest fallback=open-mistral-7b | DB host={DB_URI.split('@')[-1].split('/')[0] if '@' in DB_URI else DB_URI}")

# ── State ─────────────────────────────────────────────────────────────────────

class AppState(TypedDict):
    messages: Annotated[list, add_messages]
    phone_number      : str
    thread_id         : str
    preferences       : dict   # long-term, from preferences table
    preference_updates: dict   # extracted this turn, cleared after write
    search_query      : str    # natural language, extracted this turn
    sql_query         : str    # generated from search_query
    search_results    : list   # rows from the "Property" table
    reply             : str    # clean reply text (no tags)

    # ── Booking / visit-confirmation sub-flow ──────────────────────────────
    # Unlike the fields above, these three are NOT reset every turn in run()/
    # _run_graph() -- they're meant to persist across multiple turns (email,
    # then visit time, then send) for as long as a booking is in progress, so
    # they're only ever written by finalize/booking nodes and cleared once
    # the confirmation email has actually been sent.
    finalize_property : dict   # extracted this turn only, cleared after use
    booking_stage     : str    # "" | "awaiting_email" | "awaiting_visit_time" | "awaiting_confirmation"
    pending_property  : dict   # the property the customer finalized
    customer_email    : str
    visit_datetime_iso: str    # resolved visit time, held until confirmed and sent

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    try:
        return psycopg.connect(DB_URI, autocommit=True, row_factory=dict_row)
    except Exception:
        logger.exception("DB connection failed")
        raise



# ── Login (runs before graph) ─────────────────────────────────────────────────

def login() -> tuple[str, str]:
    """
    Login flow:
      1. User enters phone number
      2. If existing sessions found → user picks one or starts new
      3. Returns (phone_number, thread_id)
    """
    print("\n" + "═"*50)
    print("  PROPERTY ENQUIRY AGENT")
    print("═"*50)
    phone = input("\nEnter your phone number to login: ").strip()

    with get_conn() as conn:
        # ensure user exists
        conn.execute("""
            INSERT INTO users (phone_number) VALUES (%s)
            ON CONFLICT (phone_number) DO NOTHING
        """, (phone,))

        # ensure preferences row exists
        conn.execute("""
            INSERT INTO preferences (phone_number) VALUES (%s)
            ON CONFLICT (phone_number) DO NOTHING
        """, (phone,))

        # fetch all existing sessions for this user
        rows = conn.execute(
            """SELECT thread_id, created_at, last_active
               FROM sessions
               WHERE phone_number = %s
               ORDER BY last_active DESC""",
            (phone,)
        ).fetchall()

        if not rows:
            # first time user — no choice to make
            thread_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO sessions (thread_id, phone_number) VALUES (%s, %s)",
                (thread_id, phone)
            )
            print(f"\n✓ New session started. Welcome!")

        else:
            # show existing sessions and let user choose
            print(f"\nFound {len(rows)} existing session(s):\n")
            for i, row in enumerate(rows, 1):
                print(f"  [{i}] thread: {row['thread_id']}")
                print(f"       started : {row['created_at'].strftime('%d %b %Y %I:%M %p')}")
                print(f"       last active: {row['last_active'].strftime('%d %b %Y %I:%M %p')}\n")

            print(f"  [N] Start a new chat")
            print(f"  [D] Delete a session\n")

            choice = input("Your choice: ").strip().upper()

            if choice == "N":
                thread_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO sessions (thread_id, phone_number) VALUES (%s, %s)",
                    (thread_id, phone)
                )
                print(f"\n✓ New session started.")

            elif choice == "D":
                del_input = input("Enter session number to delete: ").strip()
                try:
                    del_index = int(del_input) - 1
                    if 0 <= del_index < len(rows):
                        tid_to_delete = str(rows[del_index]["thread_id"])
                        delete_thread(phone, tid_to_delete)
                        print(f"\n✓ Session deleted. Starting a new session.")
                    else:
                        print("\n✗ Invalid number. Starting a new session.")
                except ValueError:
                    print("\n✗ Invalid input. Starting a new session.")

                # after deletion always start fresh
                thread_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO sessions (thread_id, phone_number) VALUES (%s, %s)",
                    (thread_id, phone)
                )

            else:
                # treat as session number
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(rows):
                        thread_id = str(rows[index]["thread_id"])
                        conn.execute(
                            "UPDATE sessions SET last_active = NOW() WHERE thread_id = %s",
                            (thread_id,)
                        )
                        print(f"\n✓ Resuming session {index + 1}.")
                    else:
                        print("\n✗ Invalid choice. Resuming most recent session.")
                        thread_id = str(rows[0]["thread_id"])
                except ValueError:
                    print("\n✗ Invalid input. Resuming most recent session.")
                    thread_id = str(rows[0]["thread_id"])

    return phone, thread_id


def delete_thread(phone_number: str, thread_id: str) -> None:
    """
    Permanently delete a session and its checkpoints.
    Removes from:
      - sessions table
      - all PostgresSaver checkpoint tables (checkpoints, checkpoint_blobs, checkpoint_writes)

    Call explicitly — never called automatically anywhere in the flow.
    """
    with get_conn() as conn:
        # delete from sessions table
        conn.execute(
            "DELETE FROM sessions WHERE thread_id = %s AND phone_number = %s",
            (thread_id, phone_number)
        )
        # delete all checkpointer data for this thread
        conn.execute(
            "DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,)
        )
        conn.execute(
            "DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,)
        )
        conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,)
        )
    print(f"✓ Thread {thread_id} and all its checkpoints deleted.")

# ── Nodes ─────────────────────────────────────────────────────────────────────

# Preferences live in "Voice-user-pref": the same typed shape as Prisma's
# "UserPreference", but keyed by phone_number instead of userId, since a voice
# caller has no Google account to hang a userId off. The old `preferences`
# table squeezed all of this into bhk/min_amount/max_amount plus two free-text
# columns, so amenities, furnishing and rent-vs-buy had nowhere to live and
# were re-guessed from prose on every single query.
PREF_TABLE = '"Voice-user-pref"'
PREF_COLUMNS = ("city", "locality", "listingType", "bhk", "minArea", "maxArea",
                "minPrice", "maxPrice", "amenities", "furnishing", "otherPreference")

LISTING_TYPES = {"RENT", "SALE"}
FURNISHINGS = {"UNFURNISHED", "SEMI_FURNISHED", "FURNISHED"}


def load_preferences(state: AppState) -> dict:
    cols = ", ".join(f'"{c}"' for c in PREF_COLUMNS)
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT {cols} FROM {PREF_TABLE} WHERE phone_number = %s",
            (state["phone_number"],)
        ).fetchone()
    prefs = {k: v for k, v in row.items() if v not in (None, [])} if row else {}
    logger.info(f"{_ctx(state)} load_preferences -> {prefs}")
    return {"preferences": prefs}


SYSTEM_PROMPT   = """You are a helpful property search assistant for properties to rent and to buy across major Indian cities.

User's saved preferences: {preferences}

Every city and locality we actually have, as "City: its localities":
{known_locations}
Never name a city or locality outside that list, and never describe a specific property
unless it was returned by a search earlier in this conversation.

═══ HOW TO THINK BEFORE RESPONDING ═══

Before generating any response, reason through these questions silently:

STEP 1 — What is the user actually doing right now?
  Think: Is this a greeting? A question about history? A casual remark?
         A request to search? An expression of preference? A confirmation?

  Examples of what people mean:
  - "hi", "hello", "hey"           → just starting a conversation, wants a warm greeting
  - "what did i say before?"       → wants me to recall something from our chat history
  - "what are my preferences?"     → wants to know what's saved, NOT a search request
  - "i like HSR"                   → expressing a preference, may or may not want search
  - "show me flats in HSR"         → explicitly wants property listings
  - "thanks", "okay", "got it"     → just acknowledging, conversation is closing
  - "yes" after showing results    → confirming they liked what they saw

STEP 2 — Has the user mentioned any new preference right now?
  Think: Did they say anything about BHK, budget, area, or amenities IN THIS MESSAGE?
         Not what's saved — what did they say RIGHT NOW?

  If yes → extract ONLY those fields into PREF_UPDATE
  If no  → skip PREF_UPDATE entirely

  Soft preferences count too:
  - "i like HSR" → area preference
  - "something affordable" → not specific enough, skip
  - "i prefer 2bhk" → bhk preference
  - "pet friendly would be nice" → otherPreference

STEP 3 — Does the user want to see property listings RIGHT NOW?
  Think: Did they explicitly or clearly ask to find/show/search/list properties?
         OR is this just a greeting, question, or casual remark?

  Ask yourself: "Would a human real estate agent pull out their listings folder right now?"

  - Greeting → No, they would say hello back first
  - "show me flats" → Yes, clearly
  - "what about HSR?" → Yes, they want to see HSR options
  - "i like HSR" → Only if they haven't seen HSR listings recently in this conversation
  - "what did i say before?" → No, they want conversation recall not listings
  - "thanks" → No, conversation is wrapping up
  - Saved preferences exist but user just said "hi" → No, preferences don't trigger search

STEP 4 — What information do I have to work with?
  Think: What's in our conversation history? What's in saved preferences?
         If user asks about history → look through our messages above, not saved preferences
         If user asks to search → combine what they said NOW with saved preferences as defaults

STEP 5 — Has the user just finalized / decided on ONE specific property from the listings already shown earlier in this conversation?
  Think: Are they clearly picking one option, not just asking about it?

  - "I'll take that one", "let's go with the HSR one", "book that", "the second one sounds good, let's do that", "yes I want to visit this one" → Yes, they're finalizing
  - "tell me more about the second one" → No, that's just a question, not a decision
  - "show me more options" → No, that's still browsing
  - "thanks", "bye", "exit", "okay great" or any conversation-closing message → No, NEVER emit FINALIZE on a goodbye/closing turn
  - A visit was already booked for this property earlier in this conversation → No, never re-finalize a property that's already booked

  If yes → include a FINALIZE tag: a JSON object with whatever fields of that specific property you can identify from what was already shown in this conversation (area, bhk, rent, amenities). Only include fields you actually know from the conversation; omit fields you don't. Also make REPLY gently and naturally ask for the customer's email address so their property details and a visit invite can be sent over.
  If no → skip FINALIZE entirely.

═══ NOW RESPOND IN THIS EXACT FORMAT ═══

REPLY: <your natural, friendly response>
PREF_UPDATE: <only if step 2 says yes — json with only the mentioned fields>
SEARCH: <only if step 3 says yes — natural language describing what to search>
FINALIZE: <only if step 5 says yes — json describing the finalized property>

═══ FORMAT RULES ═══

- REPLY is mandatory every time
- PREF_UPDATE, SEARCH, and FINALIZE are optional — include only when your thinking above justifies it
- Nothing before REPLY, nothing after the last tag you include
- PREF_UPDATE keys — use the right one, never invent others:
    city (str)             a city from the list above, e.g. "Mumbai"
    locality (str)         a locality from the list above, e.g. "Worli"
    listingType (str)      "RENT" or "SALE" — record it as soon as you know which
    bhk (int)
    minPrice / maxPrice    numbers; monthly rent for RENT, total price for SALE
    minArea / maxArea      floor size in square feet
    amenities (list)       e.g. ["gym", "swimming pool"] — real amenities only
    furnishing (str)       "UNFURNISHED", "SEMI_FURNISHED" or "FURNISHED"
    otherPreference (str)  anything genuinely not covered above
  City and locality are separate: "Worli" is locality "Worli" with city "Mumbai".
  Never put a region like "NCR" in city or locality — regions belong in SEARCH, not storage.
- FINALIZE keys: area (str), bhk (int), rent (int), amenities (str) — only the ones you actually know
- Budget hints: "under 25k" → maxPrice: 25000 | "above 10k" → minPrice: 10000 | "around 25k" → maxPrice: 25000
- If user's name is known from conversation, use it naturally in REPLY
- Strictly Never under any circumstance hallucinate or fabricate properties on your own.
- If the user states ANY concrete preference in this message (bhk, area, budget, amenity),
  you MUST include a PREF_UPDATE line with those fields -- even when the same message also
  triggers SEARCH or FINALIZE. Skipping PREF_UPDATE loses the customer's preference forever.
- Only record what the CUSTOMER chose. If you offered "Electronic City or Whitefield?" and they
  answered something else, no area was chosen -- do not save one. Never put two places in `area`.
- A saved preference stays in force until it is replaced or withdrawn, so when the customer
  DROPS one, send that key as null. Omitting the key keeps the old value forever and silently
  narrows every later search.
    "any budget, I don't care about price" -> PREF_UPDATE: {{"minPrice": null, "maxPrice": null}}
    "anywhere is fine now"                 -> PREF_UPDATE: {{"city": null, "locality": null}}
    "forget everything, start over"        -> PREF_UPDATE: {{"city": null, "locality": null, "listingType": null, "bhk": null, "minPrice": null, "maxPrice": null, "amenities": null, "furnishing": null, "otherPreference": null}}
  When they REPLACE one instead ("actually make it 4bhk"), just send the new value.
- When the customer switches between renting and buying, any budget they gave for the old mode
  no longer applies -- send the budget keys as null unless they restate a figure.

═══ WRITING THE SEARCH LINE ═══

Whoever builds the SQL sees ONLY your SEARCH line -- not this conversation. So SEARCH must
restate every constraint that is still in force, even ones agreed several turns ago:
whether they are renting or buying, bhk, the city and/or locality, the budget, and any
amenity or furnishing. A SEARCH line that drops the city searches the whole country, and
one that drops "buying" quietly returns rentals.

  Conversation so far: they said Mumbai, then "4bhk to buy"
  BAD  -> SEARCH: 4bhk to buy
  GOOD -> SEARCH: 4BHK to buy in Mumbai

If the customer names a region rather than a city (NCR, Delhi NCR, MMR, tri-city), keep the
region word in the SEARCH line -- the SQL step knows which cities each region covers.

═══ SPEAKING TO A CALLER ═══

Every REPLY is read aloud by text-to-speech. Write words a person would say:
- No markdown, no bullet points, no numbered lists, no asterisks, no headings, no emoji.
- No digits and no symbols: say "twenty five thousand rupees a month", never "25,000" or "₹25000/mo".
  Indian scale: 5000000 is "fifty lakh", 10000000 is "one crore" -- never "million".
- Never mention REPLY, PREF_UPDATE, SEARCH or FINALIZE inside the REPLY text, and never explain
  to the caller what you are or aren't recording. They are internal.
- Never describe the same property twice. If you already told them about a place earlier in this
  conversation and they ask for others, say plainly that it's the only match rather than
  re-listing it as if it were new.
- You can only speak about properties returned by a search in THIS conversation. For anything the
  search didn't give you -- pet rules, deposit, maintenance charges, legal or society terms --
  say you'll have to check and offer to connect them, rather than guessing.

═══ WORKED EXAMPLES (follow these shapes exactly) ═══

Example 1 -- a search request that states preferences (most common case):
User says: "show me 2bhk flats in Whitefield under 25000"

REPLY: Sure! Let me pull up 2BHK options in Whitefield under twenty-five thousand for you.
PREF_UPDATE: {{"bhk": 2, "maxPrice": 25000, "listingType": "RENT", "city": "Bengaluru", "locality": "Whitefield"}}
SEARCH: 2BHK flats to rent in Whitefield, Bengaluru under 25000

Example 2 -- a finalize turn that also mentions a preference:
Conversation so far included: "There's one for eighteen thousand, pet-friendly, in HSR, 2BHK."
User now says: "I'll take the 18k one, pet friendly is important to me"

REPLY: Great choice! I'll get the details of that pet-friendly 2BHK in HSR at eighteen thousand a month sent over. Could you share your email address so I can send it to you along with a visit invite?
PREF_UPDATE: {{"otherPreference": "pet friendly"}}
FINALIZE: {{"area": "HSR", "bhk": 2, "rent": 18000, "amenities": "pet friendly"}}

Every time step 5's answer is yes, your output MUST include a FINALIZE line in exactly this
shape: the word FINALIZE, a colon, then a single-line raw JSON object. The same goes for
PREF_UPDATE whenever step 2's answer is yes. Never wrap either in backticks or a code fence,
never describe the update/finalization only in prose inside REPLY without emitting the tag
line, and never skip a tag just because you already mentioned its content in REPLY.

"""


def chat(state: AppState) -> dict:
    user_turn = state["messages"][-1].content if state["messages"] else ""
    logger.info(f"{_ctx(state)} chat <- user said: {user_turn!r}")

    system = SystemMessage(
        content=SYSTEM_PROMPT.format(
            preferences=json.dumps(state.get("preferences", {})),
            known_locations=get_known_locations(),
        )
    )
    t0 = time.monotonic()
    response = llm.invoke([system] + state["messages"])
    elapsed = time.monotonic() - t0
    logger.info(f"{_ctx(state)} chat -> LLM responded in {elapsed:.2f}s")
    logger.debug(f"{_ctx(state)} chat raw LLM output:\n{response.content}")

    # with open("agent_debug.log", "a") as f:
    #     f.write(f"\n{'='*60}\n")
    #     f.write(f"RAW LLM OUTPUT:\n{response}\n")
    #     f.write(f"{'='*60}\n")
    return {"messages": [response]}


def parse_intent(state: AppState) -> dict:
    raw = state["messages"][-1].content

    # extract REPLY
    reply_match = re.search(r"REPLY:\s*(.+?)(?=PREF_UPDATE:|SEARCH:|FINALIZE:|$)", raw, re.DOTALL)
    if reply_match:
        reply = reply_match.group(1).strip()
    else:
        # No REPLY: prefix at all. Falling back to the whole raw output would
        # read the PREF_UPDATE/SEARCH/FINALIZE lines aloud to the caller, so
        # keep only the prose before the first tag.
        reply = re.split(r"(?:PREF_UPDATE|SEARCH|FINALIZE)\s*:", raw)[0].strip()
    # The model sometimes narrates its own tag bookkeeping inside REPLY --
    # "(No PREF_UPDATE or SEARCH yet...)" -- which TTS would also read out.
    reply = re.sub(r"\(\s*(?:No\s+)?(?:PREF_UPDATE|SEARCH|FINALIZE|REPLY)\b[^)]*\)", "", reply,
                   flags=re.IGNORECASE)
    reply = re.sub(r"\n\s*-{3,}\s*$", "", reply).strip()

    # extract PREF_UPDATE (tolerate a ```json fence, same as FINALIZE below --
    # without this, a fenced PREF_UPDATE silently drops the user's preferences)
    # Tolerate one to three backticks -- a single-backtick `{...}` was silently
    # dropping the customer's preferences.
    pref_match = re.search(r"PREF_UPDATE:\s*(?:`{1,3}(?:json)?\s*)?(\{.+?\})", raw, re.DOTALL)
    preference_updates = {}
    if pref_match:
        try:
            preference_updates = json.loads(pref_match.group(1))
        except json.JSONDecodeError:
            logger.warning(f"{_ctx(state)} parse_intent: PREF_UPDATE block wasn't valid JSON, ignoring: {pref_match.group(1)!r}")

    # extract SEARCH
    search_match = re.search(r"SEARCH:\s*(.+?)(?=PREF_UPDATE:|REPLY:|FINALIZE:|$)", raw, re.DOTALL)
    search_query = search_match.group(1).strip() if search_match else ""

    # extract FINALIZE (tolerate the model wrapping the JSON in a ```json fence
    # despite being told not to)
    finalize_match = re.search(r"FINALIZE:\s*(?:`{1,3}(?:json)?\s*)?(\{.+?\})", raw, re.DOTALL)
    finalize_property = {}
    if finalize_match:
        try:
            finalize_property = json.loads(finalize_match.group(1))
        except json.JSONDecodeError:
            logger.warning(f"{_ctx(state)} parse_intent: FINALIZE block wasn't valid JSON, ignoring: {finalize_match.group(1)!r}")

    logger.info(
        f"{_ctx(state)} parse_intent -> reply={reply[:80]!r}{'...' if len(reply) > 80 else ''} "
        f"pref_update={preference_updates} search_query={search_query!r} finalize_property={finalize_property}"
    )

    return {
        "reply"             : reply,
        "preference_updates": preference_updates,
        "search_query"      : search_query,
        "search_results"    : [],
        "sql_query"         : "",
        "finalize_property" : finalize_property,
    }


def _clean_pref_value(col: str, value):
    """Keep the typed columns typed. The model supplies these, and a stray
    'semi furnished' or a bare string for amenities would abort the whole turn
    on a Postgres type error, so anything unusable is dropped to NULL instead."""
    if value is None:
        return None
    if col == "listingType":
        v = str(value).strip().upper()
        return v if v in LISTING_TYPES else None
    if col == "furnishing":
        v = str(value).strip().upper().replace(" ", "_").replace("-", "_")
        v = {"FULLY_FURNISHED": "FURNISHED", "SEMIFURNISHED": "SEMI_FURNISHED"}.get(v, v)
        return v if v in FURNISHINGS else None
    if col == "amenities":
        items = value if isinstance(value, list) else [value]
        return [str(a).strip().lower() for a in items if str(a).strip()]
    if col in ("bhk", "minArea", "maxArea", "minPrice", "maxPrice"):
        try:
            return int(value) if col == "bhk" else float(value)
        except (TypeError, ValueError):
            return None
    return str(value).strip() or None


def update_preferences(state: AppState) -> dict:
    updates = state.get("preference_updates", {})
    # Only touch the columns the customer actually spoke to this turn. A
    # COALESCE-everything form could never clear a preference: once a budget was
    # set, "any budget, I don't care about price" left it in force forever and
    # kept constraining every later search. Writing just the mentioned columns
    # means a key sent as null is an explicit "forget this one", while anything
    # unmentioned keeps its stored value.
    touched = [c for c in PREF_COLUMNS if c in updates]
    if not touched:
        return {}
    updates = {c: _clean_pref_value(c, updates[c]) for c in touched}

    cols = ", ".join(f'"{c}"' for c in touched)
    placeholders = ", ".join(["%s"] * len(touched))
    assignments = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in touched)
    with get_conn() as conn:
        conn.execute(
            f"""INSERT INTO {PREF_TABLE} (id, phone_number, {cols}, "updatedAt")
                VALUES (%s, %s, {placeholders}, NOW())
                ON CONFLICT (phone_number)
                DO UPDATE SET {assignments}, "updatedAt" = NOW()""",
            (str(uuid.uuid4()), state["phone_number"], *(updates[c] for c in touched)),
        )

    merged = {**state.get("preferences", {}), **updates}
    cleared = [c for c in touched if updates[c] is None]
    merged = {k: v for k, v in merged.items() if v not in (None, [])}
    logger.info(f"{_ctx(state)} update_preferences applied {updates}"
                f"{f' (cleared {cleared})' if cleared else ''} -> merged={merged}")
    return {"preferences": merged, "preference_updates": {}}


# Callers name regions, not just cities -- "anything in NCR" must widen to every
# city in that region we actually stock, not fail as an unknown place name.
REGIONS = {
    "NCR": ["Delhi", "New Delhi", "Gurugram", "Gurgaon", "Noida", "Faridabad", "Ghaziabad", "Greater Noida"],
    "MMR": ["Mumbai", "Navi Mumbai", "Thane", "Kalyan"],
    "Tri-city": ["Chandigarh", "Mohali", "Panchkula"],
}

_locations_cache: "tuple[float, str] | None" = None


def get_known_locations(ttl_sec: int = 300) -> str:
    """Cached wrapper -- this is read on every turn by both `chat` and
    `nl_to_sql`, and on a live call an extra DB round trip per turn is latency
    the caller hears as silence. Inventory locations change rarely."""
    global _locations_cache
    now = time.monotonic()
    if _locations_cache and now - _locations_cache[0] < ttl_sec:
        return _locations_cache[1]
    locations = _fetch_known_locations()
    _locations_cache = (now, locations)
    return locations


def _fetch_known_locations() -> str:
    """Real places from "Property", as "City: locality, locality" lines.

    Location used to be one free-text `area` column; it's now city + locality,
    and a caller may name either. Grouped rather than listed flat so the model
    can't paste a combined "Mumbai / Worli" string in as a column value."""
    with get_conn() as conn:
        rows = conn.execute(
            'SELECT city, string_agg(DISTINCT locality, \', \') AS localities '
            'FROM "Property" GROUP BY city ORDER BY city'
        ).fetchall()
    listing = "\n    ".join(f"{r['city']}: {r['localities'] or '-'}" for r in rows)

    # Only advertise the region cities we hold inventory in, so a region search
    # can never widen onto a city that isn't in the table.
    stocked = {r["city"] for r in rows}
    hints = [
        f"{region} covers {', '.join(present)}"
        for region, cities in REGIONS.items()
        if (present := [c for c in cities if c in stocked])
    ]
    return listing + ("\n    Regions: " + "; ".join(hints) if hints else "")


_SALE_WORDS = re.compile(r"\b(buy|buying|purchase|purchasing|own|owning|resale|for\s+sale|lakhs?|crores?)\b", re.I)
_RENT_WORDS = re.compile(r"\b(rent|rental|renting|lease|leasing|tenant|monthly|per\s+month)\b", re.I)

RENT_BUDGET_CEILING = 500_000    # a monthly rent above this is really a sale price
SALE_BUDGET_FLOOR   = 1_000_000  # a purchase price below this is really a rent figure


def _drop_mismatched_budget(search_query: str, prefs: dict, state: AppState) -> dict:
    """A saved budget records no sense of whether it was given while renting or
    buying -- preferences keeps one min/max pair for both. So when a caller
    pivots ("2bhk under 30000" then "buy a 4bhk in Mumbai"), the rent figure
    reaches the sale query as "saleAmount" <= 30000 and matches nothing. The
    prompt asks the model to ignore such budgets and it doesn't reliably, so
    the mismatch is settled here, like the visit-time hints below."""
    is_sale = bool(_SALE_WORDS.search(search_query)) and not _RENT_WORDS.search(search_query)
    cleaned = dict(prefs)
    for key in ("minPrice", "maxPrice"):
        amount = cleaned.get(key)
        if amount is None:
            continue
        if (amount >= SALE_BUDGET_FLOOR) if is_sale else (amount <= RENT_BUDGET_CEILING):
            continue
        cleaned.pop(key)
        logger.info(f"{_ctx(state)} nl_to_sql: dropped saved {key}={amount}, it can't be a "
                    f"{'purchase price' if is_sale else 'monthly rent'} for {search_query!r}")
    return cleaned


def nl_to_sql(state: AppState) -> dict:
    """LLM converts natural language + preferences to SQL."""
    prefs = _drop_mismatched_budget(state.get("search_query", ""), state.get("preferences", {}), state)
    known_locations = get_known_locations()

    prompt = """Convert a property search request into a PostgreSQL SELECT query.
    Real places in the database, as "City: its localities" (use ONLY these — never invent one):
    {known_locations}
    city and locality are two SEPARATE columns holding those plain names, so Worli is
    city = 'Mumbai' AND locality = 'Worli'. Never put a combined "Mumbai / Worli" anywhere.

    Table schema — the table and its camelCase columns are quoted identifiers, so they
    MUST keep the double quotes shown here or PostgreSQL lowercases them and the query fails:

    "Property"(id TEXT, title TEXT, "listingType" ListingType, city TEXT, locality TEXT,
               "areaSqft" FLOAT, bhk INT, "rentAmount" FLOAT, "saleAmount" FLOAT,
               furnishing FurnishingStatus, "parkingSpots" INT, amenities TEXT[],
               "isAvailable" BOOLEAN)
      "listingType" is 'RENT' or 'SALE'.
      "rentAmount" is set only on RENT rows, "saleAmount" only on SALE rows.
      furnishing is 'UNFURNISHED', 'SEMI_FURNISHED' or 'FURNISHED'.
      "areaSqft" is floor size, NOT a location.

    Search request: "{search_query}"
    User preferences (use as defaults for any missing field): {preferences}
    Preference keys: city, locality, listingType ('RENT'/'SALE'), bhk, minPrice / maxPrice
    (budget), minArea / maxArea (sqft), amenities (list), furnishing, otherPreference.
    A stored listingType decides step 1 unless this request clearly says otherwise.

    Think through this silently before answering — do not show your reasoning in the output:
    1. Renting or buying? "buy", "purchase", "price", "lakh", "crore" mean SALE; anything else
       defaults to RENT. Pick exactly ONE and write it as a single equality — omitting it mixes
       crore-priced sale flats into a rent search, and writing
       "listingType" = 'RENT' OR "listingType" = 'SALE' is worse still: AND binds tighter than
       OR, so that unparenthesized OR discards every other condition and returns the whole table.
    2. Did the user or preferences mention BHK? If yes, add "bhk = <number>". If no, skip it entirely.
    3. Did the user or preferences mention a budget? Put it on "rentAmount" for a RENT search or
       "saleAmount" for a SALE search — never the wrong one, and never rescale the number the
       customer gave. If no budget, skip it entirely. Ignore a saved budget whose size clearly
       belongs to the other mode (a 40000 budget is rent, not a purchase price).
    4. Did the user or preferences name a place? Match both columns, since a caller may say either:
       (city ILIKE '%<name>%' OR locality ILIKE '%<name>%'). If no place was named, skip it
       entirely — do not restrict to any location.
       A REGION (NCR, Delhi NCR, MMR, tri-city) is not a city — expand it to every city the
       Regions line above says it covers, OR'd together, e.g. for NCR:
       (city ILIKE '%Delhi%' OR city ILIKE '%Gurugram%')
       Older city names mean the modern one in the list: Bombay is Mumbai, Bangalore is
       Bengaluru, Calcutta is Kolkata, Madras is Chennai, Gurgaon is Gurugram. Match the
       name as it appears in the list above, never the name the caller happened to use.

    4b. Did they ask for the cheapest / most expensive / biggest / smallest? That is an
        ORDER BY, not a filter: add ORDER BY "rentAmount" ASC (or "saleAmount", or "areaSqft"
        DESC for biggest) and LIMIT 3. Never answer a superlative by filtering on a guessed
        price. "cheapest" sorts ascending; "most expensive"/"biggest" sorts descending.
    5. An amenity? amenities is an array, so use '<amenity>' = ANY(amenities), never ILIKE.
       Furnishing uses the enum; parking uses "parkingSpots" > 0.
    6. ALWAYS add "isAvailable" = TRUE — an unavailable property must never reach a caller.
    7. Combine only the conditions that apply with AND.

    ═══ FEW-SHOT EXAMPLES ═══

    Search request: "2bhk around 30k"
    Preferences: {{}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 2 AND "rentAmount" BETWEEN 25000 AND 35000 AND "isAvailable" = TRUE;

    Search request: "3bhk in Whitefield under 60000"
    Preferences: {{}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 3 AND (city ILIKE '%Whitefield%' OR locality ILIKE '%Whitefield%') AND "rentAmount" <= 60000 AND "isAvailable" = TRUE;

    Search request: "any flat in budget 20k"
    Preferences: {{"bhk": 2, "locality": "Kondapur"}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 2 AND (city ILIKE '%Kondapur%' OR locality ILIKE '%Kondapur%') AND "rentAmount" <= 20000 AND "isAvailable" = TRUE;

    Search request: "i want to buy a 4bhk in Mumbai under 5 crore"
    Preferences: {{}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'SALE' AND bhk = 4 AND (city ILIKE '%Mumbai%' OR locality ILIKE '%Mumbai%') AND "saleAmount" <= 50000000 AND "isAvailable" = TRUE;

    Search request: "a furnished place with a gym in Pune"
    Preferences: {{}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND (city ILIKE '%Pune%' OR locality ILIKE '%Pune%') AND furnishing = 'FURNISHED' AND 'gym' = ANY(amenities) AND "isAvailable" = TRUE;

    Search request: "something above 15k"
    Preferences: {{}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND "rentAmount" >= 15000 AND "isAvailable" = TRUE;

    Search request: "cheapest 2bhk options in NCR"
    Preferences: {{}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 2 AND (city ILIKE '%Delhi%' OR city ILIKE '%Gurugram%') AND "isAvailable" = TRUE ORDER BY "rentAmount" ASC LIMIT 3;

    Search request: "the most expensive place you have to buy"
    Preferences: {{}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'SALE' AND "isAvailable" = TRUE ORDER BY "saleAmount" DESC LIMIT 3;

    Search request: "flats in Bombay"
    Preferences: {{}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND (city ILIKE '%Mumbai%' OR locality ILIKE '%Mumbai%') AND "isAvailable" = TRUE;

    Search request: "just show me whatever you have"
    Preferences: {{"bhk": 3, "maxPrice": 50000}}
    SQL: SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 3 AND "rentAmount" <= 50000 AND "isAvailable" = TRUE;

    ═══ CRITICAL OUTPUT RULES ═══

    - Output ONLY a complete, valid, executable PostgreSQL query — nothing else.
    - NEVER output placeholder text like [extracted_bhk_value], <number>, [no_area_filter], or any bracketed/templated text. Every value in your output must be a real number or string, never a placeholder.
    - NEVER output comments (-- ...) inside the SQL.
    - NEVER include OR conditions like "("rentAmount" <= X OR [no_rent_filter])" — either include the real condition or omit it entirely. There is no such thing as a conditional clause in the output; you decide before writing, then write only what applies.
    - No backticks, no markdown, no explanation, no reasoning shown — just the raw SQL statement ending in a semicolon.
    - Only use the columns listed above — there is no `area` column and no `rent` column.
    - "around X" / "near X" means a ±5000 range around X for rent, unless the user gives an explicit min/max or "under/above X".
    - Spoken Indian amounts are digits: "25k" → 25000, "50 lakh" → 5000000, "1 crore" → 10000000.
    - Take every value from THIS request's search/preferences lines — the examples teach the SQL
      shape only, so never copy a bhk, budget or place out of one.

    - EVERY OR must sit inside its own parentheses, because AND binds tighter than OR.
      Right: (city ILIKE '%Delhi%' OR city ILIKE '%Gurugram%') AND bhk = 3
      Wrong: city ILIKE '%Delhi%' OR city ILIKE '%Gurugram%' AND bhk = 3
      The wrong form quietly means "every Delhi row, plus Gurugram 3BHKs" — the bhk filter
      is lost. If you cannot express something as one parenthesised group, leave it out.

    ═══ WRONG OUTPUT (never do this) ═══
    SELECT * FROM listings WHERE (bhk = [extracted_bhk_value] OR [no_bhk_filter]) AND area ILIKE '%x%';
    ↑ Wrong three times over: the table is "Property", there are no `area`/`rent` columns, and
      placeholders and OR-skip patterns must never appear in output.

    SELECT * FROM "Property" WHERE "listingType" = 'RENT' OR "listingType" = 'SALE' AND bhk = 3 AND "isAvailable" = TRUE;
    ↑ Wrong: this returns every RENT row in the table, ignoring bhk and availability entirely.
      Choose one listing type: "listingType" = 'RENT' AND bhk = 3 AND "isAvailable" = TRUE.
    """
    # schema = get_schema()   # use this to get the real schema from the DB
    prompt = prompt.format(
        search_query=state['search_query'],
        preferences=json.dumps(prefs),
        known_locations=known_locations,
    )
    t0 = time.monotonic()
    response = llm.invoke([HumanMessage(content=prompt)])
    elapsed = time.monotonic() - t0

    sql = response.content.strip().strip("```sql").strip("```").strip()

    if not sql.upper().startswith("SELECT"):
        import re
        match = re.search(r"(SELECT.+?;)", sql, re.DOTALL | re.IGNORECASE)
        sql = match.group(1) if match else 'SELECT * FROM "Property" WHERE "isAvailable" = TRUE LIMIT 5;'

    logger.info(f"{_ctx(state)} nl_to_sql ({elapsed:.2f}s) search_query={state['search_query']!r} -> {sql}")

    return {"sql_query": sql}


def query_listings(state: AppState) -> dict:
    sql = state.get("sql_query", "")
    if not sql:
        return {"search_results": []}
    try:
        t0 = time.monotonic()
        with get_conn() as conn:
            rows = conn.execute(sql).fetchall()
        elapsed = time.monotonic() - t0
        # An OR-heavy generated WHERE can match the same property twice; the
        # caller would hear it read out as two separate options.
        unique = list({r["id"]: dict(r) for r in rows}.values())
        if len(unique) != len(rows):
            logger.info(f"{_ctx(state)} query_listings dropped {len(rows) - len(unique)} duplicate row(s)")
        logger.info(f"{_ctx(state)} query_listings ran in {elapsed:.2f}s -> {len(unique)} row(s)")
        return {"search_results": unique}
    except Exception:
        logger.exception(f"{_ctx(state)} query_listings SQL error while running: {sql!r}")
        return {"search_results": []}


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _under_thousand(n: int) -> str:
    out = []
    if n >= 100:
        out.append(f"{_ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        out.append(_TENS[n // 10] + (f" {_ONES[n % 10]}" if n % 10 else ""))
    elif n:
        out.append(_ONES[n])
    return " ".join(out)


def spell_amount(n: int) -> str:
    """Indian-numbering words for a rupee figure: 4800000 -> "forty eight lakh".
    TTS reads bare digits as a digit string ("two five zero zero zero"), and the
    model still emits "25,000" on turns that skip format_reply, so the digits
    are converted here rather than relying on the prompt alone."""
    if n == 0:
        return "zero"
    parts = []
    for div, name in ((10_000_000, "crore"), (100_000, "lakh"), (1_000, "thousand")):
        if n >= div:
            q, n = divmod(n, div)
            parts.append(f"{_under_thousand(q)} {name}")
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def _price(r: dict) -> str:
    """A row carries "rentAmount" or "saleAmount" depending on "listingType",
    never both, so pick whichever is set."""
    if r.get("rentAmount"):
        return f"{spell_amount(int(r['rentAmount']))} rupees per month"
    if r.get("saleAmount"):
        return f"{spell_amount(int(r['saleAmount']))} rupees to buy"
    return "price on request"


def format_reply(state: AppState) -> dict:
    last_id = state["messages"][-1].id

    # No search happened this turn (greeting, name question, thanks, etc.)
    # -> just speak the reply `chat` already produced. Don't fabricate a
    # "no listings found" response from an empty search_results list.
    if not state.get("search_query"):
        reply = state.get("reply", "")
        logger.info(f"{_ctx(state)} format_reply: no search this turn -> {reply[:80]!r}")
        messages = state["messages"][:-1] + [AIMessage(content=reply, id=last_id)]
        return {"messages": messages}

    # A search actually ran -- compose a spoken reply from the results.
    results = state.get("search_results", [])
    logger.info(f"{_ctx(state)} format_reply: composing spoken reply from {len(results)} result(s)")

    # grab the most recent user utterance for context
    user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if results:
        listings_text = "\n".join(
            f"- {r.get('title', '')}, {r.get('locality') or r.get('city', 'Unknown')}, "
            f"{r.get('bhk', '')}BHK, {_price(r)}, {', '.join(r.get('amenities') or [])}"
            for r in results
        )
    else:
        listings_text = "No matching listings were found."

    system_prompt = SystemMessage(content="""
    You are a voice assistant. Your replies are spoken aloud through text-to-speech, never read as text. Turn the raw property search results into a short, natural, spoken-language reply.

    Rules:
    - Spell out numbers the way a person would say them aloud. "30000" becomes "thirty thousand rupees", never "3-0-0-0-0" or "30,000".
    - Use Indian numbering, the way an Indian speaker would: 5000000 is "fifty lakh", 10000000 is "one crore", 48000000 is "four crore eighty lakh". Never say "million" or "billion".
    - Each result is either to rent or to buy. Say "per month" for a rent figure and just the price for a purchase -- never call a sale price monthly rent.
    - Never use digits, commas, currency symbols, or abbreviations. No ₹. Say "per month", not "/mo".
    - Keep it conversational and concise: like someone speaking on a phone call.
    - Do not fabricate the results or hallucinate on making your own answers, Only truth is the Search results you will get
    - If there are no results, say so plainly and ask if they'd like to adjust budget or location.
    - No markdown, no bullet points, no symbols of any kind -- this is spoken, not displayed.
    """.strip())

    human_prompt = HumanMessage(content=f"""
    User asked: "{user_query}"

    Search results:
    {listings_text}

    Write the spoken reply now.
    """.strip())

    response = llm.invoke([system_prompt, human_prompt])
    reply = response.content
    logger.info(f"{_ctx(state)} format_reply -> {reply[:120]!r}{'...' if len(reply) > 120 else ''}")

    messages = state["messages"][:-1] + [AIMessage(content=reply, id=last_id)]
    return {"messages": messages}


# ── Booking / visit-confirmation flow ─────────────────────────────────────────
# Triggered once `chat` detects a FINALIZE. Runs as its own short, linear
# sub-flow (start_booking -> [later turns] -> booking_flow) that's kept
# completely separate from the main chat/parse_intent/search graph so it
# can't get tangled up in that graph's parallel fan-out.

SMTP_SENDER_EMAIL         = os.getenv("SMTP_SENDER_EMAIL", "")
SMTP_SENDER_APP_PASSWORD  = os.getenv("SMTP_SENDER_APP_PASSWORD", "")
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Voice transcription of a spoken email rarely comes back as a clean
# "user@domain.com" -- callers say "at", "dot", "dash"/"underscore" out loud,
# and STT sometimes inserts stray spaces or hyphens inside the address. This
# cleans up those common patterns before the regex above tries to match.
_EMAIL_SPOKEN_REPLACEMENTS = [
    (re.compile(r"\s+at\s+the\s+rate\s+", re.IGNORECASE), "@"),
    (re.compile(r"\s+at\s+", re.IGNORECASE), "@"),
    (re.compile(r"\s+dot\s+", re.IGNORECASE), "."),
    (re.compile(r"\s+underscore\s+", re.IGNORECASE), "_"),
    (re.compile(r"\s+(?:dash|hyphen)\s+", re.IGNORECASE), "-"),
]


def _mold_spoken_email(text: str) -> str:
    molded = text
    for pattern, replacement in _EMAIL_SPOKEN_REPLACEMENTS:
        molded = pattern.sub(replacement, molded)
    # Only collapse spaces directly touching the email's "glue" characters
    # (stray STT spacing like "gorei@ gmail. com") -- stripping every space
    # in the whole string would merge in unrelated leading words like "its".
    molded = re.sub(r"\s*([@._-])\s*", r"\1", molded)
    return molded


def describe_property(prop: dict) -> str:
    """Human-readable one-liner for a property dict, for use in replies/emails."""
    if not prop:
        return "the property you selected"
    parts = []
    if prop.get("bhk"):
        parts.append(f"{prop['bhk']}BHK")
    if prop.get("area"):
        parts.append(f"in {prop['area']}")
    if prop.get("rent"):
        parts.append(f"at ₹{prop['rent']}/mo")
    if prop.get("amenities"):
        parts.append(f"({prop['amenities']})")
    return " ".join(parts) if parts else "the property you selected"


def start_booking(state: AppState) -> dict:
    """First step of finalizing: remember which property, then ask for the
    email -- explicitly requesting it spelled out letter by letter. A
    normally-spoken email is the noisiest input in the whole flow (STT mangles
    it badly); spelled-out letters transcribe far more reliably and feed
    straight into _llm_extract_email's reconstruction. We override chat's
    free-form REPLY so this ask is deterministic instead of LLM-phrased."""
    pending = state.get("finalize_property", {})
    logger.info(f"{_ctx(state)} start_booking: customer finalized {pending} -> asking for spelled-out email")
    return {
        "pending_property" : pending,
        "booking_stage"    : "awaiting_email",
        "finalize_property": {},
        "search_query"     : "",  # force format_reply down its plain-reply path
        "reply"            : (
            f"Great choice! To send the details for {describe_property(pending)}, please spell out "
            f"your email address for me, letter by letter, so I get it exactly right. You can say "
            f"the domain, like gmail dot com, normally."
        ),
    }


def _llm_extract_email(text: str, state: AppState) -> str:
    """Last-resort fallback for spoken email formats the regex/molding
    heuristics above can't untangle (letters spelled out one at a time,
    unusual phrasing, etc). The LLM's answer is never trusted blindly --
    it's re-validated against the same email-shape regex before being
    accepted, so a confused LLM can only fail closed, never inject garbage."""
    prompt = f"""A customer said this when asked for their email address: "{text}"

    Figure out the email address they meant, correcting for how speech-to-text
    transcribes a spoken email: "at" or "at the rate" means @, "dot" means .,
    "dash"/"hyphen" means -, "underscore" means _, and letters are sometimes
    spelled out one at a time (e.g. "s h u b h a n g" means "shubhang").
    Ignore filler words like "sure", "it's", "yeah" that aren't part of the address.

    Output ONLY the email address in the form local@domain.tld -- nothing else.
    If this genuinely isn't an attempt at an email address, output exactly: UNCLEAR
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    candidate = response.content.strip()
    match = _EMAIL_RE.search(candidate)
    if match:
        logger.info(f"{_ctx(state)} extract_email: LLM fallback resolved {text!r} -> {match.group(0)}")
        return match.group(0)
    logger.warning(f"{_ctx(state)} extract_email: could not resolve {text!r} (LLM fallback said {candidate!r})")
    return ""


_SPELLED_OUT_RE = re.compile(r"(?:\b\w\b\s+){2,}\b\w\b")


def _looks_spelled_out(text: str) -> bool:
    """Detects a run of 3+ single-character tokens ('s h u b h a n g at gmail').
    The regex/molding heuristic below silently mangles these -- it grabs only
    the last spelled-out letter as the local-part -- so this pattern is
    checked first and routed straight to the LLM fallback instead."""
    return bool(_SPELLED_OUT_RE.search(text))


def extract_email(text: str, state: AppState = None) -> str:
    if state is not None and _looks_spelled_out(text):
        return _llm_extract_email(text, state)

    match = _EMAIL_RE.search(text) or _EMAIL_RE.search(_mold_spoken_email(text))
    if match:
        return match.group(0)
    if state is not None:
        return _llm_extract_email(text, state)
    return ""


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _upcoming_dates_reference(now: datetime, days: int = 14) -> str:
    """A concrete lookup table of the next N days with their real weekday
    names, so the LLM only has to *read off* the right date instead of doing
    its own day-of-week arithmetic -- exactly the kind of math LLMs get
    wrong under pressure (observed: "this Saturday" resolved to a Sunday)."""
    lines = []
    for i in range(days):
        d = now + timedelta(days=i)
        label = " (today)" if i == 0 else " (tomorrow)" if i == 1 else ""
        lines.append(f"{d.strftime('%A, %d %B %Y')} = {d.strftime('%Y-%m-%d')}{label}")
    return "\n    ".join(lines)


def _deterministic_date_hint(text: str, now: datetime) -> "datetime.date | None":
    """If the customer named an unambiguous relative day (today, tomorrow,
    day after tomorrow, or exactly one weekday name), compute the actual
    correct calendar date for it here in code, so it can be used to verify
    (and if needed, correct) whatever date the LLM comes back with. Returns
    None when the phrasing is anything else (explicit date, ambiguous, etc)
    -- in that case the LLM's date is trusted as-is."""
    t = text.lower()
    if re.search(r"\bday after tomorrow\b", t):
        return (now + timedelta(days=2)).date()
    if re.search(r"\btomorrow\b", t):
        return (now + timedelta(days=1)).date()
    if re.search(r"\btoday\b|\btonight\b", t):
        return now.date()

    mentioned = [w for w in _WEEKDAYS if re.search(rf"\b{w}\b", t)]
    if len(mentioned) == 1:
        # "next <weekday>" is genuinely ambiguous (the coming one vs next week's),
        # and the simple next-occurrence math below reliably means neither -- so
        # defer to the LLM whenever "next" is present, and only apply the hint to
        # plain "saturday"/"this saturday" where next-occurrence is unambiguous.
        if re.search(r"\bnext\b", t):
            return None
        target = _WEEKDAYS.index(mentioned[0])
        days_ahead = (target - now.weekday()) % 7
        return (now + timedelta(days=days_ahead)).date()
    return None


_TIME_AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s*m\.?\b", re.IGNORECASE)


def _deterministic_time_hint(text: str) -> "tuple | None":
    """If the customer named a clock time with an explicit am/pm ("6pm",
    "6:30 pm"), compute (hour, minute) here -- the am/pm makes it unambiguous,
    and the LLM has been observed misparsing it (returned 16:00 for "6pm").
    Times without am/pm ("6 o'clock", "at 6") stay ambiguous, so return None
    and let the LLM decide."""
    m = _TIME_AMPM_RE.search(text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    if not (1 <= hour <= 12) or minute > 59:
        return None
    is_pm = m.group(3).lower() == "p"
    if is_pm and hour != 12:
        hour += 12
    elif not is_pm and hour == 12:
        hour = 0
    return (hour, minute)


def normalize_visit_datetime(text: str, state: AppState) -> str:
    """Ask the LLM to turn a natural-language date/time into a concrete ISO
    8601 datetime. Returns "" if it can't produce one confidently.

    The LLM's date is cross-checked (and corrected if needed) against
    _deterministic_date_hint whenever the phrasing is unambiguous -- the LLM
    is only fully trusted for the time-of-day and for date phrasing that
    hint can't resolve (explicit dates like "5th Aug")."""
    now = datetime.now()
    prompt = f"""Today is {now.strftime('%A, %d %B %Y, %I:%M %p')}.
    For reference, here is the exact date of each of the next two weeks:
    {_upcoming_dates_reference(now)}

    A customer said this in response to "when would you like to visit a property": "{text}"

    Convert this into a single concrete future date and time for the visit,
    using the reference dates above to get the day-of-week exactly right --
    do not compute it yourself, look it up in the list.
    Output ONLY an ISO 8601 datetime like 2026-08-09T16:00:00 -- nothing else,
    no explanation.

    CRITICAL: only resolve a datetime if the customer actually named a day and/or
    time. If they were vague, non-committal, or didn't give one, output exactly:
    UNCLEAR
    Never invent a date or time the customer didn't say.

    Examples:
    "this saturday at 4pm"        -> the ISO datetime of the coming Saturday 16:00
    "tomorrow morning around 10"  -> tomorrow at 10:00
    "5th aug somewhere around 8pm"-> that date at 20:00
    "whenever works honestly"     -> UNCLEAR
    "sometime soon"               -> UNCLEAR
    "i'm not sure yet"            -> UNCLEAR
    "you decide"                  -> UNCLEAR
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    candidate = response.content.strip()
    try:
        resolved = datetime.fromisoformat(candidate)
    except ValueError:
        logger.warning(f"{_ctx(state)} normalize_visit_datetime: could not resolve {text!r} (LLM said {candidate!r})")
        return ""

    hint_date = _deterministic_date_hint(text, now)
    if hint_date is not None and resolved.date() != hint_date:
        logger.warning(
            f"{_ctx(state)} normalize_visit_datetime: LLM date mismatch for {text!r} -- "
            f"LLM said {resolved.date()}, correcting to {hint_date} (keeping time {resolved.time()})"
        )
        resolved = datetime.combine(hint_date, resolved.time())

    hint_time = _deterministic_time_hint(text)
    if hint_time is not None and (resolved.hour, resolved.minute) != hint_time:
        logger.warning(
            f"{_ctx(state)} normalize_visit_datetime: LLM time mismatch for {text!r} -- "
            f"LLM said {resolved.strftime('%H:%M')}, correcting to {hint_time[0]:02d}:{hint_time[1]:02d}"
        )
        resolved = resolved.replace(hour=hint_time[0], minute=hint_time[1], second=0)

    candidate = resolved.isoformat()
    logger.info(f"{_ctx(state)} normalize_visit_datetime: {text!r} -> {candidate}")
    return candidate


def _build_ics(prop: dict, customer_email: str, visit_dt: "datetime") -> str:
    """Hand-built single-VEVENT iCalendar invite (METHOD:REQUEST) so mail
    clients render it as an RSVP the customer can accept/decline and add to
    their calendar as a reminder."""
    dtstart = visit_dt.strftime("%Y%m%dT%H%M%S")
    dtend = (visit_dt + timedelta(hours=1)).strftime("%Y%m%dT%H%M%S")
    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    uid = f"{uuid.uuid4()}@property-agent"
    summary = f"Property visit: {describe_property(prop)}"
    description = f"Visit for {describe_property(prop)}".replace(",", "\\,")
    location = prop.get("area", "")

    return "\r\n".join([
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//property-agent//RSVP//EN",
        "METHOD:REQUEST",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{dtstamp}",
        f"DTSTART:{dtstart}",
        f"DTEND:{dtend}",
        f"SUMMARY:{summary}",
        f"DESCRIPTION:{description}",
        f"LOCATION:{location}",
        f"ORGANIZER;CN=Property Agent:mailto:{SMTP_SENDER_EMAIL}",
        f"ATTENDEE;CN={customer_email};RSVP=TRUE:mailto:{customer_email}",
        "STATUS:CONFIRMED",
        "SEQUENCE:0",
        "BEGIN:VALARM",
        "ACTION:DISPLAY",
        "DESCRIPTION:Property visit reminder",
        "TRIGGER:-PT1H",
        "END:VALARM",
        "END:VEVENT",
        "END:VCALENDAR",
        "",
    ])


def send_confirmation_email(prop: dict, customer_email: str, visit_dt: "datetime", state: AppState) -> bool:
    """Sends the property details + a calendar RSVP invite to the customer.
    Returns True on success, False on failure (never raises)."""
    if not SMTP_SENDER_EMAIL or not SMTP_SENDER_APP_PASSWORD:
        logger.error(f"{_ctx(state)} send_confirmation_email: SMTP_SENDER_EMAIL/SMTP_SENDER_APP_PASSWORD not configured, skipping send")
        return False

    when_text = visit_dt.strftime("%A, %d %B %Y at %I:%M %p")
    body = f"""Hi,

Here are the details of the property you finalized:

  {describe_property(prop)}

Your visit is scheduled for {when_text}.

A calendar invite is attached -- accept it to block the time on your
calendar as a reminder to visit.

See you there!
"""

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"Your property visit is confirmed -- {prop.get('area', 'your selected property')}"
    msg["From"] = SMTP_SENDER_EMAIL
    msg["To"] = customer_email
    msg.attach(MIMEText(body, "plain"))

    ics_content = _build_ics(prop, customer_email, visit_dt)
    ics_part = MIMEText(ics_content, "calendar;method=REQUEST")
    ics_part.add_header("Content-Disposition", "attachment", filename="visit.ics")
    msg.attach(ics_part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_SENDER_EMAIL, SMTP_SENDER_APP_PASSWORD)
            server.sendmail(SMTP_SENDER_EMAIL, [customer_email], msg.as_string())
        logger.info(f"{_ctx(state)} send_confirmation_email: sent to {customer_email} for visit at {when_text}")
        return True
    except Exception:
        logger.exception(f"{_ctx(state)} send_confirmation_email: failed to send to {customer_email}")
        return False


_BOOKING_CANCEL_RE = re.compile(
    r"\b(cancel|stop|exit|quit|never\s?mind|not now|later|no thanks|don'?t want|"
    r"something else|other option|more option|show me|search|different|go back)\b",
    re.IGNORECASE,
)
_AFFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|confirm(?:ed)?|go\s*ahead|sounds?\s*good|that'?s\s*right|sure|please\s*do|send\s*it)\b",
    re.IGNORECASE,
)
_NEGATIVE_RE = re.compile(r"\b(no|nope|wrong|incorrect|not\s*right)\b", re.IGNORECASE)


def booking_flow(state: AppState) -> dict:
    """Handles the turns after start_booking: collecting the email, then the
    visit date/time, then sending the confirmation email. Bypasses
    chat/parse_intent/format_reply entirely -- it builds its own reply and
    goes straight to END."""
    user_msg = state["messages"][-1].content if state["messages"] else ""
    stage = state.get("booking_stage", "")
    pending = state.get("pending_property", {})

    # Escape hatch: if the customer clearly wants out (cancel, browse more,
    # go back), release the thread instead of holding it hostage asking for
    # an email/time forever. Without this, a dangling booking_stage silently
    # hijacks every future turn of the thread away from the normal
    # chat/search/preferences flow.
    if _BOOKING_CANCEL_RE.search(user_msg) and not extract_email(user_msg):
        logger.info(f"{_ctx(state)} booking_flow: customer opted out at stage={stage!r} ({user_msg!r}) -> cancelling booking")
        reply = (
            "No problem, I've put that visit booking on hold. "
            "What would you like to do next -- keep browsing or adjust your search?"
        )
        return {
            "messages"          : [AIMessage(content=reply)],
            "booking_stage"     : "",
            "pending_property"  : {},
            "customer_email"    : "",
            "visit_datetime_iso": "",
        }

    if stage == "awaiting_email":
        email = extract_email(user_msg, state)
        if not email:
            logger.info(f"{_ctx(state)} booking_flow: no email found in {user_msg!r}, re-asking")
            reply = ("I didn't quite catch that -- could you spell out your email for me, letter by "
                     "letter? You can say the domain, like gmail dot com, normally.")
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_email"}

        logger.info(f"{_ctx(state)} booking_flow: got email {email}, asking for visit time")
        reply = (
            f"Got it, {email}. And when would you be comfortable visiting "
            f"{describe_property(pending)}? Just share a date and time that works for you."
        )
        return {
            "messages"     : [AIMessage(content=reply)],
            "booking_stage": "awaiting_visit_time",
            "customer_email": email,
        }

    if stage == "awaiting_visit_time":
        iso_text = normalize_visit_datetime(user_msg, state)
        if not iso_text:
            reply = "Sorry, I didn't catch a clear date and time -- could you tell me again, like 'this Saturday at 4pm'?"
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_visit_time"}

        when_text = datetime.fromisoformat(iso_text).strftime("%A, %d %B %Y at %I:%M %p")
        reply = (
            f"Just to confirm -- I'll send the details for {describe_property(pending)} to "
            f"{state.get('customer_email', '')} for a visit on {when_text}. Shall I go ahead and send that?"
        )
        return {
            "messages"          : [AIMessage(content=reply)],
            "booking_stage"     : "awaiting_confirmation",
            "visit_datetime_iso": iso_text,
        }

    if stage == "awaiting_confirmation":
        # The customer is correcting the email right here instead of answering
        # yes/no -- update it and ask for confirmation again with the new address.
        corrected_email = extract_email(user_msg, state)
        if corrected_email and corrected_email != state.get("customer_email", ""):
            when_text = datetime.fromisoformat(state.get("visit_datetime_iso")).strftime("%A, %d %B %Y at %I:%M %p")
            reply = f"Got it, updated to {corrected_email}. Shall I go ahead and send the visit details there for {when_text}?"
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_confirmation", "customer_email": corrected_email}

        if _NEGATIVE_RE.search(user_msg):
            logger.info(f"{_ctx(state)} booking_flow: customer said the details were wrong, re-asking for email")
            reply = ("No problem -- could you spell out the correct email, letter by letter? "
                     "You can say the domain, like gmail dot com, normally.")
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_email"}

        if not _AFFIRM_RE.search(user_msg):
            when_text = datetime.fromisoformat(state.get("visit_datetime_iso")).strftime("%A, %d %B %Y at %I:%M %p")
            reply = f"Just to confirm -- should I send the visit details to {state.get('customer_email', '')} for {when_text}?"
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_confirmation"}

        visit_dt = datetime.fromisoformat(state.get("visit_datetime_iso"))
        customer_email = state.get("customer_email", "")
        sent = send_confirmation_email(pending, customer_email, visit_dt, state)

        when_text = visit_dt.strftime("%A, %d %B %Y at %I:%M %p")
        if sent:
            reply = (
                f"All set! I've sent the property details and a calendar invite to "
                f"{customer_email} for your visit on {when_text}."
            )
        else:
            reply = (
                f"I've got everything noted for your visit on {when_text}, but ran into an "
                f"issue sending the confirmation email -- our team will follow up with you directly."
            )

        # booking done (successfully or not) -- clear the sub-flow state so a
        # future finalize on a different property starts fresh.
        return {
            "messages"          : [AIMessage(content=reply)],
            "booking_stage"     : "",
            "pending_property"  : {},
            "customer_email"    : "",
            "visit_datetime_iso": "",
        }

    # Shouldn't happen, but don't leave the caller with dead air.
    logger.error(f"{_ctx(state)} booking_flow: called with unexpected booking_stage={stage!r}")
    return {"messages": [AIMessage(content="Sorry, something went wrong with your booking -- could you try again?")], "booking_stage": ""}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_preferences(state: AppState) -> str:
    """What to do once preferences (if any) are settled for this turn."""
    if state.get("finalize_property"):
        return "start_booking"
    if state.get("search_query"):
        return "nl_to_sql"
    return "format_reply"


def route_after_parse(state: AppState) -> str:
    """Every turn takes exactly one linear path -- never a parallel fan-out --
    so format_reply always fires exactly once, and nl_to_sql/start_booking
    always see preferences already updated with this turn's mentions."""
    if state.get("preference_updates"):
        return "update_preferences"
    return route_after_preferences(state)


def route_after_load_preferences(state: AppState) -> str:
    """Mid-booking turns (we're waiting on an email or a visit time) skip the
    normal chat/parse_intent flow entirely and go straight to booking_flow."""
    if state.get("booking_stage") in ("awaiting_email", "awaiting_visit_time", "awaiting_confirmation"):
        return "booking_flow"
    return "chat"

# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph(checkpointer):
    builder = StateGraph(AppState)

    builder.add_node("load_preferences",   load_preferences)
    builder.add_node("chat",               chat)
    builder.add_node("parse_intent",       parse_intent)
    builder.add_node("update_preferences", update_preferences)
    builder.add_node("nl_to_sql",          nl_to_sql)
    builder.add_node("query_listings",     query_listings)
    builder.add_node("format_reply",       format_reply)
    builder.add_node("start_booking",      start_booking)
    builder.add_node("booking_flow",       booking_flow)

    builder.add_edge(START,                "load_preferences")

    builder.add_conditional_edges(
        "load_preferences",
        route_after_load_preferences,
        {
            "booking_flow": "booking_flow",
            "chat"        : "chat",
        }
    )
    builder.add_edge("chat",               "parse_intent")

    builder.add_conditional_edges(
        "parse_intent",
        route_after_parse,
        {
            "update_preferences": "update_preferences",
            "nl_to_sql"         : "nl_to_sql",
            "format_reply"      : "format_reply",
            "start_booking"     : "start_booking",
        }
    )
    builder.add_conditional_edges(
        "update_preferences",
        route_after_preferences,
        {
            "nl_to_sql"     : "nl_to_sql",
            "format_reply"  : "format_reply",
            "start_booking" : "start_booking",
        }
    )

    builder.add_edge("nl_to_sql",          "query_listings")
    builder.add_edge("query_listings",     "format_reply")
    builder.add_edge("start_booking",      "format_reply")
    builder.add_edge("format_reply",       END)
    builder.add_edge("booking_flow",       END)

    return builder.compile(checkpointer=checkpointer)

# ── Interactive loop ──────────────────────────────────────────────────────────

def run():
    phone, thread_id = login()

    print(f"\nPhone   : {phone}")
    print(f"Session : {thread_id}")
    print("\nType your message. Commands: 'prefs' to see saved preferences, 'quit' to exit.")
    print("─"*50)

    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("Goodbye!")
                break

            if user_input.lower() == "prefs":
                with get_conn() as conn:
                    row = conn.execute(
                        f"SELECT {', '.join(chr(34)+c+chr(34) for c in PREF_COLUMNS)} FROM {PREF_TABLE} WHERE phone_number = %s",
                        (phone,)
                    ).fetchone()
                print(f"\nSaved preferences: {dict(row) if row else 'none yet'}")
                continue

            result = graph.invoke(
                {
                    "messages"          : [HumanMessage(content=user_input)],
                    "phone_number"      : phone,
                    "thread_id"         : thread_id,
                    "preferences"       : {},
                    "preference_updates": {},
                    "search_query"      : "",
                    "sql_query"         : "",
                    "search_results"    : [],
                    "reply"             : "",
                    "finalize_property" : {},
                    # booking_stage / pending_property / customer_email are
                    # deliberately NOT reset here -- they need to persist
                    # across turns while a booking is in progress. They're
                    # only ever written by start_booking/booking_flow.
                },
                config=config,
            )

            final_reply = result["messages"][-1].content
            print(f"\nAgent: {final_reply}")

if __name__ == "__main__":
    run()
