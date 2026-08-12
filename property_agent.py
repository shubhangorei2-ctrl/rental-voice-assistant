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
import ast
import threading
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
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

# Fast path timeout. Retries multiply the wait, so a stuck primary attempt used
# to burn 3 x 15s before failing over -- unusable on a live call.
FAST_TIMEOUT_SEC = 8

# How much conversation to resend to the chat model each turn (~10 exchanges).
MAX_HISTORY_MESSAGES = 20

primary_llm = ChatMistralAI(model="mistral-large-latest", max_retries=2, timeout=LLM_TIMEOUT_SEC, temperature=0.1, api_key=API_KEY)
# open-mistral-7b was the old fallback, but it returns malformed SQL, so the
# failover path silently produced broken queries. mistral-small handles every
# mechanical task here and measured ~3x faster than large on the same prompt.
# max_retries=2 rather than 1: the Mistral API returns 429 under concurrent
# calls, and a single attempt turns a transient rate-limit into a failed turn.
# Two attempts at an 8s timeout still cap far below the old 3 x 15s.
fallback_llm = ChatMistralAI(model="mistral-small-latest", max_retries=2, timeout=FAST_TIMEOUT_SEC, temperature=0.1, api_key=API_KEY)
fast_base = ChatMistralAI(model="mistral-small-latest", max_retries=2, timeout=FAST_TIMEOUT_SEC, temperature=0.1, api_key=API_KEY)

# `llm` drives the conversation: persona, tag emission, preference extraction.
# `fast_llm` drives the mechanical steps (text -> SQL, rows -> sentence) where a
# smaller model measured as good and far quicker.
llm = primary_llm.with_fallbacks([fallback_llm])
fast_llm = fast_base.with_fallbacks([primary_llm])
logger.info(f"LLM configured | chat=mistral-large-latest fast=mistral-small-latest | DB host={DB_URI.split('@')[-1].split('/')[0] if '@' in DB_URI else DB_URI}")

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

    # Ids of properties already described to this caller. Like the booking fields
    # above, deliberately NOT reset each turn -- it has to outlive the turn to be
    # any use. Telling the prompt "never repeat a property" was not reliable, so
    # the filtering is done in code.
    shown_ids         : list

# ── DB helpers ────────────────────────────────────────────────────────────────

# One pool for the whole process. Opening a fresh connection per query cost a
# TLS handshake to Neon every time -- measured at 329ms versus 52ms through a
# warm pool, and a search turn makes three such calls. check_connection is what
# lets the pool survive Neon's idle auto-suspend: dead connections are discarded
# on checkout instead of surfacing as "the connection is closed".
_pool: "ConnectionPool | None" = None
_pool_lock = threading.Lock()


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=DB_URI,
                    min_size=1,
                    max_size=6,
                    kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
                    check=ConnectionPool.check_connection,
                    open=True,
                )
                logger.info("DB pool opened (min=1 max=6)")
    return _pool


def get_conn():
    """Borrow a pooled connection. Returns a context manager, so every existing
    `with get_conn() as conn:` call site works unchanged -- the connection is
    returned to the pool on exit rather than closed."""
    try:
        return _get_pool().connection()
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


SYSTEM_PROMPT   = """You are a property assistant on a phone call. You cover properties to rent
and to buy across major Indian cities. You are speaking, not writing.

Every city and locality we actually have, as "City: its localities":
{known_locations}
Never name a city or locality outside that list. When offering alternatives, pick from it.

═══ DECIDE WHAT THIS TURN IS ═══

1. Greeting, thanks, or small talk -> just reply warmly. Saved preferences do NOT trigger a search.
2. A question about what was said earlier -> answer from the conversation above, don't search.
3. A request to see properties -> search. Ask yourself: would an agent reach for their listings now?
   "show me flats", "what about Pune?", "anything cheaper" -> yes.
   A question about a property you already showed -- its deposit, maintenance, pet rules, lock-in,
   notice period, size or amenities -- is ALSO a search: re-search for that same property so the
   details come back from our records. Do not answer those from memory, and do not say you lack a
   detail without searching for it first. SEARCH: the deposit and maintenance for the 2BHK in
   Kondapur.
4. Did they state a preference THIS message (bhk, budget, place, amenity, furnishing, rent vs buy)?
   -> emit PREF_UPDATE with only those fields. Soft ones count: "I like Whitefield" is a locality.
5. Did they pick ONE property already shown, and is it not already booked? "I'll take that one",
   "book the Worli one" -> emit FINALIZE. "tell me more about it" or "show me others" -> NOT a pick.
   Never finalize on a goodbye.

═══ OUTPUT FORMAT ═══

REPLY: <what you say out loud>
PREF_UPDATE: <only if step 4 -- JSON, only the fields they stated>
SEARCH: <only if step 3 -- plain English describing what to look for>
FINALIZE: <only if step 5 -- JSON describing the chosen property>

REPLY is mandatory and comes first; nothing before it, nothing after the last tag. Never mention
REPLY, PREF_UPDATE, SEARCH or FINALIZE inside REPLY, and never explain what you are recording.

PREF_UPDATE keys: city, locality, listingType ("RENT"/"SALE"), bhk, minPrice, maxPrice,
minArea, maxArea, amenities (list), furnishing ("UNFURNISHED"/"SEMI_FURNISHED"/"FURNISHED"),
otherPreference. city and locality are separate: Worli is locality "Worli", city "Mumbai".
Never store a region like NCR. Budgets: "under 25k" -> maxPrice 25000, "above 10k" -> minPrice
10000, "under 1 crore" -> maxPrice 10000000, "50 lakh" -> 5000000.

Record only what the CUSTOMER chose -- if you offered "Electronic City or Whitefield?" and they
answered something else, no locality was chosen. When they DROP a preference, send that key as
null ("any budget" -> {{"minPrice": null, "maxPrice": null}}; "anywhere is fine" -> {{"city": null,
"locality": null}}); omitting it keeps the old value forever. Switching between renting and buying
clears any budget given for the other mode.

FINALIZE keys: city, locality, bhk, rent (rental) or price (purchase), amenities. In the same
REPLY, ask for their email AND ask them to spell it letter by letter, saying the domain like
gmail dot com normally -- a spoken address is the noisiest thing on a call. Ask for the email
only; don't ask for the visit date in the same breath.

═══ THE SEARCH LINE ═══

Whoever writes the SQL sees ONLY your SEARCH line, never this conversation. So restate every
constraint still in force: rent or buy, bhk, city and locality, budget, amenity, furnishing.
  they said Mumbai, then "4bhk to buy"
  BAD  -> SEARCH: 4bhk to buy        GOOD -> SEARCH: 4BHK to buy in Mumbai
Keep a region word (NCR, MMR, tri-city) in the line -- the SQL step knows its cities.

═══ SPEAKING ═══

No markdown, bullets, asterisks, headings or emoji. No digits or symbols: say "twenty five
thousand rupees a month", never "25,000" or "Rs 25000/mo". Indian scale only -- fifty lakh,
one crore, never million. Keep it to a couple of sentences, like someone on a phone.

═══ NEVER GUESS, NEVER REPEAT, ASK INSTEAD ═══

Everything you say about a property must come from a search result in THIS conversation. Those
results carry rent or price, size, furnishing, amenities and -- when we hold them -- pet rules,
deposit, maintenance, lock-in and notice period. Answer freely from those.

For anything NOT given to you -- brokerage, society bylaws, paperwork, legal or possession terms,
schools, commute, whether the landlord will negotiate -- say you don't have it in front of you and
offer to get it confirmed. Never fill the gap with something plausible: a wrong deposit or pet rule
costs the customer a wasted visit.

"Assuming" means inventing a VALUE they never gave -- a city, a budget, a deposit. Never do that.
It does NOT mean refusing to search. A search with fewer filters is perfectly good: SEARCH with
whatever you have and offer to narrow afterwards. "the cheapest 2BHK to rent" needs no city;
neither does "anything with a pool". Showing two options and asking "shall I narrow this to a
city?" beats interrogating someone before you have shown them anything.

Only ask a question when there is genuinely nothing to search on -- "I need a place", "help me
out" -- and then ask for ONE thing, never three. And ask it once: if you already asked for the
city and they didn't give one, search without it rather than asking again.

But NEVER ask for what you already have: anything in the saved preferences above, anything said
earlier in this conversation, or their email or visit time once given. Re-asking sounds like you
weren't listening.

Never describe the same property twice. If they want other options and there are none, say so
plainly rather than re-listing the one you already gave.

═══ EXAMPLES ═══

"show me 2bhk flats in Whitefield under 25000"
REPLY: Sure, let me pull up two BHK options in Whitefield under twenty five thousand.
PREF_UPDATE: {{"bhk": 2, "maxPrice": 25000, "listingType": "RENT", "city": "Bengaluru", "locality": "Whitefield"}}
SEARCH: 2BHK to rent in Whitefield, Bengaluru under 25000

after a 2BHK in Whitefield at eighteen thousand was shown: "I'll take that one, pet friendly matters"
REPLY: Great choice. Could you spell your email out for me letter by letter, and say the domain like gmail dot com normally?
PREF_UPDATE: {{"otherPreference": "pet friendly"}}
FINALIZE: {{"city": "Bengaluru", "locality": "Whitefield", "bhk": 2, "rent": 18000}}
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
    # The whole transcript used to be resent every turn, so the prompt -- and
    # with it the response time -- grew for the length of the call. A window
    # keeps it flat. Kept generous because callers do refer back ("what did I
    # say earlier?", "which two amenities did I mention?").
    history = state["messages"][-MAX_HISTORY_MESSAGES:]
    if len(state["messages"]) > MAX_HISTORY_MESSAGES:
        logger.info(f"{_ctx(state)} chat: history windowed "
                    f"{len(state['messages'])} -> {len(history)} messages")

    t0 = time.monotonic()
    response = llm.invoke([system] + history)
    elapsed = time.monotonic() - t0
    logger.info(f"{_ctx(state)} chat -> LLM responded in {elapsed:.2f}s")
    logger.debug(f"{_ctx(state)} chat raw LLM output:\n{response.content}")

    # with open("agent_debug.log", "a") as f:
    #     f.write(f"\n{'='*60}\n")
    #     f.write(f"RAW LLM OUTPUT:\n{response}\n")
    #     f.write(f"{'='*60}\n")
    return {"messages": [response]}


_ASKS_FOR_EMAIL_RE = re.compile(
    r"(?:share|spell|give|provide|send me|have)\b[^.?!]{0,50}\bemail\b"
    r"|\byour email\s*(?:address|id)?\b",
    re.IGNORECASE,
)


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

    # The model sometimes writes the email request into REPLY but forgets the
    # FINALIZE tag, so booking_stage stays empty and the caller's next utterance
    # -- their actual email -- is handled as ordinary chat and thrown away. They
    # then get asked for it a second time. Observed on a live call. If the reply
    # asks for an email, start the booking regardless of the missing tag.
    if (not finalize_property
            and not state.get("booking_stage")
            and _ASKS_FOR_EMAIL_RE.search(reply)):
        logger.warning(f"{_ctx(state)} parse_intent: REPLY asks for an email but no FINALIZE tag "
                       f"was emitted; starting the booking anyway so the answer isn't lost")
        finalize_property = {"from_reply": True}

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


def _looks_like_sale(text: str) -> bool:
    """True only when the wording clearly points at buying; an explicit rent word
    wins, so "rent to own" is not read as a purchase."""
    return bool(_SALE_WORDS.search(text)) and not _RENT_WORDS.search(text)

RENT_BUDGET_CEILING = 500_000    # a monthly rent above this is really a sale price
SALE_BUDGET_FLOOR   = 1_000_000  # a purchase price below this is really a rent figure


def _drop_mismatched_budget(search_query: str, prefs: dict, state: AppState) -> dict:
    """A saved budget records no sense of whether it was given while renting or
    buying -- preferences keeps one min/max pair for both. So when a caller
    pivots ("2bhk under 30000" then "buy a 4bhk in Mumbai"), the rent figure
    reaches the sale query as "saleAmount" <= 30000 and matches nothing. The
    prompt asks the model to ignore such budgets and it doesn't reliably, so
    the mismatch is settled here, like the visit-time hints below."""
    is_sale = _looks_like_sale(search_query)
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


def _budget_in_prefs(prefs: dict) -> bool:
    return any((prefs or {}).get(k) is not None for k in ("minPrice", "maxPrice"))


# "3bhk" contains a digit but names no budget, so bhk tokens are removed before
# looking for a figure. "cheapest" is deliberately not a budget word -- that is a
# sort, and a price filter invented alongside it is exactly what we want to drop.
_BHK_TOKEN_RE = re.compile(r"\b\d+\s*(?:bhk|bedroom|bed|rk)\b", re.IGNORECASE)
_BUDGET_HINT_RE = re.compile(
    r"\b\d{3,}\b"
    r"|\b\d+(?:\.\d+)?\s*(?:k|lakhs?|crores?|thousand|cr|l)\b"
    r"|\b(?:under|below|above|over|upto|up\s+to|within|budget|price|costing|between|around|near|about|roughly|less\s+than|more\s+than|at\s+least)\b",
    re.IGNORECASE,
)


def _budget_stated(text: str) -> bool:
    """Did the caller actually name a budget in this request?"""
    return bool(_BUDGET_HINT_RE.search(_BHK_TOKEN_RE.sub(" ", text or "")))


def _enforce_sql_guards(sql: str, search_query: str, state: AppState, prefs: dict) -> str:
    """Add back the two clauses that must never be missing, whichever model wrote
    the query. On a neutral request ("anything in Kakkanad") the model tends to
    drop "listingType" entirely, which silently mixes crore-priced sale flats
    into a rent search; and a missing "isAvailable" offers a caller a property
    that is already gone. Both are cheap to verify and too costly to get wrong,
    so they are enforced here rather than left to the prompt."""
    # A budget nobody asked for is the costliest error mode: it silently returns
    # nothing, so the caller is told there is no inventory when there is. If
    # neither the request nor the stored preferences carry a figure, no price
    # comparison belongs in the query.
    if not _budget_stated(search_query) and not _budget_in_prefs(prefs):
        stripped = re.sub(
            r'\s*AND\s*"(?:rent|sale)Amount"\s*(?:<=|>=|<|>|=)\s*\d+(?:\.\d+)?'
            r'|\s*AND\s*"(?:rent|sale)Amount"\s+BETWEEN\s+\d+(?:\.\d+)?\s+AND\s+\d+(?:\.\d+)?',
            "", sql, flags=re.IGNORECASE)
        if stripped != sql:
            logger.warning(f"{_ctx(state)} nl_to_sql: stripped an invented budget "
                           f"(no figure in request or preferences)")
            sql = stripped

    added = []
    clauses = []
    if '"listingType"' not in sql:
        mode = "SALE" if _looks_like_sale(search_query) else "RENT"
        clauses.append(f"\"listingType\" = '{mode}'")
        added.append(f"listingType={mode}")
    if '"isAvailable"' not in sql:
        clauses.append('"isAvailable" = TRUE')
        added.append("isAvailable")
    if not clauses:
        return sql

    extra = " AND ".join(clauses)
    if re.search(r"\bWHERE\b", sql, re.IGNORECASE):
        # existing conditions get parenthesised so our AND cannot be captured by
        # a top-level OR the model may have written
        sql = re.sub(r"\bWHERE\b\s*", f"WHERE {extra} AND ", sql, count=1, flags=re.IGNORECASE)
    else:
        tail = re.search(r"\b(ORDER\s+BY|LIMIT)\b", sql, re.IGNORECASE)
        if tail:
            sql = f"{sql[:tail.start()].rstrip()} WHERE {extra} {sql[tail.start():]}"
        else:
            sql = f"{sql.rstrip().rstrip(';')} WHERE {extra};"
    logger.warning(f"{_ctx(state)} nl_to_sql: model omitted {', '.join(added)}; enforced in code")
    return sql


def nl_to_sql(state: AppState) -> dict:
    """LLM converts natural language + preferences to SQL."""
    prefs = _drop_mismatched_budget(state.get("search_query", ""), state.get("preferences", {}), state)
    known_locations = get_known_locations()

    prompt = """Write ONE PostgreSQL SELECT over "Property". Output only the SQL, ending in a semicolon.

    Real places (use ONLY these, never invent one):
    {known_locations}
    city and locality are SEPARATE columns holding those plain names: Worli is
    city='Mumbai' AND locality='Worli'. Never write a combined "Mumbai / Worli".

    "Property"(id TEXT, title TEXT, "listingType" ListingType, city TEXT, locality TEXT,
               "areaSqft" FLOAT, bhk INT, "rentAmount" FLOAT, "saleAmount" FLOAT,
               furnishing FurnishingStatus, "parkingSpots" INT, amenities TEXT[],
               "isAvailable" BOOLEAN)
    The table name and every camelCase column MUST keep the double quotes shown, or
    PostgreSQL lowercases them and the query fails. "listingType" is 'RENT' or 'SALE'.
    "rentAmount" is set only on RENT rows, "saleAmount" only on SALE rows. furnishing is
    'UNFURNISHED'/'SEMI_FURNISHED'/'FURNISHED'. "areaSqft" is floor size, NOT a location.
    There is no `area` column and no `rent` column.

    Search request: "{search_query}"
    Preferences (defaults for anything the request omits): {preferences}
    Keys: city, locality, listingType, bhk, minPrice/maxPrice, minArea/maxArea, amenities,
    furnishing, otherPreference. A stored listingType decides rule 1 unless the request says otherwise.

    RULES
    1. RENT or SALE — every query MUST contain exactly one "listingType" equality, with no
       exceptions. "buy"/"purchase"/"lakh"/"crore" mean SALE. If the request mentions neither
       renting nor buying, still write "listingType" = 'RENT'. Leaving it out mixes
       crore-priced sale flats into a rent search. Never write
       "listingType" = 'RENT' OR "listingType" = 'SALE':
       AND binds tighter than OR, so an unparenthesized OR drops every other condition and
       returns the whole table. Every OR you write must sit inside its own parentheses.
    2. BHK if stated: bhk = N. Otherwise omit.
    3. Budget goes on "rentAmount" for RENT or "saleAmount" for SALE — never both, never the
       wrong one, never rescaled. Map the words to the operator exactly:
          "under/below/less than/upto/within/budget of X" -> <= X   (inclusive, never <)
          "over/above/at least/more than X"               -> >= X   (inclusive, never >)
          "around/near/about/roughly X"                   -> BETWEEN X-5000 AND X+5000
          "between X and Y"                               -> BETWEEN X AND Y
       Only "around"-type wording gets a BETWEEN. "under X" is <= X, never a range.
       Ignore a saved budget whose size belongs to the other mode (40000 is rent, not a price).
       Digits: 25k=25000, 1 lakh=100000, 50 lakh=5000000, 1 crore=10000000.
    4. A place, if named, matches both columns:
       (city ILIKE '%X%' OR locality ILIKE '%X%'). If none named, omit entirely.
       A REGION is not a city — expand to the cities the Regions line lists, OR'd inside
       parentheses, e.g. NCR -> (city ILIKE '%Delhi%' OR city ILIKE '%Gurugram%').
       Old names map to the listed one: Bombay=Mumbai, Bangalore=Bengaluru,
       Calcutta=Kolkata, Madras=Chennai, Gurgaon=Gurugram.
    5. cheapest/most expensive/biggest/smallest is an ORDER BY, not a filter:
       ORDER BY "rentAmount" ASC (or "saleAmount"/"areaSqft", DESC for most/biggest) LIMIT 3.
       Never answer a superlative by guessing a price.
    6. Amenity: '<amenity>' = ANY(amenities) — never ILIKE on it. Parking: "parkingSpots" > 0.
    7. ALWAYS add "isAvailable" = TRUE.
    8. No placeholders, no bracketed text, no SQL comments, no markdown, no explanation.
       Take every value from the request/preferences above — the examples teach shape only.

    EXAMPLES
    "2bhk under 40k" / {{}}
    SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 2 AND "rentAmount" <= 40000 AND "isAvailable" = TRUE;

    "3bhk in Whitefield under 60000" / {{}}
    SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 3 AND (city ILIKE '%Whitefield%' OR locality ILIKE '%Whitefield%') AND "rentAmount" <= 60000 AND "isAvailable" = TRUE;

    "anything in Kakkanad" / {{}}            <- says neither rent nor buy, so RENT
    SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND (city ILIKE '%Kakkanad%' OR locality ILIKE '%Kakkanad%') AND "isAvailable" = TRUE;

    "2bhk around 30k" / {{}}
    SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 2 AND "rentAmount" BETWEEN 25000 AND 35000 AND "isAvailable" = TRUE;

    "i want to buy a 4bhk in Mumbai under 5 crore" / {{}}
    SELECT * FROM "Property" WHERE "listingType" = 'SALE' AND bhk = 4 AND (city ILIKE '%Mumbai%' OR locality ILIKE '%Mumbai%') AND "saleAmount" <= 50000000 AND "isAvailable" = TRUE;

    "cheapest 2bhk options in NCR" / {{}}
    SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 2 AND (city ILIKE '%Delhi%' OR city ILIKE '%Gurugram%') AND "isAvailable" = TRUE ORDER BY "rentAmount" ASC LIMIT 3;

    "a furnished place with a gym in Pune" / {{}}
    SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND (city ILIKE '%Pune%' OR locality ILIKE '%Pune%') AND furnishing = 'FURNISHED' AND 'gym' = ANY(amenities) AND "isAvailable" = TRUE;

    "just show me whatever you have" / {{"bhk": 3, "maxPrice": 50000}}
    SELECT * FROM "Property" WHERE "listingType" = 'RENT' AND bhk = 3 AND "rentAmount" <= 50000 AND "isAvailable" = TRUE;
    """
    prompt = prompt.format(
        search_query=state['search_query'],
        preferences=json.dumps(prefs),
        known_locations=known_locations,
    )
    t0 = time.monotonic()
    response = fast_llm.invoke([HumanMessage(content=prompt)])
    elapsed = time.monotonic() - t0

    sql = response.content.strip().strip("```sql").strip("```").strip()

    if not sql.upper().startswith("SELECT"):
        import re
        match = re.search(r"(SELECT.+?;)", sql, re.DOTALL | re.IGNORECASE)
        sql = match.group(1) if match else 'SELECT * FROM "Property" WHERE "isAvailable" = TRUE LIMIT 5;'

    # `prefs` here is the sanitised set: a rent budget already dropped for a sale
    # search must not then justify keeping an invented one.
    sql = _enforce_sql_guards(sql, state.get("search_query", ""), state, prefs)

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


_WANTS_ALTERNATIVES_RE = re.compile(
    r"\b(another|other|others|else|different|more options?|anything more|"
    r"something else|next one|alternatives?)\b", re.IGNORECASE)


def _policy_facts(r: dict) -> str:
    """The handful of facts callers actually ask about, pulled out of the
    `policies` jsonb so the agent can answer them instead of guessing.

    Pet rules, deposit and maintenance were the three questions it used to
    invent answers for -- the data was in the table all along, just never put in
    front of the model. Kept to one short line per property, because everything
    added here is sent on every search turn and prompt size drives latency."""
    p = r.get("policies") or {}
    facts = []

    pet = p.get("petPolicy") or {}
    if pet.get("allowed") is True:
        kinds = ", ".join(pet.get("petTypesAllowed") or [])
        facts.append("pets allowed" + (f" ({kinds})" if kinds else ""))
    elif pet.get("allowed") is False:
        facts.append("pets not allowed")

    dep = (p.get("securityDeposit") or {}).get("amount") or r.get("depositAmount")
    if dep:
        months = (p.get("securityDeposit") or {}).get("months")
        facts.append(f"deposit {spell_amount(int(dep))} rupees"
                     + (f" ({months} months)" if months else ""))

    maint = p.get("maintenance") or {}
    amount = maint.get("monthlyAmount") or maint.get("monthlySocietyCharges")
    if amount:
        included = " included in rent" if maint.get("includedInRent") else " charged separately"
        facts.append(f"maintenance {spell_amount(int(amount))} rupees a month{included}")

    lease = p.get("leaseTerms") or {}
    if lease.get("lockInMonths"):
        facts.append(f"{lease['lockInMonths']} month lock-in")
    if lease.get("noticePeriodMonths"):
        facts.append(f"{lease['noticePeriodMonths']} month notice")

    booking = (p.get("paymentTerms") or {}).get("bookingAmount")
    if booking:
        facts.append(f"booking amount {spell_amount(int(booking))} rupees")

    return "; ".join(facts)


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
    shown = list(state.get("shown_ids") or [])

    # When the caller explicitly wants something OTHER than what they've heard,
    # drop what they've already been told about. Without this the agent re-reads
    # the same property back as if it were a new option.
    user_turn = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_turn = msg.content
            break
    if _WANTS_ALTERNATIVES_RE.search(user_turn or "") and shown:
        fresh = [r for r in results if r.get("id") not in shown]
        if not fresh and results:
            reply = ("That's actually the only match I have for what you've asked for. "
                     "Would you like me to widen the budget or try another area?")
            logger.info(f"{_ctx(state)} format_reply: all {len(results)} result(s) already "
                        f"described; telling the caller there are no others")
            last_id = state["messages"][-1].id
            return {"messages": state["messages"][:-1] + [AIMessage(content=reply, id=last_id)]}
        results = fresh or results
    logger.info(f"{_ctx(state)} format_reply: composing spoken reply from {len(results)} result(s)")

    # grab the most recent user utterance for context
    user_query = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if not results:
        # Handing "no matches" to a model invites it to fill the silence: on a
        # live turn it invented a whole property, "Green Meadows", with a made-up
        # size and rent. There is nothing to phrase here, so nothing is asked.
        reply = ("I couldn't find anything matching that. Would you like me to widen the "
                 "budget, try a different area, or change the number of bedrooms?")
        logger.info(f"{_ctx(state)} format_reply: 0 results -> fixed reply, LLM not called")
        last_id = state["messages"][-1].id
        return {"messages": state["messages"][:-1] + [AIMessage(content=reply, id=last_id)]}

    if results:
        listings_text = "\n".join(
            f"- {r.get('title', '')}, {r.get('locality') or r.get('city', 'Unknown')}, "
            f"{r.get('bhk', '')}BHK, {_price(r)}, {spell_amount(int(r['areaSqft']))} square feet, "
            f"{str(r.get('furnishing') or '').replace('_', ' ').lower()}, "
            f"amenities: {', '.join(r.get('amenities') or []) or 'none listed'}"
            + (f" | {_policy_facts(r)}" if _policy_facts(r) else "")
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
    - The results may carry pet rules, deposit, maintenance, lock-in and notice period after a "|". Use them if the customer asked about those; otherwise leave them out rather than reciting everything.
    - State no number, amenity or policy that is not in the results above. If the customer asked something the results don't cover, say you'll get it confirmed.
    - Every figure is already written out in words for you. Copy it EXACTLY as given -- "twenty two thousand rupees per month" is not "two thousand two hundred". Never restate a number in your own words or re-do the arithmetic.
    - If the results repeat a property the customer has already been told about and they asked for something different, say there are no other matches instead of describing it again.
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
    return {"messages": messages,
            "shown_ids": shown + [r["id"] for r in results if r.get("id") not in shown]}


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


def _amenity_words(value) -> str:
    """Amenities as speakable words. The FINALIZE tag sometimes carries a real
    list and sometimes a stringified one, and the raw value was interpolated
    straight into speech -- callers heard "(['clubhouse', 'Wi-Fi'])" complete
    with brackets and quotes."""
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                # literal_eval copes with the apostrophe in "children's play area",
                # which a naive quote-matching regex truncates to "children"
                parsed = ast.literal_eval(text)
                items = [str(v) for v in parsed] if isinstance(parsed, (list, tuple)) else [str(parsed)]
            except (ValueError, SyntaxError):
                items = [p.strip(" '\"") for p in text.strip("[]").split(",")]
        else:
            items = [text]
    return ", ".join(i.strip().strip("'\"") for i in items if i.strip())


def describe_property(prop: dict) -> str:
    """Human-readable one-liner for a property dict, for use in replies/emails."""
    if not prop:
        return "the property you selected"
    parts = []
    if prop.get("bhk"):
        parts.append(f"{prop['bhk']}BHK")
    where = prop.get("locality") or prop.get("city") or prop.get("area")
    if where:
        parts.append(f"in {where}")
    # a rental carries `rent`, a purchase carries `price` -- never both
    if prop.get("rent"):
        parts.append(f"at ₹{prop['rent']}/mo")
    elif prop.get("price"):
        parts.append(f"at ₹{prop['price']}")
    amenities = _amenity_words(prop.get("amenities"))
    if amenities:
        parts.append(f"({amenities})")
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


# Single characters separated by spaces OR the hyphens/dots that speech-to-text
# inserts when a caller spells: "S-H-U-B-H-A-N-G-O-R. E. I". The space-only form
# missed the hyphenated one, and the molding heuristic then produced
# "S-H-U-B-H-A-N-G-O-R.E.I@gmail.com" -- spoken back to the caller as garbage.
_SPELLED_OUT_RE = re.compile(
    # letters given one at a time: "s h u b" / "S-H-U-B" / "S. H. U. B"
    r"(?:\b\w\b[\s.\-]+){2,}\b\w\b"
    # or dictated in chunks, which STT returns hyphen-joined and usually capitalised:
    # "SHU-BHANG-OREI"
    r"|\b[A-Za-z]{2,8}(?:-[A-Za-z]{2,8}){1,}\b(?=[^\s]*@|\s|$)"
)

# Strict shape for an address we are willing to send mail to.
_EMAIL_STRICT_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9._%+-]*[a-z0-9])?@[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z]{2,})+$"
)

# Words that mean "this address is wrong" -- checked before any attempt to read a
# new address out of the same sentence.
_EMAIL_REJECT_RE = re.compile(
    r"\b(?:not|isn'?t|ain'?t)\s+(?:correct|right|it|quite\s+right)\b"
    r"|\b(?:in)?correct(?:ly)?\b|\bwrong\b|\bmistake\b|\btypo\b|\bmisspell\w*\b"
    r"|\bthat'?s\s+not\b|\bnope\b",
    re.IGNORECASE,
)


def _looks_spelled_out(text: str) -> bool:
    """A run of 3+ single-character tokens, however STT chose to separate them."""
    return bool(_SPELLED_OUT_RE.search(text))


def _mentions_an_address(text: str) -> bool:
    """Is the caller plausibly giving an address at all? Without this check the
    LLM fallback is handed sentences like "but the email is not correct" and
    happily invents a well-formed address that was never said."""
    if "@" in text or _looks_spelled_out(text):
        return True
    return bool(re.search(r"\b(at the rate|at|dot|gmail|yahoo|outlook|hotmail|icloud|rediff|proton)\b",
                          text, re.IGNORECASE))


def _supported_by_text(email: str, text: str) -> bool:
    """Every character of the address must appear, in order, in what the caller
    actually said. Speech gives us the letters one way or another -- spelled,
    run together, or with "dot"/"at" spoken -- so a genuine address always
    survives this, while an invented one ("shubhang.singh@domain.com" from a
    sentence containing no such letters) cannot."""
    needle = re.sub(r"[^a-z0-9]", "", email.lower())
    hay = re.sub(r"[^a-z0-9]", "", text.lower())
    pos = 0
    for ch in needle:
        pos = hay.find(ch, pos)
        if pos < 0:
            return False
        pos += 1
    return True


_SAID_A_SEPARATOR_RE = re.compile(r"\b(dash|hyphen|underscore|minus)\b", re.IGNORECASE)


def _local_part_as_spoken(text: str) -> str:
    """The portion of the caller's sentence before the "@" / "at" -- i.e. what
    they said for the local part, so separators claimed there can be checked
    without the domain's own "gmail dot com" confusing matters."""
    split = re.split(r"@|\bat the rate\b|\bat\b", text or "", maxsplit=1, flags=re.IGNORECASE)
    return split[0]


def _strip_dictation_separators(email: str, text: str) -> str:
    """Speech-to-text chops a dictated address into chunks and rejoins them with
    punctuation the caller never uttered: "SHU-BHANG-OREI@gmail.com" and
    "shubhang.orei@gmail.com" are both really shubhangorei@gmail.com. On a call a
    genuine separator is spoken -- "dash", "hyphen", "dot" -- so a separator in
    the local part is kept only if the corresponding word was actually said
    before the "at". The domain is left alone; "gmail dot com" is unambiguous."""
    local, at, domain = email.partition("@")
    if not at:
        return email
    spoken = _local_part_as_spoken(text)
    if "-" in local and not _SAID_A_SEPARATOR_RE.search(spoken):
        local = local.replace("-", "")
    # A dot is only an artefact when the address was dictated. A literal, already
    # lowercase "john.doe@gmail.com" that STT recognised as one token is left
    # alone -- stripping that would break a perfectly good address.
    dictated = bool(_looks_spelled_out(text)) or re.search(r"[A-Z]{2,}", text or "")
    if ("." in local and dictated
            and not re.search(r"\b(dot|point|period)\b", spoken, re.IGNORECASE)):
        local = local.replace(".", "")
    return local + "@" + domain


def _accept_email(candidate: str, text: str, state: AppState, how: str) -> str:
    """Normalise, then refuse anything malformed or unsupported by the caller."""
    email = (candidate or "").strip().strip(".,;:").lower()
    normalised = _strip_dictation_separators(email, text)
    if normalised != email:
        logger.info(f"{_ctx(state)} extract_email: dropped dictation separators, "
                    f"{email!r} -> {normalised!r} (caller never spoke them)")
        email = normalised
    if not email or not _EMAIL_STRICT_RE.match(email):
        logger.warning(f"{_ctx(state)} extract_email: rejected {candidate!r} from {how} (bad shape)")
        return ""
    if not _supported_by_text(email, text):
        logger.warning(f"{_ctx(state)} extract_email: rejected {email!r} from {how} -- "
                       f"its characters are not in what the caller said ({text!r})")
        return ""
    return email


def extract_email(text: str, state: AppState = None) -> str:
    if not _mentions_an_address(text):
        return ""

    if state is not None and _looks_spelled_out(text):
        return _accept_email(_llm_extract_email(text, state), text, state, "llm/spelled")

    match = _EMAIL_RE.search(text) or _EMAIL_RE.search(_mold_spoken_email(text))
    if match:
        got = _accept_email(match.group(0), text, state or {}, "regex")
        if got:
            return got
    if state is not None:
        return _accept_email(_llm_extract_email(text, state), text, state, "llm")
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


def _spoken_when(dt: "datetime") -> str:
    """A visit time as a person would say it. The four-digit year used to reach
    clean_for_voice, which spells any 4+ digit run as a rupee figure -- the
    caller heard "Saturday, 15 August two thousand twenty six at 06:00 PM"."""
    hour = dt.strftime("%I").lstrip("0") or "12"
    minutes = "" if dt.minute == 0 else f" {dt.strftime('%M')}"
    return f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B')} at {hour}{minutes} {dt.strftime('%p')}"


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

        # If the visit time is already settled -- the usual case when the caller
        # is correcting a mis-heard address -- don't make them say it twice.
        already = state.get("visit_datetime_iso")
        if already:
            when_text = _spoken_when(datetime.fromisoformat(already))
            logger.info(f"{_ctx(state)} booking_flow: got corrected email {email}, "
                        f"visit time already known -> straight to confirmation")
            reply = (f"Got it, {email}. So that's a visit to {describe_property(pending)} on "
                     f"{when_text}. Shall I send the details there?")
            return {
                "messages"      : [AIMessage(content=reply)],
                "booking_stage" : "awaiting_confirmation",
                "customer_email": email,
            }

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

        when_text = _spoken_when(datetime.fromisoformat(iso_text))
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
        # "Yeah sure, but the email is not correct" contains an affirmation AND a
        # rejection. The complaint has to win, and it has to be tested BEFORE we
        # try to read a new address out of the same sentence -- otherwise the
        # sentence gets mined for an address it never contained.
        if _EMAIL_REJECT_RE.search(user_msg):
            logger.info(f"{_ctx(state)} booking_flow: customer says the address is wrong "
                        f"({user_msg!r}) -> back to awaiting_email")
            reply = ("No problem, let's fix that. Could you spell your email out for me letter "
                     "by letter? Say the domain, like gmail dot com, normally.")
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_email",
                    "customer_email": ""}

        # The customer is correcting the email right here instead of answering
        # yes/no -- update it and ask for confirmation again with the new address.
        corrected_email = extract_email(user_msg, state)
        if corrected_email and corrected_email != state.get("customer_email", ""):
            when_text = _spoken_when(datetime.fromisoformat(state.get("visit_datetime_iso")))
            reply = f"Got it, updated to {corrected_email}. Shall I go ahead and send the visit details there for {when_text}?"
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_confirmation", "customer_email": corrected_email}

        if _NEGATIVE_RE.search(user_msg):
            logger.info(f"{_ctx(state)} booking_flow: customer said the details were wrong, re-asking for email")
            reply = ("No problem -- could you spell out the correct email, letter by letter? "
                     "You can say the domain, like gmail dot com, normally.")
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_email",
                    "customer_email": ""}

        if not _AFFIRM_RE.search(user_msg):
            when_text = _spoken_when(datetime.fromisoformat(state.get("visit_datetime_iso")))
            reply = f"Just to confirm -- should I send the visit details to {state.get('customer_email', '')} for {when_text}?"
            return {"messages": [AIMessage(content=reply)], "booking_stage": "awaiting_confirmation"}

        visit_dt = datetime.fromisoformat(state.get("visit_datetime_iso"))
        customer_email = state.get("customer_email", "")
        sent = send_confirmation_email(pending, customer_email, visit_dt, state)

        when_text = _spoken_when(visit_dt)
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
