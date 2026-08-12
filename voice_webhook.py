"""
VAPI Voice Webhook
==================

Run:
    uvicorn voice_webhook:app --port 8000
"""

import os
import time
import uuid
import json
import re
import asyncio
import threading
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

# import all nodes and state from your existing file
# (this import also runs property_agent's logging.basicConfig(), so the
# handlers/format configured there apply to this module's logger too)
from property_agent import (
    AppState,
    spell_amount,
    load_preferences,
    chat,
    parse_intent,
    update_preferences,
    nl_to_sql,
    query_listings,
    format_reply,
    route_after_parse,
    route_after_preferences,
    route_after_load_preferences,
    start_booking,
    booking_flow,
    get_conn,
    DB_URI,
)

logger = logging.getLogger("property_agent.webhook")

# ── Global definitions (populated during lifespan) ───────────────────────────
checkpointer = None
graph = None
db_pool = None

# ── Lifespan Context Manager ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, graph, db_pool

    logger.info("Startup: opening checkpointer connection pool and compiling graph...")
    # A single long-lived connection (the old PostgresSaver.from_conn_string
    # approach) silently dies on Neon's idle auto-suspend or any network blip
    # and never reconnects -- every call then fails with "the connection is
    # closed" until the process is restarted (seen in production). A pool with
    # check=check_connection health-checks each connection on checkout and
    # transparently discards/replaces dead ones, so the webhook self-heals.
    # kwargs match what from_conn_string sets, which PostgresSaver requires.
    db_pool = ConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        check=ConnectionPool.check_connection,
        open=False,
    )
    db_pool.open()
    checkpointer = PostgresSaver(db_pool)
    checkpointer.setup()

    # interrupt_before "nl_to_sql" lets _sse_stream_with_filler peek at
    # whether a turn is about to do an actual property search (the one slow,
    # multi-LLM-call path) and only then inject a "let me check" filler --
    # every other turn (chat-only, preference-only, finalize, booking_flow)
    # finishes fast enough on its own and gets no filler. This only affects
    # this webhook's graph instance; property_agent.build_graph() (used by
    # the CLI) is untouched.
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["nl_to_sql"])
    logger.info("Startup complete: webhook ready to accept calls")

    yield

    logger.info("Shutting down: closing checkpointer connection pool")
    db_pool.close()

app = FastAPI(lifespan=lifespan)

# ── Build graph structure ─────────────────────────────────────────────────────
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

# ── In-memory call registry ───────────────────────────────────────────────────
# maps call_id → {phone_number, thread_id}
active_calls: dict = {}


# 4+ digit runs, or comma-grouped figures -- but never something glued to a
# letter, so "2BHK" and "3BHK" survive untouched.
_AMOUNT_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{2,3})+|\d{4,})(?!\.?\d)(?![A-Za-z])")


_BHK_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}


def clean_for_voice(reply: str) -> str:
    # "4BHK" is read out as "four B-H-K" by TTS; say it the way a person would.
    reply = re.sub(r"\b(\d+)\s*BHK\b",
                   lambda m: f"{_BHK_WORDS.get(int(m.group(1)), m.group(1))} bedroom",
                   reply, flags=re.IGNORECASE)
    # the model also writes it as "four BHK", where the digit rule doesn't apply
    reply = re.sub(r"\bBHK\b", "bedroom", reply)
    reply = re.sub(r"\*+", "", reply)
    reply = re.sub(r"#+\s*", "", reply)
    # Markdown list markers read aloud as "dash" / "one dot"; strip them before
    # newlines become sentence breaks below.
    reply = re.sub(r"^\s*(?:[-•]|\d+\.)\s+", "", reply, flags=re.MULTILINE)
    reply = re.sub(r"\n+", ". ", reply)
    reply = re.sub(r"₹", "rupees ", reply)
    # "/month" must not be rewritten by the "/mo" rule -- that produced
    # "per monthnth" in live replies. Longest form first, then bare "/mo".
    reply = re.sub(r"/month\b", " per month", reply, flags=re.IGNORECASE)
    reply = re.sub(r"/mo\b", " per month", reply, flags=re.IGNORECASE)
    reply = _AMOUNT_RE.sub(lambda m: spell_amount(int(m.group(1).replace(",", ""))), reply)
    return re.sub(r"\s{2,}", " ", reply).strip()


def get_or_create_call(call_id: str, phone_number: str) -> dict:
    """Fetch existing call session, or create one on the fly (covers the
    case where our lifecycle webhook didn't fire before the first turn)."""
    call_data = active_calls.get(call_id)
    if call_data:
        return call_data

    thread_id = str(uuid.uuid4())
    call_data = {"phone_number": phone_number, "thread_id": thread_id, "started_at": time.time()}
    active_calls[call_id] = call_data

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO users (phone_number)
            VALUES (%s) ON CONFLICT (phone_number) DO NOTHING
        """, (phone_number,))
        conn.execute("""
            INSERT INTO preferences (phone_number)
            VALUES (%s) ON CONFLICT (phone_number) DO NOTHING
        """, (phone_number,))
        conn.execute("""
            INSERT INTO sessions (thread_id, phone_number)
            VALUES (%s, %s)
        """, (thread_id, phone_number))

    logger.info(f"[call={call_id}] new call session created: phone={phone_number} thread={thread_id}")
    return call_data


def _build_initial_state(phone_number: str, thread_id: str, transcript: str) -> dict:
    return {
        "messages"          : [HumanMessage(content=transcript)],
        "phone_number"      : phone_number,
        "thread_id"         : thread_id,
        "preferences"       : {},
        "preference_updates": {},
        "search_query"      : "",
        "sql_query"         : "",
        "search_results"    : [],
        "reply"             : "",
        "finalize_property" : {},
        # booking_stage / pending_property / customer_email are deliberately
        # NOT reset here -- see property_agent.run() for why.
    }


def _invoke_to_completion(initial_state: dict, config: dict) -> dict:
    """Runs the graph, automatically resuming past the nl_to_sql interrupt
    (see lifespan()) until the turn is fully done. Callers that don't care
    about the intermediate pause -- i.e. everyone except the filler stream --
    can just call this and get the same final state as a plain invoke()."""
    result = graph.invoke(initial_state, config=config)
    while graph.get_state(config).next:
        result = graph.invoke(None, config=config)
    return result


def _run_graph(phone_number: str, thread_id: str, transcript: str) -> str:
    """Runs the LangGraph pipeline and returns a voice-cleaned reply.
    Never raises -- returns a spoken fallback message on error so Vapi
    always gets something to say instead of dead air."""
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.monotonic()
    try:
        result = _invoke_to_completion(_build_initial_state(phone_number, thread_id, transcript), config)
        reply = clean_for_voice(result["messages"][-1].content)
        elapsed = time.monotonic() - t0
        logger.info(f"[phone={phone_number} thread={thread_id[:8]}] graph.invoke completed in {elapsed:.2f}s -> {reply[:120]!r}")
        return reply
    except Exception:
        elapsed = time.monotonic() - t0
        logger.exception(
            f"[phone={phone_number} thread={thread_id[:8]}] graph.invoke FAILED after {elapsed:.2f}s "
            f"for transcript={transcript!r}"
        )
        return "Sorry, I ran into an issue looking that up. Could you say that again?"


# One might assume a voice call enforces turn-taking on its own -- but it
# doesn't: observed live, when a reply is slow, Vapi re-sends the growing
# transcript (original question + "hello?" + "is someone there?") as a NEW
# overlapping /chat/completions request for the *same* call, while the first
# one is still stuck in a slow LLM call. That runs the graph concurrently
# against the same thread from two directions, so a slow turn's stale update
# can overwrite a faster one's progress. One lock per phone number forces a
# given customer's turns to process strictly in order, while different
# customers stay fully concurrent.
_conversation_locks: dict = {}


def _get_conversation_lock(phone_number: str) -> asyncio.Lock:
    lock = _conversation_locks.get(phone_number)
    if lock is None:
        lock = asyncio.Lock()
        _conversation_locks[phone_number] = lock
    return lock


# ── THE ENDPOINT VAPI ACTUALLY CALLS FOR CONVERSATION ─────────────────────────
# Set the "Custom LLM URL" in the Vapi dashboard to your ngrok base URL
# (e.g. https://partition-resurrect-outbid.ngrok-free.dev). Vapi appends
# "/chat/completions" itself.
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    logger.debug(f"[CUSTOM LLM REQUEST]:\n{json.dumps(body, indent=2)}")

    messages_list = body.get("messages", [])
    call = body.get("call", {})
    call_id = call.get("id", "unknown")
    phone_number = call.get("customer", {}).get("number", "unknown").lstrip("+")
    wants_stream = body.get("stream", True)  # Vapi sends stream=true by default

    # last user turn
    transcript = ""
    for msg in reversed(messages_list):
        if msg.get("role") == "user":
            transcript = (msg.get("content") or "").strip()
            break

    if not transcript:
        logger.info(f"[call={call_id} phone={phone_number}] empty transcript, replying with nothing")
        if wants_stream:
            return StreamingResponse(_sse_stream(call_id, ""), media_type="text/event-stream")
        return JSONResponse(_completion_payload(call_id, ""))

    call_data = get_or_create_call(call_id, phone_number)
    phone_number = call_data["phone_number"]
    thread_id = call_data["thread_id"]
    logger.info(f"[call={call_id} phone={phone_number} thread={thread_id[:8]}] user said: {transcript!r} (stream={wants_stream})")

    if wants_stream:
        return StreamingResponse(
            _sse_stream_with_filler(call_id, phone_number, thread_id, transcript),
            media_type="text/event-stream",
        )

    # Non-streaming path, only used if Vapi ever sends {"stream": false}
    async with _get_conversation_lock(phone_number):
        reply = await asyncio.to_thread(_run_graph, phone_number, thread_id, transcript)
    return JSONResponse(_completion_payload(call_id, reply))


# ── Token streaming ───────────────────────────────────────────────────────────
# Nothing used to reach Vapi until the whole turn had finished computing, so the
# caller heard silence for the entire turn. Forwarding tokens as they are
# generated lets Vapi's TTS start on the first few words.

_TAG_RE = re.compile(r"(PREF_UPDATE|SEARCH|FINALIZE)\s*:", re.IGNORECASE)
_REPLY_PREFIX_RE = re.compile(r"^\s*REPLY\s*:\s*", re.IGNORECASE)
_TAG_HOLDBACK = 14  # longest tag name + colon, so a marker is never half-emitted


class _ReplyGate:
    """Exposes only the REPLY part of chat's tagged output, incrementally.

    chat emits "REPLY: ... PREF_UPDATE: {...} SEARCH: ...". Streaming that raw
    would have TTS read the tag names to the caller -- the same bug already
    fixed once in parse_intent. Text is released only once it cannot still turn
    out to be the start of a tag marker."""

    def __init__(self):
        self._buf = ""
        self._sent = 0
        self.closed = False

    def feed(self, token: str) -> str:
        if self.closed:
            return ""
        self._buf += token
        text = _REPLY_PREFIX_RE.sub("", self._buf)
        match = _TAG_RE.search(text)
        if match:
            text = text[:match.start()]
            self.closed = True
        safe = text if self.closed else text[:max(0, len(text) - _TAG_HOLDBACK)]
        out = safe[self._sent:]
        self._sent += len(out)
        return out


def _split_sentences(buf: str):
    """Peel off complete sentences. clean_for_voice rewrites whole figures
    ("25,000" -> "twenty five thousand"), so it must never see a number split
    across two token boundaries -- flushing on sentence ends guarantees that."""
    done = []
    while True:
        m = re.search(r"[.!?](?=\s|$)", buf)
        if not m:
            break
        done.append(buf[:m.end()])
        buf = buf[m.end():].lstrip()
    return done, buf


async def _graph_token_stream(phone_number: str, thread_id: str, transcript: str, config: dict):
    """Bridge LangGraph's synchronous token stream onto the event loop.

    The checkpointer is the sync PostgresSaver, so graph.stream() must run in a
    worker thread; tokens are handed back through a queue. Yields
    (node_name, text) pairs, and resumes past the nl_to_sql interrupt itself."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    done = object()

    def pump():
        try:
            def forward(payload):
                for chunk, meta in graph.stream(payload, config, stream_mode="messages"):
                    text = getattr(chunk, "content", "") or ""
                    if text:
                        loop.call_soon_threadsafe(
                            queue.put_nowait, (meta.get("langgraph_node", ""), text))

            forward(_build_initial_state(phone_number, thread_id, transcript))
            while graph.get_state(config).next:      # paused before nl_to_sql
                forward(None)
        except Exception as exc:                      # surfaced to the caller below
            loop.call_soon_threadsafe(queue.put_nowait, ("__error__", repr(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    threading.Thread(target=pump, daemon=True).start()
    while True:
        item = await queue.get()
        if item is done:
            return
        yield item


async def _sse_stream_with_filler(call_id: str, phone_number: str, thread_id: str, transcript: str):
    """Streams a reply, injecting a filler line ONLY when this turn is about
    to run an actual property search (chat -> parse_intent -> nl_to_sql ->
    query_listings -> format_reply -- the one slow, multi-LLM-call path).
    Every other turn (plain chat, a preference-only update, a finalize, or a
    booking_flow email/date-collection turn) finishes fast enough on its own,
    so it gets no filler and just speaks its real reply immediately -- saying
    "let me check that for you" before something that isn't a check would be
    misleading, and tacking it onto an already-fast reply just adds a beat of
    unnecessary silence.

    This relies on the graph being compiled with interrupt_before=["nl_to_sql"]
    (see lifespan()): invoking it either runs a fast turn all the way to
    completion in one go, or pauses right before nl_to_sql, which is exactly
    the signal to speak the filler before resuming the slow part.

    <flush /> tells Vapi to speak the filler immediately rather than buffer
    it while waiting for more text to arrive.
    """
    created = int(time.time())
    chunk_id = f"chatcmpl-{call_id}"
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.monotonic()
    first_at = None
    spoken = []
    gate = _ReplyGate()
    pending = ""      # partial sentence not yet safe to voice-clean
    failed = False

    def delta(text):
        return f"data: {json.dumps(_delta_chunk(chunk_id, created, text))}\n\n"

    # See _get_conversation_lock: Vapi can fire an overlapping request for
    # this same call if the previous turn is slow to reply, so this turn
    # waits its turn instead of racing the graph concurrently.
    async with _get_conversation_lock(phone_number):
        try:
            async for node, text in _graph_token_stream(phone_number, thread_id, transcript, config):
                if node == "__error__":
                    raise RuntimeError(text)

                # chat's output carries the internal tags, so it goes through the
                # gate. format_reply's output is already the caller-facing reply.
                if node == "chat":
                    text = gate.feed(text)
                elif node != "format_reply":
                    continue          # other nodes emit no caller-facing text
                if not text:
                    continue

                sentences, pending = _split_sentences(pending + text)
                for sentence in sentences:
                    cleaned = clean_for_voice(sentence)
                    if not cleaned:
                        continue
                    if first_at is None:
                        first_at = time.monotonic() - t0
                        # tell Vapi to start speaking rather than buffer for more
                        cleaned += "<flush />"
                    spoken.append(cleaned)
                    yield delta(cleaned)

            if pending.strip():
                cleaned = clean_for_voice(pending)
                if cleaned:
                    if first_at is None:
                        first_at = time.monotonic() - t0
                    spoken.append(cleaned)
                    yield delta(cleaned)
        except Exception:
            failed = True
            logger.exception(
                f"[phone={phone_number} thread={thread_id[:8]}] streaming FAILED after "
                f"{time.monotonic() - t0:.2f}s for transcript={transcript!r}"
            )

        # Nodes that build their reply in code rather than from an LLM
        # (start_booking, booking_flow) stream no tokens at all, so fall back to
        # whatever the finished state holds. Also covers a mid-stream failure.
        if not spoken:
            try:
                final = (await asyncio.to_thread(graph.get_state, config)).values
                reply = clean_for_voice(final["messages"][-1].content)
            except Exception:
                logger.exception(f"[phone={phone_number}] could not read final state")
                reply = "Sorry, I ran into an issue looking that up. Could you say that again?"
            if first_at is None:
                first_at = time.monotonic() - t0
            spoken.append(reply)
            yield delta(reply)

    elapsed = time.monotonic() - t0
    logger.info(
        f"[phone={phone_number} thread={thread_id[:8]}] turn done in {elapsed:.2f}s "
        f"(first audio at {first_at:.2f}s){' [FAILED]' if failed else ''} "
        f"-> {' '.join(spoken)[:120]!r}"
    )

    yield f"data: {json.dumps(_final_chunk(chunk_id, created))}\n\n"
    yield "data: [DONE]\n\n"


def _sse_stream(call_id: str, content: str):
    """Simple single-shot SSE stream (used only for the empty-transcript case)."""
    created = int(time.time())
    chunk_id = f"chatcmpl-{call_id}"
    yield f"data: {json.dumps(_delta_chunk(chunk_id, created, content))}\n\n"
    yield f"data: {json.dumps(_final_chunk(chunk_id, created))}\n\n"
    yield "data: [DONE]\n\n"


def _delta_chunk(chunk_id: str, created: int, content: str) -> dict:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "property-agent",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}
        ],
    }


def _final_chunk(chunk_id: str, created: int) -> dict:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "property-agent",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }


def _completion_payload(call_id: str, content: str) -> dict:
    """Build a non-streamed OpenAI-compatible chat completion object.
    Only used when a request explicitly sends {"stream": false}."""
    return {
        "id": f"chatcmpl-{call_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "property-agent",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


# ── Server URL webhook (call lifecycle only) ──────────────────────────────────
# Configure this SEPARATELY under Assistant -> Advanced -> Server URL.
# This is NOT the Custom LLM URL.
@app.post("/")
async def server_events(request: Request):
    body = await request.json()
    logger.debug(f"[SERVER EVENT]:\n{json.dumps(body, indent=2)}")
    message  = body.get("message", {})
    msg_type = message.get("type", "")
    call     = message.get("call", {})
    call_id  = call.get("id", "unknown")
    phone_number = call.get("customer", {}).get("number", "unknown").lstrip("+")

    if msg_type == "status-update" and message.get("status") == "in-progress":
        get_or_create_call(call_id, phone_number)
        logger.info(f"[call={call_id}] CALL STARTED phone={phone_number}")
        return JSONResponse({"status": "ok"})

    if msg_type == "status-update" and message.get("status") == "ended":
        call_data = active_calls.pop(call_id, None)

        # Compute call duration. Prefer our own recorded start time; fall back
        # to VAPI's reported duration if the start event never fired.
        if call_data and call_data.get("started_at"):
            duration = time.time() - call_data["started_at"]
        else:
            duration = float(message.get("durationSeconds", 0) or 0)

        logger.info(f"[call={call_id}] CALL ENDED duration={duration:.1f}s")
        return JSONResponse({"status": "ok"})

    logger.debug(f"[call={call_id}] ignored server event type={msg_type!r}")
    return JSONResponse({"status": "ignored"})


@app.get("/health")
async def health():
    return {"status": "running"}


# Alias: if your Vapi dashboard's Server URL is set to ".../voice" instead of
# the bare base URL, this keeps those lifecycle calls from 404ing. (You can
# alternatively just fix the dashboard Server URL to the bare ngrok base URL
# with no "/voice" suffix, since "/" above already handles it.)
@app.post("/voice")
async def voice_alias(request: Request):
    return await server_events(request)
