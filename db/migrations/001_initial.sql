-- ARIA — Initial Schema Migration
-- Version: 001
-- Run via: db/database.py → init_db()

-- ── Sessions ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT    PRIMARY KEY,
    user_id      TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    last_active  TEXT    NOT NULL,
    model_used   TEXT,
    total_tokens INTEGER DEFAULT 0
);

-- ── Conversation Turns ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversation_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    tokens      INTEGER DEFAULT 0,
    summarized  INTEGER DEFAULT 0,
    model_used  TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- ── Session Summaries ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS session_summaries (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    turn_start INTEGER NOT NULL,
    turn_end   INTEGER NOT NULL,
    summary    TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- ── Episodic Memories ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS episodic_memories (
    memory_id        TEXT    PRIMARY KEY,
    user_id          TEXT    NOT NULL,
    content          TEXT    NOT NULL,
    importance_score REAL    NOT NULL,
    decay_rate       REAL    NOT NULL,
    category         TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    last_accessed    TEXT    NOT NULL,
    access_count     INTEGER DEFAULT 0,
    is_active        INTEGER DEFAULT 1,
    chroma_id        TEXT
);

-- ── Documents ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT    PRIMARY KEY,
    user_id     TEXT    NOT NULL,
    filename    TEXT    NOT NULL,
    file_type   TEXT    NOT NULL,
    page_count  INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    upload_at   TEXT    NOT NULL,
    file_hash   TEXT,
    is_active   INTEGER DEFAULT 1
);

-- ── Document Chunks ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id     TEXT    PRIMARY KEY,
    doc_id       TEXT    NOT NULL,
    content      TEXT    NOT NULL,
    page_number  INTEGER,
    section      TEXT,
    char_offset  INTEGER,
    token_count  INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 0,
    chroma_id    TEXT,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
);

-- ── Token Usage ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS token_usage (
    model_name  TEXT    NOT NULL,
    date        TEXT    NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    PRIMARY KEY (model_name, date)
);

-- ── Analytics Events ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analytics_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    session_id    TEXT,
    user_id       TEXT,
    agent         TEXT,
    action        TEXT,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    latency_ms    INTEGER DEFAULT 0,
    model_used    TEXT,
    tool_used     TEXT,
    success       INTEGER DEFAULT 1,
    error         TEXT
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_turns_session      ON conversation_turns(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_user      ON episodic_memories(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_chunks_doc         ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_analytics_timestamp ON analytics_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_token_usage_date   ON token_usage(model_name, date);
