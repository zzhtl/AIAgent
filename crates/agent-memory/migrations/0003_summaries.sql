CREATE TABLE IF NOT EXISTS summaries (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT    NOT NULL,
    body              TEXT    NOT NULL,
    -- Messages with id <= cutoff_message_id are considered "compressed into
    -- this summary". NULL means "everything in the session at the time".
    cutoff_message_id INTEGER,
    created_at        INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, id);
