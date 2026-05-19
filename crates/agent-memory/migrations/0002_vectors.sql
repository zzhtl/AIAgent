CREATE TABLE IF NOT EXISTS vectors (
    key            TEXT    PRIMARY KEY,
    text           TEXT    NOT NULL,
    embedding_json TEXT    NOT NULL,
    metadata_json  TEXT    NOT NULL DEFAULT '{}',
    created_at     INTEGER NOT NULL
);
