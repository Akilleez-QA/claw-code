/// Local SQLite-backed RAG (retrieval-augmented generation) store.
///
/// Designed for offline facility edge deployments — no network calls.
/// Documents are indexed by chunking text into overlapping windows and
/// storing them with a simple BM25-style TF score for retrieval.
///
/// Schema:
///   documents(id, source_path, ingested_at)
///   chunks(id, doc_id, chunk_index, text, token_count)
///
/// Retrieval is keyword-overlap scoring (no embeddings required on Pi 4).

use rusqlite::{params, Connection, Result as SqlResult};
use serde_json::Value as JsonValue;

const CHUNK_CHARS: usize = 800;
const CHUNK_OVERLAP: usize = 200;
const DEFAULT_TOP_K: usize = 5;

pub struct RagStore {
    conn: Connection,
}

impl RagStore {
    /// Open or create the RAG database at the given path.
    pub fn open(db_path: &str) -> SqlResult<Self> {
        let conn = Connection::open(db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS documents (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 source_path TEXT NOT NULL UNIQUE,
                 ingested_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS chunks (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                 chunk_index INTEGER NOT NULL,
                 text TEXT NOT NULL,
                 token_count INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks(doc_id);",
        )?;
        Ok(Self { conn })
    }

    /// Open an in-memory database (useful for testing / ephemeral sessions).
    pub fn in_memory() -> SqlResult<Self> {
        Self::open(":memory:")
    }

    /// Ingest a text document from a file path.
    pub fn ingest_file(&mut self, path: &str) -> SqlResult<usize> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| rusqlite::Error::InvalidPath(e.to_string().into()))?;
        self.ingest_text(path, &text)
    }

    /// Ingest a raw text string under a given source label.
    pub fn ingest_text(&mut self, source_label: &str, text: &str) -> SqlResult<usize> {
        let now = chrono_now();
        let tx = self.conn.transaction()?;

        // Upsert document record, replacing chunks on re-ingest.
        tx.execute(
            "INSERT INTO documents(source_path, ingested_at) VALUES(?1, ?2)
             ON CONFLICT(source_path) DO UPDATE SET ingested_at=excluded.ingested_at",
            params![source_label, now],
        )?;
        let doc_id: i64 = tx.query_row(
            "SELECT id FROM documents WHERE source_path = ?1",
            params![source_label],
            |row| row.get(0),
        )?;

        tx.execute("DELETE FROM chunks WHERE doc_id = ?1", params![doc_id])?;

        let chunks = chunk_text(text, CHUNK_CHARS, CHUNK_OVERLAP);
        let count = chunks.len();
        for (idx, chunk) in chunks.iter().enumerate() {
            let token_est = chunk.split_whitespace().count() as i64;
            tx.execute(
                "INSERT INTO chunks(doc_id, chunk_index, text, token_count) VALUES(?1,?2,?3,?4)",
                params![doc_id, idx as i64, chunk, token_est],
            )?;
        }
        tx.commit()?;
        Ok(count)
    }

    /// Search for chunks relevant to `query`. Returns up to `top_k` results.
    pub fn search(&self, query: &str, top_k: Option<usize>) -> SqlResult<Vec<RagResult>> {
        let k = top_k.unwrap_or(DEFAULT_TOP_K);
        let query_terms = tokenize(query);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        // Load all chunks — on Pi 4 with typical facility docs this fits in RAM.
        let mut stmt = self.conn.prepare(
            "SELECT c.id, d.source_path, c.chunk_index, c.text
             FROM chunks c JOIN documents d ON d.id = c.doc_id",
        )?;
        let all: Vec<(i64, String, usize, String)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get::<_, i64>(2)? as usize, row.get(3)?))
            })?
            .filter_map(|r| r.ok())
            .collect();

        // Score by keyword overlap (simple TF-IDF approximation).
        let mut scored: Vec<(f64, i64, String, usize, String)> = all
            .into_iter()
            .map(|(id, path, idx, text)| {
                let score = overlap_score(&query_terms, &text);
                (score, id, path, idx, text)
            })
            .filter(|(score, ..)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(score, _id, source, chunk_index, text)| RagResult {
                source,
                chunk_index,
                text,
                score,
            })
            .collect())
    }

    /// Return document count.
    pub fn document_count(&self) -> SqlResult<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
    }
}

#[derive(Debug, Clone)]
pub struct RagResult {
    pub source: String,
    pub chunk_index: usize,
    pub text: String,
    pub score: f64,
}

/// Execute a RAG tool call from LLM-supplied JSON input.
pub fn execute_rag_tool(store: &RagStore, input: &JsonValue) -> JsonValue {
    let action = input["action"].as_str().unwrap_or("search");

    match action {
        "search" => {
            let query = input["query"].as_str().unwrap_or("");
            let top_k = input["top_k"].as_u64().map(|n| n as usize);
            match store.search(query, top_k) {
                Ok(results) if results.is_empty() => serde_json::json!({
                    "success": true,
                    "results": [],
                    "message": "No relevant documents found for this query."
                }),
                Ok(results) => {
                    let formatted: Vec<JsonValue> = results
                        .iter()
                        .map(|r| {
                            serde_json::json!({
                                "source": r.source,
                                "chunk": r.chunk_index,
                                "score": r.score,
                                "text": r.text
                            })
                        })
                        .collect();
                    serde_json::json!({ "success": true, "results": formatted })
                }
                Err(e) => serde_json::json!({ "success": false, "error": e.to_string() }),
            }
        }
        "status" => match store.document_count() {
            Ok(count) => serde_json::json!({ "success": true, "document_count": count }),
            Err(e) => serde_json::json!({ "success": false, "error": e.to_string() }),
        },
        _ => serde_json::json!({ "success": false, "error": format!("unknown action: {action}") }),
    }
}

/// JSON tool schema for the RAG tool.
pub fn tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["search", "status"],
                "description": "search: find relevant document chunks. status: show index stats."
            },
            "query": {
                "type": "string",
                "description": "Natural language query (required for search)"
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum results to return (default 5)"
            }
        },
        "required": ["action"]
    })
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = Vec::new();
    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut start = 0;
    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        chunks.push(chars[start..end].iter().collect::<String>().trim().to_string());
        if end == chars.len() {
            break;
        }
        start += step;
    }
    chunks.into_iter().filter(|c| !c.is_empty()).collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 2)
        .map(|t| t.to_lowercase())
        .collect()
}

fn overlap_score(query_terms: &[String], text: &str) -> f64 {
    let text_terms = tokenize(text);
    let text_len = text_terms.len() as f64;
    if text_len == 0.0 {
        return 0.0;
    }
    let hits = query_terms
        .iter()
        .filter(|qt| text_terms.iter().any(|tt| tt == *qt))
        .count() as f64;
    // TF-weighted: more hits relative to chunk length = higher score
    hits * (1.0 + (hits / text_len).ln().max(0.0))
}

fn chrono_now() -> String {
    // No chrono dep — use system time as ISO-8601 approximation
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_and_search_basic() {
        let mut store = RagStore::in_memory().expect("in-memory db");
        let chunks = store
            .ingest_text(
                "test.md",
                "R-410A is a common refrigerant used in modern HVAC systems. \
                 Its critical temperature is around 161°F. \
                 Technicians should always check suction pressure when diagnosing issues.",
            )
            .expect("ingest");
        assert!(chunks >= 1, "expected at least one chunk");

        let results = store.search("suction pressure R-410A", None).expect("search");
        assert!(!results.is_empty(), "expected search results");
        assert!(
            results[0].text.contains("suction") || results[0].text.contains("R-410A"),
            "top result should be relevant"
        );
    }

    #[test]
    fn empty_query_returns_empty() {
        let store = RagStore::in_memory().expect("in-memory db");
        let results = store.search("", None).expect("search");
        assert!(results.is_empty());
    }

    #[test]
    fn reingest_replaces_chunks() {
        let mut store = RagStore::in_memory().expect("in-memory db");
        store.ingest_text("doc.md", "first version content").unwrap();
        store.ingest_text("doc.md", "second version updated").unwrap();
        assert_eq!(store.document_count().unwrap(), 1);
        let results = store.search("updated", None).unwrap();
        assert!(!results.is_empty());
    }
}
