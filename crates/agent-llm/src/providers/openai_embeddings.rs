//! OpenAI embeddings provider.
//!
//! Wraps the `/v1/embeddings` endpoint with the same auth strategy as
//! `OpenAiProvider`. Default model `text-embedding-3-small` (1536-dim,
//! cheap). DeepSeek does not (yet) serve OpenAI-compatible embeddings, so
//! this implementation is OpenAI-only.

use agent_core::llm::LlmError;
use agent_core::memory::EmbeddingProvider;
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

pub struct OpenAiEmbeddingProvider {
    http: Client,
    base_url: String,
    model: String,
    dimension: usize,
}

impl OpenAiEmbeddingProvider {
    pub fn new(api_key: impl Into<String>) -> Result<Self, LlmError> {
        Self::with_model(api_key, "text-embedding-3-small", 1536)
    }

    pub fn with_model(
        api_key: impl Into<String>,
        model: impl Into<String>,
        dimension: usize,
    ) -> Result<Self, LlmError> {
        let mut headers = HeaderMap::new();
        let value = format!("Bearer {}", api_key.into());
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&value).map_err(|_| LlmError::Auth("api key has non-ascii bytes".into()))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| LlmError::Network(e.to_string()))?;
        Ok(Self {
            http,
            base_url: "https://api.openai.com/v1".into(),
            model: model.into(),
            dimension,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAiEmbeddingProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LlmError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}/embeddings", self.base_url.trim_end_matches('/'));
        let body = json!({ "model": self.model, "input": texts });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;
        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let message = resp.text().await.unwrap_or_default();
            return Err(LlmError::Provider { status, message });
        }
        let parsed: EmbeddingsResponse = resp
            .json()
            .await
            .map_err(|e| LlmError::InvalidResponse(e.to_string()))?;
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}
