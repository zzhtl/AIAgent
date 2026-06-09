//! Per-model price table for rough cost estimation.
//!
//! Numbers are USD per million tokens (input / output). They drift —
//! treat the output as a coarse indicator, not invoicing. Unknown models
//! return zero rather than guessing.

use agent_core::TokenUsage;

#[derive(Debug, Clone, Copy)]
pub struct Price {
    pub input_per_million: f64,
    pub output_per_million: f64,
    /// Price for *cached* input tokens (prompt-cache hits). OpenAI ≈ 0.5×
    /// input, DeepSeek ≈ 0.25×, Anthropic cache-read ≈ 0.1×. Coarse — the
    /// cached portion is `TokenUsage::cached_tokens`, a subset of
    /// `prompt_tokens` (providers are normalised to that invariant).
    pub cached_input_per_million: f64,
}

const PRICES: &[(&str, Price)] = &[
    // OpenAI (cached ≈ 0.5× input)
    ("gpt-4o-mini", Price { input_per_million: 0.15, output_per_million: 0.6, cached_input_per_million: 0.075 }),
    ("gpt-4o",      Price { input_per_million: 2.5,  output_per_million: 10.0, cached_input_per_million: 1.25 }),
    ("gpt-4.1-mini", Price { input_per_million: 0.4, output_per_million: 1.6, cached_input_per_million: 0.2 }),
    ("gpt-4.1",     Price { input_per_million: 2.0,  output_per_million: 8.0, cached_input_per_million: 1.0 }),
    ("o1-mini",     Price { input_per_million: 1.1,  output_per_million: 4.4, cached_input_per_million: 0.55 }),
    ("o1",          Price { input_per_million: 15.0, output_per_million: 60.0, cached_input_per_million: 7.5 }),
    // DeepSeek (cache-hit rate)
    ("deepseek-chat",     Price { input_per_million: 0.27, output_per_million: 1.1, cached_input_per_million: 0.07 }),
    ("deepseek-reasoner", Price { input_per_million: 0.55, output_per_million: 2.19, cached_input_per_million: 0.14 }),
    // Anthropic (cache-read ≈ 0.1× input)
    ("claude-haiku-4",  Price { input_per_million: 0.8,  output_per_million: 4.0, cached_input_per_million: 0.08 }),
    ("claude-sonnet-4", Price { input_per_million: 3.0,  output_per_million: 15.0, cached_input_per_million: 0.3 }),
    ("claude-opus-4",   Price { input_per_million: 15.0, output_per_million: 75.0, cached_input_per_million: 1.5 }),
    // Older Claude 3.x for completeness
    ("claude-3-5-sonnet", Price { input_per_million: 3.0,  output_per_million: 15.0, cached_input_per_million: 0.3 }),
    ("claude-3-5-haiku",  Price { input_per_million: 0.8,  output_per_million: 4.0, cached_input_per_million: 0.08 }),
    ("claude-3-opus",     Price { input_per_million: 15.0, output_per_million: 75.0, cached_input_per_million: 1.5 }),
];

/// Look up the price for a model id by longest-prefix match (lowercased).
/// Returns `None` for unknown models.
pub fn price_for(model: &str) -> Option<Price> {
    let lower = model.to_ascii_lowercase();
    let mut best: Option<(&str, Price)> = None;
    for (prefix, p) in PRICES {
        if lower.contains(prefix)
            && best.map(|(b, _)| prefix.len() > b.len()).unwrap_or(true)
        {
            best = Some((prefix, *p));
        }
    }
    best.map(|(_, p)| p)
}

/// Estimate cost in USD. Returns 0.0 for unknown models. Cached input tokens
/// (`usage.cached_tokens`, a subset of `prompt_tokens`) are billed at the
/// model's cache rate; the rest of the prompt at the full input rate.
pub fn estimate_cost_usd(model: &str, usage: TokenUsage) -> f64 {
    let Some(p) = price_for(model) else {
        return 0.0;
    };
    let cached = usage.cached_tokens.min(usage.prompt_tokens);
    let non_cached_prompt = usage.prompt_tokens.saturating_sub(cached);
    let prompt_usd = (non_cached_prompt as f64 / 1_000_000.0) * p.input_per_million
        + (cached as f64 / 1_000_000.0) * p.cached_input_per_million;
    let completion_usd = (usage.completion_tokens as f64 / 1_000_000.0) * p.output_per_million;
    prompt_usd + completion_usd
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_is_zero() {
        let cost = estimate_cost_usd("non-existent-model-xyz", TokenUsage {
            prompt_tokens: 1000,
            completion_tokens: 1000,
            cached_tokens: 0,
        });
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn known_model_matches_prefix() {
        let p = price_for("gpt-4o-mini-2024-07-18").expect("price");
        assert_eq!(p.input_per_million, 0.15);
    }

    #[test]
    fn longer_prefix_wins() {
        // `gpt-4o-mini` is longer than `gpt-4o`, so it wins.
        let p = price_for("gpt-4o-mini").expect("price");
        assert_eq!(p.input_per_million, 0.15);
    }

    #[test]
    fn cached_tokens_billed_at_cache_rate() {
        // 1M prompt tokens, all cached → cost is the cache rate, not the
        // full input rate. gpt-4o-mini: cache 0.075 vs input 0.15.
        let usage = TokenUsage { prompt_tokens: 1_000_000, completion_tokens: 0, cached_tokens: 1_000_000 };
        let cost = estimate_cost_usd("gpt-4o-mini", usage);
        assert!((cost - 0.075).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn partial_cache_splits_input() {
        // 1M prompt of which 800k cached: 200k @ 0.15 + 800k @ 0.075.
        let usage = TokenUsage { prompt_tokens: 1_000_000, completion_tokens: 0, cached_tokens: 800_000 };
        let cost = estimate_cost_usd("gpt-4o-mini", usage);
        let expected = 0.2 * 0.15 + 0.8 * 0.075;
        assert!((cost - expected).abs() < 1e-9, "got {cost}");
    }

    #[test]
    fn zero_cache_matches_full_input_rate() {
        let usage = TokenUsage { prompt_tokens: 1_000_000, completion_tokens: 1_000_000, cached_tokens: 0 };
        let cost = estimate_cost_usd("gpt-4o-mini", usage);
        assert!((cost - (0.15 + 0.6)).abs() < 1e-9, "got {cost}");
    }
}
