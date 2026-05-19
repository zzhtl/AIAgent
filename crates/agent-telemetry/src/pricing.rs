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
}

const PRICES: &[(&str, Price)] = &[
    // OpenAI
    ("gpt-4o-mini", Price { input_per_million: 0.15, output_per_million: 0.6 }),
    ("gpt-4o",      Price { input_per_million: 2.5,  output_per_million: 10.0 }),
    ("gpt-4.1-mini", Price { input_per_million: 0.4, output_per_million: 1.6 }),
    ("gpt-4.1",     Price { input_per_million: 2.0,  output_per_million: 8.0 }),
    ("o1-mini",     Price { input_per_million: 1.1,  output_per_million: 4.4 }),
    ("o1",          Price { input_per_million: 15.0, output_per_million: 60.0 }),
    // DeepSeek
    ("deepseek-chat",     Price { input_per_million: 0.27, output_per_million: 1.1 }),
    ("deepseek-reasoner", Price { input_per_million: 0.55, output_per_million: 2.19 }),
    // Anthropic (rough current-gen rates)
    ("claude-haiku-4",  Price { input_per_million: 0.8,  output_per_million: 4.0 }),
    ("claude-sonnet-4", Price { input_per_million: 3.0,  output_per_million: 15.0 }),
    ("claude-opus-4",   Price { input_per_million: 15.0, output_per_million: 75.0 }),
    // Older Claude 3.x for completeness
    ("claude-3-5-sonnet", Price { input_per_million: 3.0,  output_per_million: 15.0 }),
    ("claude-3-5-haiku",  Price { input_per_million: 0.8,  output_per_million: 4.0 }),
    ("claude-3-opus",     Price { input_per_million: 15.0, output_per_million: 75.0 }),
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

/// Estimate cost in USD. Returns 0.0 for unknown models.
pub fn estimate_cost_usd(model: &str, usage: TokenUsage) -> f64 {
    let Some(p) = price_for(model) else {
        return 0.0;
    };
    let prompt_usd = (usage.prompt_tokens as f64 / 1_000_000.0) * p.input_per_million;
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
}
