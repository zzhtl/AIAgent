//! Provider registry.
//!
//! Resolves an `LlmProvider` by name. Besides ready-made instances it holds
//! lazy *factories*: a provider is constructed only when it is actually
//! selected, so an unselected provider's credentials (API keys read from the
//! environment) are never touched. This is the seam for configuring multiple
//! providers and switching between them at runtime.

use std::collections::HashMap;
use std::sync::Arc;

use agent_core::llm::LlmProvider;

/// Builds a provider on demand, returning the provider or an error message
/// (e.g. a missing API key).
type ProviderFactory = Box<dyn Fn() -> Result<Arc<dyn LlmProvider>, String> + Send + Sync>;

#[derive(Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    factories: HashMap<String, ProviderFactory>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a ready-made provider instance, keyed by its `name()`.
    pub fn register(&mut self, provider: Arc<dyn LlmProvider>) {
        self.providers.insert(provider.name().to_string(), provider);
    }

    /// Register a named factory. The provider is constructed only when
    /// [`ProviderRegistry::build`] is called for `name`, so an unselected
    /// provider's credentials are never read.
    pub fn register_factory(
        &mut self,
        name: impl Into<String>,
        factory: impl Fn() -> Result<Arc<dyn LlmProvider>, String> + Send + Sync + 'static,
    ) {
        self.factories.insert(name.into(), Box::new(factory));
    }

    /// Return an already-registered instance, if any.
    pub fn get(&self, name: &str) -> Option<Arc<dyn LlmProvider>> {
        self.providers.get(name).cloned()
    }

    /// Resolve a provider by name: a pre-registered instance first, otherwise a
    /// registered factory built on demand. `None` if the name is unknown;
    /// `Some(Err)` if a factory failed (e.g. a missing API key).
    pub fn build(&self, name: &str) -> Option<Result<Arc<dyn LlmProvider>, String>> {
        if let Some(p) = self.providers.get(name) {
            return Some(Ok(p.clone()));
        }
        self.factories.get(name).map(|f| f())
    }

    /// Every known provider name (instances and factories).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().chain(self.factories.keys()).map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factory_builds_on_demand_and_unknown_is_none() {
        let mut reg = ProviderRegistry::new();
        reg.register_factory("boom", || Err("no key".to_string()));
        // Unknown name → None (lets the caller report "unsupported provider").
        assert!(reg.build("missing").is_none());
        // Known factory that fails → Some(Err) with the factory's message.
        match reg.build("boom") {
            Some(Err(e)) => assert_eq!(e, "no key"),
            Some(Ok(_)) => panic!("expected the factory to fail"),
            None => panic!("expected the factory to be found"),
        }
    }
}
