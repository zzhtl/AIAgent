//! Type-keyed extension map for carrying arbitrary per-invocation state.
//!
//! A minimal hand-rolled equivalent of `http::Extensions`: each concrete type
//! maps to at most one value. The embedder installs typed handles (a retrieval
//! client, a ticket context, an auth/tenant token) before a run; custom tools
//! pull them back out via `ToolContext::get_ext`. Kept dependency-free because
//! the need is ~40 lines.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// A type-indexed container. Each `T` maps to at most one value.
///
/// Values are not required to be `Clone`, so the map itself is not `Clone`;
/// `ToolContext` shares one set across concurrent tool calls via `Arc`.
#[derive(Default)]
pub struct Extensions {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Extensions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a value, returning the previous value of the same type if any.
    pub fn insert<T: Send + Sync + 'static>(&mut self, val: T) -> Option<T> {
        self.map
            .insert(TypeId::of::<T>(), Box::new(val))
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }

    /// Fetch a shared reference to the value of type `T`, if present.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.map.get(&TypeId::of::<T>()).and_then(|b| b.downcast_ref::<T>())
    }

    pub fn contains<T: Send + Sync + 'static>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

impl std::fmt::Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extensions").field("len", &self.map.len()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Marker(u32);

    struct OtherMarker;

    #[test]
    fn insert_and_get_round_trip() {
        let mut ext = Extensions::new();
        assert!(ext.is_empty());
        assert!(ext.insert(Marker(7)).is_none());
        assert_eq!(ext.get::<Marker>(), Some(&Marker(7)));
        assert!(ext.contains::<Marker>());
        assert_eq!(ext.len(), 1);
    }

    #[test]
    fn insert_replaces_same_type() {
        let mut ext = Extensions::new();
        ext.insert(Marker(1));
        let prev = ext.insert(Marker(2));
        assert_eq!(prev, Some(Marker(1)));
        assert_eq!(ext.get::<Marker>(), Some(&Marker(2)));
    }

    #[test]
    fn missing_type_returns_none() {
        let mut ext = Extensions::new();
        ext.insert(Marker(1));
        assert!(ext.get::<OtherMarker>().is_none());
        assert!(!ext.contains::<OtherMarker>());
    }
}
