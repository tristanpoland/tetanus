#![allow(unused_macros)]

//! # Tetanus - Extended Rust Standard Library
//! 
//! `tetanus` is a utility library that extends Rust's standard library with additional
//! functionality, convenient macros, and ergonomic tools for common programming tasks.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Creates a vector with given elements
#[macro_export]
macro_rules! vec_of {
    ($elem:expr; $n:expr) => {
        vec![$elem; $n]
    };
}

/// Creates a HashMap from key-value pairs
#[macro_export]
macro_rules! map {
    ($($key:expr => $value:expr),* $(,)?) => {{
        let mut map = HashMap::new();
        $(
            map.insert($key, $value);
        )*
        map
    }};
}

/// Creates a HashSet from elements
#[macro_export]
macro_rules! set {
    ($($element:expr),* $(,)?) => {{
        let mut set = HashSet::new();
        $(
            set.insert($element);
        )*
        set
    }};
}

/// Extension trait for Option types
pub trait OptionExt<T> {
    fn is_none_or<F>(&self, f: F) -> bool where F: FnOnce(&T) -> bool;
    fn map_ref<U, F>(&self, f: F) -> Option<U> where F: FnOnce(&T) -> U;
}

impl<T> OptionExt<T> for Option<T> {
    fn is_none_or<F>(&self, f: F) -> bool where F: FnOnce(&T) -> bool {
        match self {
            None => true,
            Some(x) => f(x)
        }
    }

    fn map_ref<U, F>(&self, f: F) -> Option<U> where F: FnOnce(&T) -> U {
        self.as_ref().map(f)
    }
}

/// Thread-safe mutable container
#[derive(Default)]
pub struct Atomic<T>(Arc<Mutex<T>>);

impl<T> Atomic<T> {
    pub fn new(value: T) -> Self {
        Self(Arc::new(Mutex::new(value)))
    }

    pub fn get_mut(&self) -> std::sync::MutexGuard<T> {
        self.0.lock().unwrap()
    }

    pub fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

/// Extension trait for strings
pub trait StringExt {
    fn truncate(&mut self, max_chars: usize);
    fn to_snake_case(&self) -> String;
    fn to_camel_case(&self) -> String;
}

impl StringExt for String {
    fn truncate(&mut self, max_chars: usize) {
        let char_count = self.chars().count();
        if char_count > max_chars {
            let truncated: String = self.chars()
                .take(max_chars)
                .collect();
            self.clear();
            self.push_str(&truncated);
            self.push_str("...");
        }
    }

    fn to_snake_case(&self) -> String {
        let mut result = String::with_capacity(self.len() * 2);
        let mut chars = self.chars().peekable();
        
        while let Some(current) = chars.next() {
            if current.is_uppercase() {
                if !result.is_empty() && !result.ends_with('_') {
                    let next_is_lower = chars.peek().map_or(false, |next| next.is_lowercase());
                    if next_is_lower {
                        result.push('_');
                    }
                }
                result.extend(current.to_lowercase());
            } else {
                result.push(current);
            }
        }
        result
    }

    fn to_camel_case(&self) -> String {
        let mut result = String::with_capacity(self.len());
        let mut capitalize_next = true;
        
        for c in self.chars() {
            if c == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.extend(c.to_uppercase());
                capitalize_next = false;
            } else {
                result.push(c);
            }
        }
        result
    }
}

/// Extension trait for vectors
pub trait VecExt<T> {
    fn remove_all<F>(&mut self, predicate: F) where F: Fn(&T) -> bool;
    fn replace_all<F>(&mut self, predicate: F, replacement: T) where F: Fn(&T) -> bool, T: Clone;
    fn insert_sorted(&mut self, element: T) where T: Ord;
}

impl<T> VecExt<T> for Vec<T> {
    fn remove_all<F>(&mut self, predicate: F) where F: Fn(&T) -> bool {
        self.retain(|x| !predicate(x));
    }

    fn replace_all<F>(&mut self, predicate: F, replacement: T) where F: Fn(&T) -> bool, T: Clone {
        for item in self.iter_mut() {
            if predicate(item) {
                *item = replacement.clone();
            }
        }
    }

    fn insert_sorted(&mut self, element: T) where T: Ord {
        match self.binary_search(&element) {
            Ok(pos) | Err(pos) => self.insert(pos, element),
        }
    }
}

/// A fixed-size ring buffer
pub struct RingBuffer<T> {
    buffer: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(item);
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.buffer.iter()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.capacity
    }
}

/// Result extension methods
pub trait ResultExt<T, E> {
    fn on_error<F>(self, f: F) -> Self where F: FnOnce(&E);
    fn ignore_err(self) -> Option<T>;
}

impl<T, E> ResultExt<T, E> for Result<T, E> {
    fn on_error<F>(self, f: F) -> Self where F: FnOnce(&E) {
        if let Err(ref e) = self {
            f(e);
        }
        self
    }

    fn ignore_err(self) -> Option<T> {
        self.ok()
    }
}

/// Rate limiter with token bucket algorithm
pub struct RateLimiter {
    capacity: u32,
    tokens: u32,
    refill_rate: f64,
    last_refill: Instant,
}

impl RateLimiter {
    pub fn new(capacity: u32, refill_rate_per_second: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate: refill_rate_per_second,
            last_refill: Instant::now(),
        }
    }

    pub fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens > 0 {
            self.tokens -= 1;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        let new_tokens = (elapsed * self.refill_rate) as u32;
        if new_tokens > 0 {
            self.tokens = (self.tokens + new_tokens).min(self.capacity);
            self.last_refill = now;
        }
    }
}

/// Timing utilities
pub mod timing {
    use super::*;

    /// High-precision timer
    pub struct Timer(Instant);

    impl Timer {
        pub fn new() -> Self {
            Self(Instant::now())
        }

        pub fn elapsed(&self) -> Duration {
            self.0.elapsed()
        }

        pub fn elapsed_ms(&self) -> u128 {
            self.elapsed().as_millis()
        }
    }

    /// Simple stopwatch for multiple timing measurements
    pub struct Stopwatch {
        start: Instant,
        splits: Vec<Duration>,
    }

    impl Stopwatch {
        pub fn new() -> Self {
            Self {
                start: Instant::now(),
                splits: Vec::new(),
            }
        }

        pub fn split(&mut self) -> Duration {
            let split = self.start.elapsed();
            self.splits.push(split);
            split
        }

        pub fn splits(&self) -> &[Duration] {
            &self.splits
        }

        pub fn reset(&mut self) {
            self.start = Instant::now();
            self.splits.clear();
        }
    }
}

/// Creates a memoized function
#[macro_export]
macro_rules! memoize {
    (fn $name:ident($($arg:ident: $type:ty),*) -> $ret:ty $body:block) => {
        fn $name($($arg: $type),*) -> $ret {
            use std::collections::HashMap;
            use std::sync::Mutex;
            use std::sync::Once;
            
            static INIT: Once = Once::new();
            static mut CACHE: Option<Mutex<HashMap<($($type),*), $ret>>> = None;
            
            INIT.call_once(|| {
                unsafe {
                    CACHE = Some(Mutex::new(HashMap::new()));
                }
            });
            
            let cache = unsafe { CACHE.as_ref().unwrap() };
            let mut cache = cache.lock().unwrap();
            
            if let Some(result) = cache.get(&($($arg),*)) {
                return result.clone();
            }
            
            let result = (|| $body)();
            cache.insert(($($arg),*), result.clone());
            result
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_of() {
        let v = vec_of!(42; 3);
        assert_eq!(v, vec![42, 42, 42]);
    }

    #[test]
    fn test_map() {
        let m = map! {
            "a" => 1,
            "b" => 2,
        };
        assert_eq!(m.get("a"), Some(&1));
        assert_eq!(m.get("b"), Some(&2));
    }

    #[test]
    fn test_set() {
        let s = set![1, 2, 3];
        assert!(s.contains(&1));
        assert!(s.contains(&2));
        assert!(s.contains(&3));
    }

    #[test]
    fn test_option_ext() {
        let some_val = Some(42);
        let none_val: Option<i32> = None;

        assert!(!some_val.is_none_or(|x| x > 100));
        assert!(some_val.is_none_or( |x| x == 42));
        assert!(none_val.is_none_or( |x| x > 0));

        assert_eq!(some_val.map_ref(|&x| x * 2), Some(84));
        assert_eq!(none_val.map_ref(|&x| x * 2), None);
    }

    #[test]
    fn test_vec_ext() {
        let mut v = vec![1, 2, 3, 4, 5];
        v.remove_all(|&x| x % 2 == 0);
        assert_eq!(v, vec![1, 3, 5]);

        let mut v = vec![1, 2, 3];
        v.replace_all(|&x| x > 1, 0);
        assert_eq!(v, vec![1, 0, 0]);

        let mut v = vec![1, 3, 5];
        v.insert_sorted(4);
        assert_eq!(v, vec![1, 3, 4, 5]);
    }

    #[test]
    fn test_ring_buffer() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert!(buffer.is_full());
        
        buffer.push(4);
        let items: Vec<_> = buffer.iter().copied().collect();
        assert_eq!(items, vec![2, 3, 4]);
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(3, 1.0);
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_stopwatch() {
        use std::thread::sleep;
        
        let mut sw = timing::Stopwatch::new();
        sleep(Duration::from_millis(10));
        let split1 = sw.split();
        assert!(split1.as_millis() >= 10);
        
        sleep(Duration::from_millis(10));
        let split2 = sw.split();
        assert!(split2.as_millis() >= 20);
        
        assert_eq!(sw.splits().len(), 2);
    }
}