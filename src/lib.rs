#![allow(unused_macros)]

//! # Tetanus - Extended Rust Standard Library
//! 
//! `tetanus` is a utility library that extends Rust's standard library with additional
//! functionality, convenient macros, and ergonomic tools for common programming tasks.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::io::{self, Read};
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

/// Extension trait for iterators to provide statistical operations
pub trait StatisticsExt: Iterator + Sized
where
    Self::Item: Into<f64> + Copy,
{
    fn mean(mut self) -> Option<f64> {
        let mut count = 0;
        let mut sum = 0.0;
        
        while let Some(value) = self.next() {
            count += 1;
            sum += value.into();
        }
        
        if count > 0 {
            Some(sum / count as f64)
        } else {
            None
        }
    }

    fn variance(mut self) -> Option<f64> {
        let mut values: Vec<f64> = Vec::new();
        while let Some(value) = self.next() {
            values.push(value.into());
        }
        
        if values.is_empty() {
            return None;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
            
        Some(variance)
    }

    fn std_dev(self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }
}

impl<T: Iterator> StatisticsExt for T 
where
    T::Item: Into<f64> + Copy
{}

/// A thread-safe cache with automatic expiration of entries
pub struct ExpiringCache<K, V> {
    cache: Arc<Mutex<HashMap<K, (V, Instant)>>>,
    ttl: Duration,
}

impl<K: Eq + Hash + Clone, V: Clone> ExpiringCache<K, V> {
    pub fn new(ttl: Duration) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            ttl,
        }
    }

    pub fn insert(&self, key: K, value: V) {
        let mut cache = self.cache.lock().unwrap();
        cache.insert(key, (value, Instant::now()));
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.lock().unwrap();
        
        if let Some((value, timestamp)) = cache.get(key) {
            if timestamp.elapsed() > self.ttl {
                cache.remove(key);
                None
            } else {
                Some(value.clone())
            }
        } else {
            None
        }
    }

    pub fn cleanup(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.retain(|_, (_, timestamp)| timestamp.elapsed() <= self.ttl);
    }
}

/// A retry mechanism with exponential backoff
pub struct RetryWithBackoff {
    max_attempts: u32,
    initial_delay: Duration,
    max_delay: Duration,
    factor: f64,
}

impl RetryWithBackoff {
    pub fn new(max_attempts: u32, initial_delay: Duration, max_delay: Duration, factor: f64) -> Self {
        Self {
            max_attempts,
            initial_delay,
            max_delay,
            factor,
        }
    }

    pub async fn retry<F, T, E>(&self, mut operation: F) -> Result<T, E>
    where
        F: FnMut() -> Result<T, E>,
    {
        let mut attempts = 0;
        let mut delay = self.initial_delay;

        loop {
            match operation() {
                Ok(value) => return Ok(value),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.max_attempts {
                        return Err(e);
                    }
                    
                    std::thread::sleep(delay);
                    delay = Duration::from_secs_f64(
                        (delay.as_secs_f64() * self.factor)
                            .min(self.max_delay.as_secs_f64())
                    );
                }
            }
        }
    }
}

/// Extension trait for reading chunks of data with progress tracking
pub trait ChunkedReadExt: Read {
    fn read_chunks_with_progress<F>(
        &mut self,
        chunk_size: usize,
        mut progress_callback: F
    ) -> io::Result<Vec<u8>>
    where
        F: FnMut(usize, usize)
    {
        let mut buffer = Vec::new();
        let mut chunk = vec![0; chunk_size];
        let mut total_bytes = 0;
        
        loop {
            match self.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    buffer.extend_from_slice(&chunk[..n]);
                    total_bytes += n;
                    progress_callback(n, total_bytes);
                }
                Err(e) => return Err(e),
            }
        }
        
        Ok(buffer)
    }
}

impl<R: Read> ChunkedReadExt for R {}

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

#[cfg(test)]
mod extension_tests {
    use super::*;
    use std::io::Cursor;
    use std::thread;

    #[test]
    fn test_statistics_ext() {
        let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // Test mean
        assert_eq!(
            numbers.iter().copied().mean(),
            Some(3.0)
        );
        
        // Test variance
        let variance = numbers.iter().copied().variance().unwrap();
        assert!((variance - 2.0).abs() < 1e-10);
        
        // Test standard deviation
        let std_dev = numbers.iter().copied().std_dev().unwrap();
        assert!((std_dev - 1.4142135623730951).abs() < 1e-10);
        
        // Test empty iterator
        let empty: Vec<f64> = vec![];
        assert_eq!(empty.iter().copied().mean(), None);
        assert_eq!(empty.iter().copied().variance(), None);
        assert_eq!(empty.iter().copied().std_dev(), None);
        
        // Test with integers
        let int_numbers = vec![1, 2, 3, 4, 5];
        assert_eq!(
            int_numbers.iter().copied().mean(),
            Some(3.0)
        );
    }

    #[test]
    fn test_expiring_cache() {
        let cache = ExpiringCache::new(Duration::from_millis(100));
        
        // Test basic insertion and retrieval
        cache.insert("key1", "value1");
        assert_eq!(cache.get(&"key1"), Some("value1"));
        
        // Test expiration
        cache.insert("key2", "value2");
        thread::sleep(Duration::from_millis(150));
        assert_eq!(cache.get(&"key2"), None);
        
        // Test cleanup
        cache.insert("key3", "value3");
        thread::sleep(Duration::from_millis(50));
        cache.insert("key4", "value4");
        thread::sleep(Duration::from_millis(60));
        
        cache.cleanup();
        assert_eq!(cache.get(&"key3"), None);
        assert_eq!(cache.get(&"key4"), Some("value4"));
    }

    #[tokio::test]
    async fn test_retry_with_backoff() {
        let retry = RetryWithBackoff::new(
            3,                                    // max attempts
            Duration::from_millis(10),           // initial delay
            Duration::from_millis(100),          // max delay
            2.0                                  // backoff factor
        );

        // Test successful operation
        let mut counter = 0;
        let result = retry.retry(|| {
            counter += 1;
            Ok::<_, &str>(counter)
        }).await;
        assert_eq!(result, Ok(1));
        assert_eq!(counter, 1);

        // Test operation that fails then succeeds
        let mut attempts = 0;
        let result = retry.retry(|| {
            attempts += 1;
            if attempts < 2 {
                Err("not yet")
            } else {
                Ok(attempts)
            }
        }).await;
        assert_eq!(result, Ok(2));
        assert_eq!(attempts, 2);

        // Test operation that always fails
        let mut fail_counter = 0;
        let result: Result<(), &str> = retry.retry(|| {
            fail_counter += 1;
            Err("always fails")
        }).await;
        assert!(result.is_err());
        assert_eq!(fail_counter, 3); // Should have tried 3 times
    }

    #[test]
    fn test_chunked_read_with_progress() {
        // Create test data
        let data = (0..100).collect::<Vec<u8>>();
        let mut cursor = Cursor::new(data.clone());
        
        // Track progress
        let mut chunks_received = 0;
        let mut total_bytes_received = 0;
        
        // Read with progress tracking
        let result = cursor.read_chunks_with_progress(10, |chunk_size, total| {
            chunks_received += 1;
            assert!(chunk_size <= 10); // Ensure chunk size is never larger than specified
            assert!(total <= 100);     // Ensure total never exceeds data size
            total_bytes_received = total;
        }).unwrap();
        
        // Verify results
        assert_eq!(result, data);
        assert_eq!(total_bytes_received, 100);
        assert_eq!(chunks_received, 10);
        
        // Test empty read
        let mut empty_cursor = Cursor::new(Vec::<u8>::new());
        let empty_result = empty_cursor.read_chunks_with_progress(10, |_, _| {
            panic!("Progress callback should not be called for empty read");
        }).unwrap();
        assert!(empty_result.is_empty());
    }
}