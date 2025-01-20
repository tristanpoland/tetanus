# 🦀 Tetanus

> A rusty extension of the standard library with powerful utilities and ergonomic tools.

[![Crates.io](https://img.shields.io/crates/v/tetanus.svg)](https://crates.io/crates/tetanus)
[![Documentation](https://docs.rs/tetanus/badge.svg)](https://docs.rs/tetanus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 🎯 Zero-cost abstractions
- 🔒 Thread-safe utilities
- 🚀 Performance-focused data structures
- 📦 Convenient macros
- ⚡ Rate limiting tools
- ⏱️ High-precision timing utilities
- 📊 Statistical operations
- 🔁 Retry mechanisms
- 📝 Chunked I/O with progress tracking

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
tetanus = "0.1.0"
```

## Usage Examples

### Collection Macros

Create collections with easy-to-read syntax:

```rust
use tetanus::{map, set, vec_of};

// Create a vector with repeated elements
let zeros = vec_of!(0; 5);  // [0, 0, 0, 0, 0]

// Create a HashMap inline
let config = map! {
    "host" => "localhost",
    "port" => "8080",
};

// Create a HashSet inline
let permissions = set!["read", "write", "execute"];
```

### Option Extensions

Enhanced Option handling:

```rust
use tetanus::OptionExt;

let value = Some(42);
assert!(value.is_none_or(|&x| x == 42));  // true
assert!(!value.is_none_or(|&x| x > 100)); // false

let doubled = value.map_ref(|&x| x * 2);  // Some(84)
```

### Thread-Safe Containers

Easy-to-use atomic containers:

```rust
use tetanus::Atomic;

let counter = Atomic::new(0);
let counter_clone = counter.clone();

std::thread::spawn(move || {
    *counter_clone.get_mut() += 1;
});

*counter.get_mut() += 1;
```

### String Utilities

String manipulation tools:

```rust
use tetanus::StringExt;

let mut s = String::from("HelloWorld");
s.to_snake_case();  // "hello_world"
s.to_camel_case();  // "HelloWorld"
s.truncate(5);      // "Hello..."
```

### Vector Extensions

Enhanced vector operations:

```rust
use tetanus::VecExt;

let mut numbers = vec![1, 2, 3, 4, 5];

// Remove all even numbers
numbers.remove_all(|&x| x % 2 == 0);

// Replace numbers greater than 3 with 0
numbers.replace_all(|&x| x > 3, 0);

// Insert maintaining sort order
numbers.insert_sorted(4);
```

### Ring Buffer

Fixed-size circular buffer:

```rust
use tetanus::RingBuffer;

let mut buffer = RingBuffer::new(3);
buffer.push(1);
buffer.push(2);
buffer.push(3);
buffer.push(4);  // Automatically removes 1

assert_eq!(buffer.iter().collect::<Vec<_>>(), vec![2, 3, 4]);
```

### Rate Limiter

Token bucket rate limiting:

```rust
use tetanus::RateLimiter;

let mut limiter = RateLimiter::new(
    capacity: 100,        // bucket size
    refill_rate: 10.0     // tokens per second
);

if limiter.try_acquire() {
    // Perform rate-limited operation
}
```

### Timing Utilities

High-precision timing tools:

```rust
use tetanus::timing::{Timer, Stopwatch};

// Simple timer
let timer = Timer::new();
// ... do work ...
println!("Operation took {} ms", timer.elapsed_ms());

// Stopwatch for multiple measurements
let mut sw = Stopwatch::new();
// ... do work ...
let split1 = sw.split();
// ... do more work ...
let split2 = sw.split();
println!("Splits: {:?}", sw.splits());
```

### Function Memoization

Cache function results automatically:

```rust
use tetanus::memoize;

memoize! {
    fn fib(n: u64) -> u64 {
        if n <= 1 {
            n
        } else {
            fib(n - 1) + fib(n - 2)
        }
    }
}

// First call computes the result
let result1 = fib(10);  // Computed
// Subsequent calls use cached result
let result2 = fib(10);  // Retrieved from cache
```

### Statistical Operations

Compute statistics directly on iterators:

```rust
use tetanus::StatisticsExt;

let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];

let mean = numbers.iter().copied().mean();        // Some(3.0)
let variance = numbers.iter().copied().variance(); // Some(2.0)
let std_dev = numbers.iter().copied().std_dev();   // Some(√2)
```

### Expiring Cache

Thread-safe cache with automatic entry expiration:

```rust
use tetanus::ExpiringCache;
use std::time::Duration;

let cache = ExpiringCache::new(Duration::from_secs(5));
cache.insert("key", "value");

// Value expires after 5 seconds
std::thread::sleep(Duration::from_secs(6));
assert_eq!(cache.get(&"key"), None);
```

### Retry Mechanism with Exponential Backoff

Automatically retry operations with configurable backoff:

```rust
use tetanus::RetryWithBackoff;
use std::time::Duration;

let retry = RetryWithBackoff::new(
    max_attempts: 3,
    initial_delay: Duration::from_millis(10),
    max_delay: Duration::from_millis(100),
    factor: 2.0
);

// Retry an operation that might fail
let result = retry.retry(|| {
    // Your operation here
}).await;
```

### Chunked Reading with Progress Tracking

Read large files with progress callbacks:

```rust
use tetanus::ChunkedReadExt;
use std::fs::File;
use std::io::BufReader;

let file = File::open("large_file.txt")?;
let mut reader = BufReader::new(file);

let data = reader.read_chunks_with_progress(1024, |chunk_size, total_read| {
    println!("Read {} bytes, total: {} bytes", chunk_size, total_read);
})?;
```

## Performance

Tetanus is designed with performance in mind:
- Zero-cost abstractions where possible
- Minimal memory usage
- Cache-friendly data structures
- Minimal runtime overhead

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to:
- The Rust community for inspiration
- Contributors and users for feedback and suggestions
- Various Rust crates that influenced the design

---

Made with ❤️ by the Tetanus team