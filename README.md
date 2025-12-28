# dense-slotmap-mem

A fixed-capacity, generation-tracked **dense slot map** with a raw memory interface, optimized for FFI, VM integration, and low-level systems (Game Engines). Should typically not be used in a normal Rust application.

## What is it?

A slot map is a data structure that provides stable "handles" (in this implementation: ID + generation) to elements. To make it more cache and iteration-friendly it uses **swap-remove** to keep all elements contiguous in memory.

## Features

- **Dense storage** - Elements are always contiguous, removal uses swap-remove
- **Generational indices** - Handles remain stable even when elements move
- **Raw memory API** - Works with `*mut u8` pointers, not generic Rust types
- **Fixed capacity** - No allocations after initialization
- **FFI-friendly** - Compatible with C/C++ and VM environments
- **`no_std` compatible** - Works in embedded and bare-metal environments

## Usage

```rust
use dense_slotmap_mem::{init, allocate, insert, remove, is_alive, layout_size};

unsafe {
    let capacity = 100u16;
    let element_size = 4u32; // e.g., for u32 values
    let size = layout_size(capacity, element_size);

    let mut memory = vec![0u8; size];
    let base = memory.as_mut_ptr();

    // Initialize the slot map
    init(base, capacity, element_size);

    // Allocate a slot and get a stable handle
    let (id, generation) = allocate(base).unwrap();

    // Insert data using the handle
    let value = 42u32;
    insert(base, id, generation, (&raw const value).cast::<u8>());

    // Check if handle is still valid
    assert!(is_alive(base, id, generation));

    // Remove by handle - last element swaps into this slot
    remove(base, id, generation);

    // Old handle is now invalid
    assert!(!is_alive(base, id, generation));
}
```

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2025 Peter Bjorklund. All Rights Reserved.
