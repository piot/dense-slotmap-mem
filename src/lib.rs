/*
 * Copyright (c) Peter Bjorklund. All rights reserved. https://github.com/piot/dense-slotmap-mem
 * Licensed under the MIT License. See LICENSE in the project root for license information.
 */

//! Generation-tracked dense slot map with a compact memory layout: Swamp Vec header,
//! dense values, and a trailer containing sparse bookkeeping arrays (4-byte aligned).
//!
//! # Memory Layout
//!
//! ```text
//! [ Header (8B) | Dense Values | Trailer ]
//!
//! Header (Swamp Vec compatible):
//!   +0: capacity (u16), len (u16), element_size (u32)
//!
//! Dense Values:
//!   offset 8, size = capacity * element_size
//!
//! Trailer (4-byte aligned):
//!   - Header (12B): magic (u32), elem_size (u32), free_top (u16), pad (u16)
//!   - Arrays (each u16[capacity]):
//!     * id_to_index: ID -> dense index (0xFFFF = invalid)
//!     * index_to_id: dense index -> ID
//!     * generation:  ID -> generation counter
//!     * free_stack:  reusable ID stack (length = free_top)
//! ```
//!
//! # Handles
//!
//! Stable references are `(id: u16, generation: u16)` pairs. A handle is valid when:
//! - `id < capacity`
//! - `id_to_index[id] != 0xFFFF`
//! - `generation[id]` matches the handle's generation
//!
//! # Operations
//!
//! - **Allocate**: Pop ID from free stack, append to dense array at `len++`
//! - **Remove**: Swap-remove from dense array, increment generation, push ID to free stack
//! - **Access**: Validate handle -> get dense index -> access `values[index]`
//!
//! # Invariants
//!
//! - `len + free_top == capacity` (all IDs are either in-use or on free stack)
//! - Sentinel value for invalid entries: `0xFFFF`

#![no_std]

// Since we are doing low level memory manipulation with raw pointers
// we have to turn off these warnings
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use core::ptr;

// Constants for the new layout
pub const VEC_HEADER_MAGIC_CODE: u32 = 0xC001_C0DE;
const SVEC_TRAILER_MAGIC: u32 = 0x5356_4543; // TODO: 'SVEC' historical Magic code, should probably be changed in the future
const INVALID_U16: u16 = 0xFFFF;
const HEADER_SIZE: usize = 8; // capacity(2) + len(2) + element_size(4)
const VALUES_OFFSET: usize = HEADER_SIZE;
const TRAILER_HEADER_SIZE: usize = 12;

/// Align to 4-byte boundary
#[inline]
const fn align4(x: usize) -> usize {
    (x + 3) & !3
}

/// Validate slot map integrity in debug builds.
/// Checks magic code, `element_size` consistency, and reasonable values for capacity/len.
#[inline]
#[allow(unused_variables)]
pub fn debug_validate_slotmap(base: *const u8) {
    #[cfg(debug_assertions)]
    unsafe {
        //eprintln!("slotmap:{base:p} validate!");
        // Check alignment
        debug_assert_eq!((base as usize) & 3, 0, "base must be 4-byte aligned");

        // Read header fields
        let capacity = *base.cast::<u16>();
        let len = *base.add(2).cast::<u16>();
        let elem_size = *base.add(4).cast::<u32>();

        // Validate basic constraints
        debug_assert_ne!(capacity, 0, "capacity must not be 0 (did you call init()?)");
        debug_assert!(
            len <= capacity,
            "len ({len}) must not exceed capacity ({capacity}) - memory corruption or uninitialized slot map"
        );
        debug_assert_ne!(elem_size, 0, "element_size must not be 0 (did you call init()?)");
        debug_assert!(
            elem_size <= 1024 * 1024,
            "element_size ({elem_size}) is unreasonably large - possible memory corruption"
        );

        // Check trailer magic and element_size consistency
        let trailer_off = trailer_offset(capacity, elem_size);
        let trailer_magic = *base.add(trailer_off).cast::<u32>();
        let trailer_elem_size = *base.add(trailer_off + 4).cast::<u32>();

        debug_assert_eq!(
            trailer_magic,
            SVEC_TRAILER_MAGIC,
            "Invalid trailer magic at offset {trailer_off}: expected 0x{SVEC_TRAILER_MAGIC:08X}, got 0x{trailer_magic:08X}\n\
             Slot map state: capacity={capacity}, len={len}, elem_size={elem_size}\n\
             This likely means the slot map was not properly initialized with init(), \
             or the memory has been corrupted, or you're using the wrong base pointer"
        );
        debug_assert_eq!(
            trailer_elem_size,
            elem_size,
            "element_size mismatch: header has {elem_size}, trailer has {trailer_elem_size} - memory corruption detected"
        );

        // Validate free_top
        let free_top = *base.add(trailer_off + 8).cast::<u16>();
        debug_assert!(
            free_top <= capacity,
            "free_top ({free_top}) must not exceed capacity ({capacity}) - memory corruption detected"
        );

        // Validate invariant: len + free_top == capacity
        debug_assert_eq!(
            u32::from(len) + u32::from(free_top),
            u32::from(capacity),
            "Invariant violated: len ({len}) + free_top ({free_top}) != capacity ({capacity})\n\
             This indicates memory corruption or a bug in the slot map implementation"
        );
    }
}

/// Compute trailer offset (after dense values, 4-byte aligned)
#[inline]
const fn trailer_offset(capacity: u16, element_size: u32) -> usize {
    let values_size = capacity as usize * element_size as usize;
    align4(VALUES_OFFSET + values_size)
}

/// Compute total bytes needed in memory for a sparse vector. Used for code generator to know
/// how much space to reserve.
#[must_use]
pub const fn layout_size(capacity: u16, element_size: u32) -> usize {
    let cap = capacity as usize;
    let trailer_off = trailer_offset(capacity, element_size);
    let arrays_off = align4(trailer_off + TRAILER_HEADER_SIZE);

    // Four arrays: id_to_index, index_to_id, generation, free_stack
    // Each is u16[capacity]
    let arrays_size = 4 * cap * size_of::<u16>();

    arrays_off + arrays_size
}

/// Alignment requirement for the sparse vector.
#[must_use]
pub const fn alignment() -> usize {
    4
}

/// Helper functions to get pointers to trailer arrays
#[inline]
const unsafe fn id_to_index_ptr(base: *mut u8, capacity: u16, element_size: u32) -> *mut u16 {
    unsafe {
        let trailer_off = trailer_offset(capacity, element_size);
        let arrays_off = align4(trailer_off + TRAILER_HEADER_SIZE);
        base.add(arrays_off).cast::<u16>()
    }
}

#[inline]
const unsafe fn index_to_id_ptr(base: *mut u8, capacity: u16, element_size: u32) -> *mut u16 {
    unsafe {
        let trailer_off = trailer_offset(capacity, element_size);
        let arrays_off = align4(trailer_off + TRAILER_HEADER_SIZE);
        let cap = capacity as usize;
        base.add(arrays_off + cap * size_of::<u16>()).cast::<u16>()
    }
}

#[inline]
const unsafe fn generation_ptr(base: *mut u8, capacity: u16, element_size: u32) -> *mut u16 {
    unsafe {
        let trailer_off = trailer_offset(capacity, element_size);
        let arrays_off = align4(trailer_off + TRAILER_HEADER_SIZE);
        let cap = capacity as usize;
        base.add(arrays_off + 2 * cap * size_of::<u16>())
            .cast::<u16>()
    }
}

#[inline]
const unsafe fn free_stack_ptr(base: *mut u8, capacity: u16, element_size: u32) -> *mut u16 {
    unsafe {
        let trailer_off = trailer_offset(capacity, element_size);
        let arrays_off = align4(trailer_off + TRAILER_HEADER_SIZE);
        let cap = capacity as usize;
        base.add(arrays_off + 3 * cap * size_of::<u16>())
            .cast::<u16>()
    }
}

#[inline]
const unsafe fn free_top_ptr(base: *mut u8, capacity: u16, element_size: u32) -> *mut u16 {
    unsafe {
        let trailer_off = trailer_offset(capacity, element_size);
        base.add(trailer_off + 8).cast::<u16>()
    }
}

/// Initialize the sparse vector to memory specified by the raw memory pointer.
/// `base` must point to a region of at least `layout_size(capacity, element_size)` bytes.
///
/// # Safety
/// - `base` must point to valid memory of at least `layout_size(capacity, element_size)` bytes
/// - `base` must be 4-byte aligned
/// - `capacity` must not be 0
/// - The memory region must not be accessed concurrently
pub unsafe fn init(base: *mut u8, capacity: u16, element_size: u32) {
    unsafe {
        debug_assert_eq!((base as usize) & 3, 0, "base must be 4-byte aligned");
        debug_assert_ne!(capacity, 0, "capacity must not be 0");

        let cap = capacity as usize;

        // Initialize header (8 bytes)
        ptr::write(base.cast::<u16>(), capacity); // capacity
        ptr::write(base.add(2).cast::<u16>(), 0); // len = 0
        ptr::write(base.add(4).cast::<u32>(), element_size); // element_size (full u32)

        // Initialize trailer header
        let trailer_off = trailer_offset(capacity, element_size);
        ptr::write(base.add(trailer_off).cast::<u32>(), SVEC_TRAILER_MAGIC);
        ptr::write(base.add(trailer_off + 4).cast::<u32>(), element_size);
        ptr::write(base.add(trailer_off + 8).cast::<u16>(), capacity); // free_top = capacity
        ptr::write(base.add(trailer_off + 10).cast::<u16>(), 0); // _pad2

        // Initialize id_to_index array (all invalid)
        let id_to_idx_ptr = id_to_index_ptr(base, capacity, element_size);
        for i in 0..cap {
            ptr::write(id_to_idx_ptr.add(i), INVALID_U16);
        }

        // Initialize index_to_id array (all invalid)
        let idx_to_id_ptr = index_to_id_ptr(base, capacity, element_size);
        for i in 0..cap {
            ptr::write(idx_to_id_ptr.add(i), INVALID_U16);
        }

        // Initialize generation array (all 1 for first use)
        let gen_ptr = generation_ptr(base, capacity, element_size);
        for i in 0..cap {
            ptr::write(gen_ptr.add(i), 1);
        }

        // Initialize free_stack with all IDs in ascending order (0..capacity)
        // IDs will be popped in reverse order (capacity-1 first), achieving LIFO reuse
        let free_stk_ptr = free_stack_ptr(base, capacity, element_size);
        #[allow(clippy::cast_possible_truncation)]
        for i in 0..cap {
            ptr::write(free_stk_ptr.add(i), i as u16);
        }
    }
}

/// Clear the slot map, removing all elements and resetting to initial state.
///
/// This is much faster than removing elements one by one.
/// All existing handles are invalidated (generations remain, so old handles will fail validation).
/// # Safety
/// `base` must point to a valid initialized slot map.
pub unsafe fn clear(base: *mut u8) {
    unsafe {
        debug_validate_slotmap(base);

        let capacity = *base.cast::<u16>();
        let element_size = element_size(base);

        // Reset len to 0
        ptr::write(base.add(2).cast::<u16>(), 0);

        // Reset free_top to capacity (all IDs available)
        let free_top_p = free_top_ptr(base, capacity, element_size);
        ptr::write(free_top_p, capacity);

        // Reset id_to_index (mark all as invalid)
        let id_to_idx_ptr = id_to_index_ptr(base, capacity, element_size);
        for i in 0..capacity as usize {
            ptr::write(id_to_idx_ptr.add(i), INVALID_U16);
        }

        // Reset index_to_id (mark all as invalid)
        let idx_to_id_ptr = index_to_id_ptr(base, capacity, element_size);
        for i in 0..capacity as usize {
            ptr::write(idx_to_id_ptr.add(i), INVALID_U16);
        }

        // Reinitialize free_stack with all IDs
        let free_stk_ptr = free_stack_ptr(base, capacity, element_size);
        #[allow(clippy::cast_possible_truncation)]
        for i in 0..capacity as usize {
            ptr::write(free_stk_ptr.add(i), i as u16);
        }

        // Note: We keep generations as-is, which means old handles remain invalid
        // If you want to allow old handles to work after clear, increment all generations here
    }
}

/// Allocate a new ID and generation. Returns (id, generation) for the new handle.
/// Implements: pop id from `free_stack`, append to dense array.
///
/// # Safety
/// - `base` must point to a valid initialized slot map
/// - `base` must be 4-byte aligned
/// - The memory region must not be accessed concurrently
pub unsafe fn allocate(base: *mut u8) -> Option<(u16, u16)> {
    unsafe {
        debug_validate_slotmap(base);

        let capacity = *base.cast::<u16>();
        let element_size = element_size(base);
        let len_ptr = base.add(2).cast::<u16>();
        let len = *len_ptr;

        let free_top_p = free_top_ptr(base, capacity, element_size);
        let free_top = *free_top_p;

        // Check if we have free IDs
        if free_top == 0 {
            //eprintln!("slotmap:{base:p} no free top");
            return None;
        }

        // Pop id from free_stack
        let new_free_top = free_top - 1;
        ptr::write(free_top_p, new_free_top);
        let free_stk_ptr = free_stack_ptr(base, capacity, element_size);
        let id = *free_stk_ptr.add(new_free_top as usize);

        // Append: index = len, then len += 1
        let index = len;
        ptr::write(len_ptr, len + 1);

        // Update mappings
        let id_to_idx_ptr = id_to_index_ptr(base, capacity, element_size);
        let idx_to_id_ptr = index_to_id_ptr(base, capacity, element_size);
        ptr::write(id_to_idx_ptr.add(id as usize), index);
        ptr::write(idx_to_id_ptr.add(index as usize), id);

        // Get generation (it was incremented on previous free, or is 0 for first use)
        let gen_ptr = generation_ptr(base, capacity, element_size);
        let generation = *gen_ptr.add(id as usize);

        //eprintln!("slotmap:{base:p} allocate id:{id} (index:{index}) gen:{generation}, len:{}", len + 1);

        Some((id, generation))
    }
}

/// Compute offset of values region (always 8 in the new layout)
#[must_use]
pub const fn values_offset(_base: *const u8) -> usize {
    VALUES_OFFSET
}

/// Validate handle and get dense index
unsafe fn validate_handle(base: *mut u8, id: u16, generation: u16) -> Option<u16> {
    unsafe {
        let capacity = *base.cast::<u16>();
        let element_size_val = element_size(base);

        // Check id bounds
        if id >= capacity {
            return None;
        }

        // Check generation
        let gen_ptr = generation_ptr(base, capacity, element_size_val);
        if *gen_ptr.add(id as usize) != generation {
            return None;
        }

        // Check id_to_index validity
        let id_to_idx_ptr = id_to_index_ptr(base, capacity, element_size_val);
        let index = *id_to_idx_ptr.add(id as usize);
        if index == INVALID_U16 {
            return None;
        }

        Some(index)
    }
}

/// Insert raw bytes at handle (id, generation).
/// Uses dense indexing: validates handle, gets dense index, writes to values[index].
/// # Safety
///
#[inline]
pub unsafe fn insert(base: *mut u8, id: u16, generation: u16, src: *const u8) -> bool {
    unsafe {
        debug_validate_slotmap(base);

        let element_size = element_size(base);

        // Validate handle and get dense index
        let Some(index) = validate_handle(base, id, generation) else { return false };


        // Write to values[index]
        let offset = VALUES_OFFSET + (index as usize) * (element_size as usize);
        ptr::copy_nonoverlapping(src, base.add(offset), element_size as usize);
        true
    }
}

/// Remove by handle; implements swap-remove in dense area.
/// # Safety
///
pub unsafe fn remove(base: *mut u8, id: u16, generation: u16) -> bool {
    unsafe {
        debug_validate_slotmap(base);

        let capacity = *base.cast::<u16>();
        let element_size_val = element_size(base);

        // Validate handle and get dense index
        let Some(index) = validate_handle(base, id, generation) else { return false };


        let len_ptr = base.add(2).cast::<u16>();
        let len = *len_ptr;
        let last = len - 1;

        //eprintln!("slotmap:{base:p} remove id:{id} (index:{index}) gen:{generation}");


        // If not removing the last element, swap with last
        if index != last {
            // Swap dense values[index] <-> values[last] using efficient swap
            let elem_size = element_size_val as usize;
            let values_index_off = VALUES_OFFSET + (index as usize) * elem_size;
            let values_last_off = VALUES_OFFSET + (last as usize) * elem_size;

            // Use ptr::swap_nonoverlapping for efficient swap
            ptr::swap_nonoverlapping(
                base.add(values_index_off),
                base.add(values_last_off),
                elem_size,
            );

            // Fix maps: moved_id is the id that was at last
            let idx_to_id_ptr = index_to_id_ptr(base, capacity, element_size_val);
            let moved_id = *idx_to_id_ptr.add(last as usize);

            // Update index_to_id[index] = moved_id
            ptr::write(idx_to_id_ptr.add(index as usize), moved_id);

            // Update id_to_index[moved_id] = index
            let id_to_idx_ptr = id_to_index_ptr(base, capacity, element_size_val);
            ptr::write(id_to_idx_ptr.add(moved_id as usize), index);
        }

        // Clear last slot maps
        let idx_to_id_ptr = index_to_id_ptr(base, capacity, element_size_val);
        ptr::write(idx_to_id_ptr.add(last as usize), INVALID_U16);

        let id_to_idx_ptr = id_to_index_ptr(base, capacity, element_size_val);
        ptr::write(id_to_idx_ptr.add(id as usize), INVALID_U16);

        // Decrement len
        ptr::write(len_ptr, last);
        //eprintln!("slotmap:{base:p} remove id:{id} (index:{index}) gen:{generation} len:{last} written");

        // Retire id: increment generation and push to free_stack
        let gen_ptr = generation_ptr(base, capacity, element_size_val);
        let old_gen = *gen_ptr.add(id as usize);
        ptr::write(gen_ptr.add(id as usize), old_gen.wrapping_add(1));

        let free_top_p = free_top_ptr(base, capacity, element_size_val);
        let free_top = *free_top_p;
        let free_stk_ptr = free_stack_ptr(base, capacity, element_size_val);
        ptr::write(free_stk_ptr.add(free_top as usize), id);
        ptr::write(free_top_p, free_top + 1);

        true
    }
}

/// Check handle validity
/// # Safety
///
pub unsafe fn is_alive(base: *mut u8, id: u16, generation: u16) -> bool {
    unsafe {
        debug_validate_slotmap(base);
        validate_handle(base, id, generation).is_some()
    }
}


/// Get a pointer to the generation array
/// # Safety
/// IMPORTANT: The returned array is indexed by ID, not by dense index.
/// To get the generation for a dense index, first look up the ID via `index_to_id_ptr_pub()`,
/// then use that ID to index into this generation array.
pub unsafe fn id_to_generation_ptr_pub(base: *mut u8) -> *mut u16 {
    unsafe {
        debug_validate_slotmap(base);

        let capacity = *base.cast::<u16>();
        let element_size = element_size(base);
        generation_ptr(base, capacity, element_size)
    }
}

/// Get the generation for a given dense index.
/// This is a convenience function that does the two-step lookup:
/// 1. Gets the ID from `index_to_id`[index]
/// 2. Gets the generation from generation[id]
/// # Safety
/// `base` must point to a valid initialized slot map.
/// `index` must be less than the current len.
pub unsafe fn get_generation_for_index(base: *mut u8, index: u16) -> Option<u16> {
    unsafe {
        debug_validate_slotmap(base);

        let len = element_count(base);
        if index >= len {
            return None;
        }

        let capacity = *base.cast::<u16>();
        let element_size = element_size(base);

        // Step 1: Get ID from index_to_id[index]
        let idx_to_id_ptr = index_to_id_ptr(base, capacity, element_size);
        let id = *idx_to_id_ptr.add(index as usize);

        if id == INVALID_U16 {
            return None;
        }

        // Step 2: Get generation from generation[id]
        let gen_ptr = generation_ptr(base, capacity, element_size);
        Some(*gen_ptr.add(id as usize))
    }
}


/// Get a pointer to the `index_to_id` array (for compatibility/debugging)
/// # Safety
///
pub unsafe fn index_to_id_ptr_pub(base: *mut u8) -> *mut u16 {
    unsafe {
        debug_validate_slotmap(base);

        let capacity = *base.cast::<u16>();
        let element_size = element_size(base);
        index_to_id_ptr(base, capacity, element_size)
    }
}

/// Get a pointer to the `id_to_index` array (for compatibility/debugging)
/// # Safety
///
pub unsafe fn id_to_index_ptr_pub(base: *mut u8) -> *mut u16 {
    unsafe {
        debug_validate_slotmap(base);

        let capacity = *base.cast::<u16>();
        let element_size = element_size(base);
        id_to_index_ptr(base, capacity, element_size)
    }
}

/// Get current element count (len)
/// # Safety
/// `base` must point to a valid initialized slot map and be 4-byte aligned.
#[must_use]
pub const unsafe fn element_count(base: *const u8) -> u16 {
    // Note: Cannot debug_assert alignment in const fn
    unsafe { *base.add(2).cast::<u16>() }
}

/// Get current element size from header (fast, deterministic)
/// # Safety
/// `base` must point to a valid initialized slot map and be 4-byte aligned.
#[must_use]
#[inline]
pub const unsafe fn element_size(base: *const u8) -> u32 {
    // Note: Cannot debug_assert alignment in const fn
    unsafe {
        // Read element_size directly from header at offset 0x04
        *base.add(4).cast::<u32>()
    }
}

/// Insert raw bytes at handle (id, generation), validating first.
/// Returns true if successful, false if the handle is invalid
/// # Safety
pub unsafe fn insert_if_alive(base: *mut u8, id: u16, generation: u16, src: *const u8) -> bool {
    unsafe {
        debug_validate_slotmap(base);

        if is_alive(base, id, generation) {
            insert(base, id, generation, src)
        } else {
            false
        }
    }
}

/// Get value pointer for a valid handle
/// Returns None if the handle is invalid
/// # Safety
pub unsafe fn get_value_ptr(base: *mut u8, id: u16, generation: u16) -> Option<*mut u8> {
    unsafe {
        debug_validate_slotmap(base);

        let element_size_val = element_size(base);

        // Validate handle and get dense index
        let index = validate_handle(base, id, generation)?;

        // Return pointer to values[index]
        let offset = VALUES_OFFSET + (index as usize) * (element_size_val as usize);
        Some(base.add(offset))
    }
}
