/*
 * Copyright (c) Peter Bjorklund. All rights reserved. https://github.com/piot/dense-slotmap-mem
 * Licensed under the MIT License. See LICENSE in the project root for license information.
 */

use dense_slotmap_mem::{
    alignment, allocate, clear, element_count, init, insert, insert_if_alive, is_alive,
    layout_size, remove, values_offset,
};

#[test]
fn test_layout_and_initialization() {
    let capacity = 4u16;
    let element_size = 2u32;
    let size = layout_size(capacity, element_size);

    // Verify layout calculation
    let header = 8;
    let values_size = capacity as usize * element_size as usize;
    let trailer_off = ((header + values_size) + 3) & !3;
    let trailer_header = 12;
    let arrays_off = ((trailer_off + trailer_header) + 3) & !3;
    let arrays_size = 4 * capacity as usize * size_of::<u16>();
    assert_eq!(size, arrays_off + arrays_size);
    assert_eq!(alignment(), 4);

    // Test initialization
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();
    unsafe {
        init(base, capacity, element_size);
        assert_eq!(*(base as *const u16), capacity);
        assert_eq!(element_count(base), 0);
    }
}

#[test]
fn test_allocate_remove_generation_tracking() {
    let capacity = 4u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);

        // Allocate all slots
        let mut handles = Vec::new();
        for _ in 0..capacity {
            let (id, generation) = allocate(base).expect("should allocate");
            assert!(id < capacity);
            assert_eq!(generation, 1, "First generation should be 1");
            assert!(is_alive(base, id, generation));
            handles.push((id, generation));
        }
        assert_eq!(element_count(base), capacity);
        assert!(allocate(base).is_none(), "Should be full");

        // Remove first slot and verify generation increment
        let (id, generation) = handles[0];
        assert!(remove(base, id, generation));
        assert!(!is_alive(base, id, generation));
        assert_eq!(element_count(base), capacity - 1);

        // Reallocate same ID with incremented generation
        let (new_id, new_generation) = allocate(base).expect("should reallocate");
        assert_eq!(new_id, id, "Should reuse same ID");
        assert_eq!(
            new_generation,
            generation + 1,
            "Generation should increment"
        );
        assert!(is_alive(base, new_id, new_generation));
        assert!(
            !is_alive(base, id, generation),
            "Old generation should be invalid"
        );
    }
}

#[test]
fn test_dense_storage_and_values() {
    let capacity = 4u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);

        // Insert values
        let values = [100u32, 200u32, 300u32];
        let mut handles = Vec::new();
        for &val in &values {
            let (id, generation) = allocate(base).unwrap();
            assert!(insert(base, id, generation, (&raw const val).cast::<u8>()));
            handles.push((id, generation));
        }

        // Verify values are stored densely, without any gaps
        let values_off = values_offset(base);
        for (i, &expected) in values.iter().enumerate() {
            let stored = *(base.add(values_off + i * element_size as usize) as *const u32);
            assert_eq!(stored, expected);
        }

        // Remove middle element - last should swap into its place (swap-remove)
        remove(base, handles[1].0, handles[1].1);
        assert_eq!(element_count(base), 2);

        let swapped = *(base.add(values_off + element_size as usize) as *const u32);
        assert_eq!(
            swapped, 300u32,
            "Last element should swap into removed slot"
        );
    }
}

#[test]
fn test_id_reuse_lifo_order() {
    let capacity = 5u16;
    let element_size = 1u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);

        // Allocate all
        let mut handles = Vec::new();
        for _ in 0..capacity {
            handles.push(allocate(base).unwrap());
        }

        // Remove three in specific order
        let remove_order = [1, 3, 0];
        let mut removed_ids = Vec::new();
        for &idx in &remove_order {
            let (id, generation) = handles[idx];
            assert!(remove(base, id, generation));
            removed_ids.push(id);
        }

        // Reallocate - should follow LIFO (stack) order
        let reused = [
            allocate(base).unwrap().0,
            allocate(base).unwrap().0,
            allocate(base).unwrap().0,
        ];

        // Last removed (ID 0) should be first reused
        assert_eq!(reused[0], removed_ids[2]);
        assert_eq!(reused[1], removed_ids[1]);
        assert_eq!(reused[2], removed_ids[0]);
    }
}

#[test]
fn test_invalid_operations() {
    let capacity = 3u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);
        let (id, generation) = allocate(base).unwrap();

        // Wrong generation
        assert!(!remove(base, id, generation + 1));
        assert!(!is_alive(base, id, generation + 1));

        // Remove then try operations with old generation
        assert!(remove(base, id, generation));
        assert!(!is_alive(base, id, generation));
        assert!(!remove(base, id, generation), "Double remove should fail");
        let value = 42u32;
        assert!(
            !insert(base, id, generation, (&raw const value).cast::<u8>()),
            "Insert to removed slot should fail"
        );

        // Reallocate and verify old generation still invalid
        let (new_id, new_generation) = allocate(base).unwrap();
        assert_eq!(new_id, id);
        assert_ne!(new_generation, generation);
        assert!(is_alive(base, new_id, new_generation));
        assert!(!is_alive(base, id, generation));
    }
}

#[test]
fn test_generation_zero_is_invalid() {
    let capacity = 3u16;
    let element_size = 1u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);

        // allocate() should never return generation 0
        for _ in 0..capacity {
            let (_, generation) = allocate(base).unwrap();
            assert_ne!(generation, 0, "Generation 0 should never be returned");
        }

        // Generation 0 should never be valid
        for id in 0..capacity {
            assert!(!is_alive(base, id, 0));
            assert!(!remove(base, id, 0));
            let value = 1u8;
            assert!(!insert_if_alive(base, id, 0, &raw const value));
        }
    }
}

#[test]
fn test_clear() {
    let capacity = 4u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);

        // Allocate some slots
        let mut handles = Vec::new();
        for i in 0..3 {
            let (id, generation) = allocate(base).unwrap();
            let value = (100 + i) as u32;
            insert(base, id, generation, (&raw const value).cast::<u8>());
            handles.push((id, generation));
        }
        assert_eq!(element_count(base), 3);

        // Clear and verify
        clear(base);
        assert_eq!(element_count(base), 0);

        // Old handles should be invalid
        for (id, generation) in &handles {
            assert!(!is_alive(base, *id, *generation));
        }

        // Should be able to allocate full capacity again
        for _ in 0..capacity {
            assert!(allocate(base).is_some());
        }
        assert_eq!(element_count(base), capacity);
        assert!(allocate(base).is_none());
    }
}

#[test]
fn test_stress_mixed_operations() {
    let capacity = 20u16;
    let element_size = 8u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);

        let mut active_handles = Vec::new();
        let mut next_value = 1000u64;

        // 500 mixed add/remove operations
        for iteration in 0..500 {
            let current_count = element_count(base) as usize;

            let should_add = if current_count == 0 {
                true
            } else if current_count == capacity as usize {
                false
            } else {
                (iteration % 3) != 0
            };

            if should_add {
                if let Some((id, generation)) = allocate(base) {
                    let value = next_value;
                    next_value += 1;
                    insert(base, id, generation, (&raw const value).cast::<u8>());
                    active_handles.push((id, generation, value));
                    assert!(is_alive(base, id, generation));
                }
            } else if !active_handles.is_empty() {
                let remove_idx = iteration % active_handles.len();
                let (id, generation, _) = active_handles.remove(remove_idx);
                assert!(remove(base, id, generation));
                assert!(!is_alive(base, id, generation));
            }

            assert_eq!(element_count(base) as usize, active_handles.len());
        }
    }
}

#[test]
fn test_iteration_with_generations() {
    use dense_slotmap_mem::{get_generation_for_index, index_to_id_ptr_pub};

    let capacity = 8u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);

        // Allocate some elements
        let mut handles = Vec::new();
        for i in 0..5 {
            let (id, generation) = allocate(base).unwrap();
            let value = (i + 100) as u32;
            insert(base, id, generation, (&raw const value).cast::<u8>());
            handles.push((id, generation));
        }

        // Remove some (causing swap-remove)
        remove(base, handles[1].0, handles[1].1);
        remove(base, handles[3].0, handles[3].1);
        assert_eq!(element_count(base), 3);

        // Allocate new elements (will reuse freed IDs)
        for _ in 0..2 {
            let (id, generation) = allocate(base).unwrap();
            let value = 200u32;
            insert(base, id, generation, (&raw const value).cast::<u8>());
        }
        assert_eq!(element_count(base), 5);

        // Iterate using proper generation lookup
        for index in 0..element_count(base) {
            let id = *index_to_id_ptr_pub(base).add(index as usize);
            let generation = get_generation_for_index(base, index).expect("valid index");
            assert!(is_alive(base, id, generation), "Each slot should be alive");
        }
    }
}

// Debug validation tests (only run in debug mode)

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Invalid trailer magic")]
fn test_debug_validation_corrupted_magic() {
    let capacity = 3u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);
        let trailer_off = 8 + ((capacity as usize * element_size as usize) + 3) & !3;
        *(base.add(trailer_off) as *mut u32) = 0xDEADBEEF;
        let _ = allocate(base); // Should panic
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "element_size mismatch")]
fn test_debug_validation_element_size_mismatch() {
    let capacity = 3u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);
        let trailer_off = 8 + ((capacity as usize * element_size as usize) + 3) & !3;
        *(base.add(trailer_off + 4) as *mut u32) = 999;
        let _ = allocate(base); // Should panic
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "len")]
fn test_debug_validation_invalid_len() {
    let capacity = 3u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);
        *(base.add(2) as *mut u16) = capacity + 10;
        let _ = allocate(base); // Should panic
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "free_top")]
fn test_debug_validation_invalid_free_top() {
    let capacity = 3u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);
        let trailer_off = 8 + ((capacity as usize * element_size as usize) + 3) & !3;
        *(base.add(trailer_off + 8) as *mut u16) = capacity + 10;
        let _ = allocate(base); // Should panic
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "Invariant violated")]
fn test_debug_validation_invariant_violation() {
    let capacity = 3u16;
    let element_size = 4u32;
    let size = layout_size(capacity, element_size);
    let mut memory_buffer = vec![0u8; size];
    let base = memory_buffer.as_mut_ptr();

    unsafe {
        init(base, capacity, element_size);
        let _ = allocate(base);
        *(base.add(2) as *mut u16) = 2; // Break invariant: len + free_top != capacity
        let _ = is_alive(base, 0, 1); // Should panic
    }
}
