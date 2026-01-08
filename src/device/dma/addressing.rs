//! Multi-dimensional address generation for DMA transfers.
//!
//! AIE2 DMA supports up to 3 dimensions (AIE2P supports 4). Each dimension
//! specifies a size (number of iterations) and stride (address increment
//! after each iteration).
//!
//! # Address Generation
//!
//! For a 2D transfer with:
//! - Base address: 0x1000
//! - D0: size=16, stride=4 (16 elements, 4 bytes apart)
//! - D1: size=4, stride=64 (4 rows, 64 bytes between row starts)
//!
//! Generated addresses:
//! ```text
//! Row 0: 0x1000, 0x1004, 0x1008, ..., 0x103C (16 addresses)
//! Row 1: 0x1040, 0x1044, 0x1048, ..., 0x107C (16 addresses)
//! Row 2: 0x1080, 0x1084, 0x1088, ..., 0x10BC (16 addresses)
//! Row 3: 0x10C0, 0x10C4, 0x10C8, ..., 0x10FC (16 addresses)
//! ```
//!
//! # Wrap Behavior
//!
//! After the innermost dimension (D0) completes, the address wraps according
//! to the stride of D1. This enables patterns like:
//! - Contiguous 1D: D0 size=N, stride=element_size
//! - 2D with gaps: D0 size=row_len, stride=element_size, D1 size=rows, stride=row_pitch
//! - Interleaved: D0 size=N, stride=2*element_size (access every other element)

/// Configuration for a single dimension.
#[derive(Debug, Clone, Copy, Default)]
pub struct DimensionConfig {
    /// Number of elements in this dimension (iteration count)
    /// Value of 0 means disabled (treated as 1 iteration)
    pub size: u32,

    /// Address stride in bytes (can be negative for reverse traversal)
    /// Applied after each iteration of this dimension
    pub stride: i32,
}

impl DimensionConfig {
    /// Create a new dimension config.
    pub fn new(size: u32, stride: i32) -> Self {
        Self { size, stride }
    }

    /// Get effective size (0 = 1 iteration)
    #[inline]
    pub fn effective_size(&self) -> u32 {
        if self.size == 0 { 1 } else { self.size }
    }

    /// Check if this dimension is enabled (has iterations)
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.size > 1
    }
}

/// Configuration for iteration mode (BD repeat with offset).
///
/// Iteration mode allows a BD to be repeated multiple times, with an
/// address offset applied between each repetition. This is separate from
/// dimensional addressing and is used for patterns like sliding windows.
///
/// # AM025 Format (Word 4)
///
/// - `current` (bits 24:19): Current iteration step (read-only during transfer)
/// - `wrap` (bits 18:13): Wrap count (actual repetitions - 1)
/// - `stepsize` (bits 12:0): Per-iteration offset (actual - 1, in words)
#[derive(Debug, Clone, Copy, Default)]
pub struct IterationConfig {
    /// Current iteration step (0 to wrap)
    pub current: u8,

    /// Wrap count (number of repetitions - 1, 0 = single iteration)
    pub wrap: u8,

    /// Per-iteration address offset in words (actual - 1)
    /// The actual offset = (stepsize + 1) * 4 bytes
    pub stepsize: u16,
}

impl IterationConfig {
    /// Create a new iteration config.
    ///
    /// # Arguments
    /// * `wrap` - Number of repetitions - 1 (0 = single iteration)
    /// * `stepsize` - Address offset per iteration in words (actual - 1)
    pub fn new(wrap: u8, stepsize: u16) -> Self {
        Self { current: 0, wrap, stepsize }
    }

    /// Check if iteration is enabled (more than one repetition)
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.wrap > 0
    }

    /// Get the actual stepsize in bytes.
    #[inline]
    pub fn stepsize_bytes(&self) -> i32 {
        ((self.stepsize as i32) + 1) * 4
    }

    /// Get total number of iterations.
    #[inline]
    pub fn total_iterations(&self) -> u8 {
        self.wrap.saturating_add(1)
    }
}

/// Multi-dimensional address generator with iteration support.
///
/// Generates a sequence of addresses based on up to 4 dimensions plus iteration.
/// Dimensions are processed from innermost (D0) to outermost (D3), with iteration
/// as an outermost loop that adds an offset and repeats the entire dimensional pattern.
///
/// The address is computed as:
/// `base + iteration_counter * iteration_stride + d0_counter * d0_stride + ...`
///
/// # Iteration Mode
///
/// Iteration is an AIE-ML feature that repeats the dimensional pattern with a
/// per-iteration address offset. This enables patterns like sliding windows:
///
/// ```text
/// With d0_size=4, d1_size=2, iteration_wrap=2, iteration_stepsize=8:
///
/// Iteration 0: [0,4,8,12, 16,20,24,28]
/// Iteration 1: [32,36,40,44, 48,52,56,60]  (base + 32)
/// Iteration 2: [64,68,72,76, 80,84,88,92]  (base + 64)
/// ```
#[derive(Debug, Clone)]
pub struct AddressGenerator {
    /// Base address
    base: u64,

    /// Dimension configurations (D0 is innermost)
    dimensions: [DimensionConfig; 4],

    /// Current position in each dimension
    counters: [u32; 4],

    /// Iteration configuration (outermost loop)
    iteration: IterationConfig,

    /// Current iteration step
    iteration_counter: u8,

    /// Total elements to generate (including all iterations)
    total_elements: u64,

    /// Elements generated so far
    elements_generated: u64,

    /// Whether we've finished
    finished: bool,
}

impl AddressGenerator {
    /// Create a new address generator for 1D transfer.
    pub fn new_1d(base: u64, count: u32, stride: i32) -> Self {
        Self::new(base, [
            DimensionConfig::new(count, stride),
            DimensionConfig::default(),
            DimensionConfig::default(),
            DimensionConfig::default(),
        ])
    }

    /// Create a new address generator for 2D transfer.
    pub fn new_2d(
        base: u64,
        d0_size: u32,
        d0_stride: i32,
        d1_size: u32,
        d1_stride: i32,
    ) -> Self {
        Self::new(base, [
            DimensionConfig::new(d0_size, d0_stride),
            DimensionConfig::new(d1_size, d1_stride),
            DimensionConfig::default(),
            DimensionConfig::default(),
        ])
    }

    /// Create a new address generator for 3D transfer.
    pub fn new_3d(
        base: u64,
        d0_size: u32,
        d0_stride: i32,
        d1_size: u32,
        d1_stride: i32,
        d2_size: u32,
        d2_stride: i32,
    ) -> Self {
        Self::new(base, [
            DimensionConfig::new(d0_size, d0_stride),
            DimensionConfig::new(d1_size, d1_stride),
            DimensionConfig::new(d2_size, d2_stride),
            DimensionConfig::default(),
        ])
    }

    /// Create a new address generator with custom dimensions (no iteration).
    pub fn new(base: u64, dimensions: [DimensionConfig; 4]) -> Self {
        Self::with_iteration(base, dimensions, IterationConfig::default())
    }

    /// Create a new address generator with dimensions and iteration.
    ///
    /// Iteration acts as an outermost loop: the dimensional pattern is repeated
    /// `iteration.wrap + 1` times, with `iteration.stepsize_bytes()` added to
    /// the base address each iteration.
    pub fn with_iteration(
        base: u64,
        dimensions: [DimensionConfig; 4],
        iteration: IterationConfig,
    ) -> Self {
        let elements_per_iteration: u64 = dimensions.iter()
            .map(|d| d.effective_size() as u64)
            .product();

        let iteration_count = iteration.total_iterations() as u64;
        let total_elements = elements_per_iteration * iteration_count;

        Self {
            base,
            dimensions,
            counters: [0; 4],
            iteration,
            iteration_counter: 0,
            total_elements,
            elements_generated: 0,
            finished: total_elements == 0,
        }
    }

    /// Compute the current address from counters and iteration.
    fn compute_address(&self) -> u64 {
        let mut addr = self.base as i64;

        // Add iteration offset
        addr += (self.iteration_counter as i64) * (self.iteration.stepsize_bytes() as i64);

        // Add dimension offsets
        for dim in 0..4 {
            addr += (self.counters[dim] as i64) * (self.dimensions[dim].stride as i64);
        }

        addr as u64
    }

    /// Get the current address.
    #[inline]
    pub fn current(&self) -> u64 {
        self.compute_address()
    }

    /// Check if all addresses have been generated.
    #[inline]
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get total number of elements.
    #[inline]
    pub fn total_elements(&self) -> u64 {
        self.total_elements
    }

    /// Get number of elements remaining.
    #[inline]
    pub fn remaining(&self) -> u64 {
        self.total_elements.saturating_sub(self.elements_generated)
    }

    /// Advance to the next address.
    ///
    /// Returns the next address, or None if finished.
    pub fn next(&mut self) -> Option<u64> {
        if self.finished {
            return None;
        }

        let addr = self.compute_address();
        self.elements_generated += 1;

        if self.elements_generated >= self.total_elements {
            self.finished = true;
            return Some(addr);
        }

        // Advance counters
        self.advance();

        Some(addr)
    }

    /// Advance counters to the next position.
    fn advance(&mut self) {
        // Start with D0 (innermost)
        for dim in 0..4 {
            self.counters[dim] += 1;

            if self.counters[dim] < self.dimensions[dim].effective_size() {
                // This dimension hasn't wrapped, we're done
                return;
            }

            // This dimension wrapped, reset counter and continue to next dimension
            self.counters[dim] = 0;
        }

        // All dimensions wrapped - advance iteration counter
        if self.iteration.is_enabled() {
            self.iteration_counter += 1;
            if self.iteration_counter > self.iteration.wrap {
                // All iterations complete
                self.finished = true;
            }
            // Dimension counters already reset to 0 above
        } else {
            // No iteration mode, we're done
            self.finished = true;
        }
    }

    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.counters = [0; 4];
        self.iteration_counter = 0;
        self.elements_generated = 0;
        self.finished = self.total_elements == 0;
    }

    /// Create an iterator over all addresses.
    pub fn iter(&self) -> AddressIterator {
        AddressIterator {
            generator: self.clone(),
        }
    }
}

/// Iterator over addresses generated by an AddressGenerator.
pub struct AddressIterator {
    generator: AddressGenerator,
}

impl Iterator for AddressIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.generator.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.generator.remaining() as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for AddressIterator {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d_contiguous() {
        let gen = AddressGenerator::new_1d(0x1000, 4, 4);
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![0x1000, 0x1004, 0x1008, 0x100C]);
    }

    #[test]
    fn test_1d_with_gap() {
        // Access every other word
        let gen = AddressGenerator::new_1d(0x1000, 4, 8);
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![0x1000, 0x1008, 0x1010, 0x1018]);
    }

    #[test]
    fn test_1d_negative_stride() {
        // Reverse traversal
        let gen = AddressGenerator::new_1d(0x100C, 4, -4);
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![0x100C, 0x1008, 0x1004, 0x1000]);
    }

    #[test]
    fn test_2d_row_major() {
        // 4x2 matrix, row-major, 4 bytes per element
        // Row stride = 4 elements * 4 bytes = 16 bytes
        let gen = AddressGenerator::new_2d(
            0x1000,
            4, 4,   // D0: 4 elements, stride 4
            2, 16,  // D1: 2 rows, stride 16
        );
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![
            0x1000, 0x1004, 0x1008, 0x100C,  // Row 0
            0x1010, 0x1014, 0x1018, 0x101C,  // Row 1
        ]);
    }

    #[test]
    fn test_2d_with_padding() {
        // 2x2 matrix with row padding (stride > row_size * element_size)
        // Each row is 2 elements but stride is 32 bytes (padding of 24 bytes)
        let gen = AddressGenerator::new_2d(
            0x1000,
            2, 4,   // D0: 2 elements per row
            2, 32,  // D1: 2 rows, 32-byte stride
        );
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![
            0x1000, 0x1004,  // Row 0
            0x1020, 0x1024,  // Row 1 (starts at base + 32)
        ]);
    }

    #[test]
    fn test_3d_transfer() {
        // 2x2x2 3D array
        let gen = AddressGenerator::new_3d(
            0x1000,
            2, 4,    // D0: 2 elements, stride 4
            2, 8,    // D1: 2 rows, stride 8
            2, 16,   // D2: 2 planes, stride 16
        );
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![
            // Plane 0
            0x1000, 0x1004,  // Row 0
            0x1008, 0x100C,  // Row 1
            // Plane 1
            0x1010, 0x1014,  // Row 0
            0x1018, 0x101C,  // Row 1
        ]);
    }

    #[test]
    fn test_total_elements() {
        let gen = AddressGenerator::new_2d(0x1000, 4, 4, 3, 16);
        assert_eq!(gen.total_elements(), 12);  // 4 * 3
    }

    #[test]
    fn test_remaining() {
        let mut gen = AddressGenerator::new_1d(0x1000, 10, 4);

        assert_eq!(gen.remaining(), 10);

        gen.next();
        assert_eq!(gen.remaining(), 9);

        gen.next();
        gen.next();
        assert_eq!(gen.remaining(), 7);
    }

    #[test]
    fn test_reset() {
        let mut gen = AddressGenerator::new_1d(0x1000, 4, 4);

        // Consume half
        gen.next();
        gen.next();
        assert_eq!(gen.remaining(), 2);

        // Reset
        gen.reset();
        assert_eq!(gen.remaining(), 4);
        assert_eq!(gen.current(), 0x1000);
    }

    #[test]
    fn test_single_element() {
        let gen = AddressGenerator::new_1d(0x1000, 1, 4);
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![0x1000]);
    }

    #[test]
    fn test_disabled_dimension() {
        // D0 has size 4, D1 has size 0 (disabled = 1 iteration)
        let gen = AddressGenerator::new_2d(0x1000, 4, 4, 0, 16);
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![0x1000, 0x1004, 0x1008, 0x100C]);
    }

    #[test]
    fn test_exact_size_iterator() {
        let gen = AddressGenerator::new_1d(0x1000, 8, 4);
        let iter = gen.iter();

        assert_eq!(iter.len(), 8);

        let addrs: Vec<u64> = iter.collect();
        assert_eq!(addrs.len(), 8);
    }

    #[test]
    fn test_column_major_access() {
        // Column-major access of a 4x2 matrix stored in row-major order
        // Elements: [0,0] [0,1] [0,2] [0,3] [1,0] [1,1] [1,2] [1,3]
        // Access:   [0,0] [1,0] [0,1] [1,1] [0,2] [1,2] [0,3] [1,3]
        // Row stride is 16 bytes (4 elements * 4 bytes)
        let gen = AddressGenerator::new_2d(
            0x1000,
            2, 16,  // D0: 2 rows (stride is row_size = 16)
            4, 4,   // D1: 4 columns (stride is element_size = 4)
        );
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![
            0x1000, 0x1010,  // Column 0: [0,0], [1,0]
            0x1004, 0x1014,  // Column 1: [0,1], [1,1]
            0x1008, 0x1018,  // Column 2: [0,2], [1,2]
            0x100C, 0x101C,  // Column 3: [0,3], [1,3]
        ]);
    }

    #[test]
    fn test_iteration_mode_simple() {
        // 1D transfer with 3 iterations
        // d0: 2 elements, stride 4
        // iteration: wrap=2 (3 iterations), stepsize=1 (offset = 8 bytes per iteration)
        let gen = AddressGenerator::with_iteration(
            0x1000,
            [
                DimensionConfig::new(2, 4),  // 2 elements, 4-byte stride
                DimensionConfig::default(),
                DimensionConfig::default(),
                DimensionConfig::default(),
            ],
            IterationConfig::new(2, 1),  // wrap=2 (3 iterations), stepsize=1 (actual=2 words=8 bytes)
        );
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![
            // Iteration 0: base + 0
            0x1000, 0x1004,
            // Iteration 1: base + 8
            0x1008, 0x100C,
            // Iteration 2: base + 16
            0x1010, 0x1014,
        ]);
        assert_eq!(addrs.len(), 6);  // 2 elements * 3 iterations
    }

    #[test]
    fn test_iteration_mode_2d() {
        // 2D transfer with 2 iterations (sliding window pattern)
        // d0: 2 elements (row), stride 4
        // d1: 2 rows, stride 16
        // iteration: wrap=1 (2 iterations), stepsize=3 (offset = 16 bytes per iteration)
        let gen = AddressGenerator::with_iteration(
            0x1000,
            [
                DimensionConfig::new(2, 4),   // 2 elements per row
                DimensionConfig::new(2, 16),  // 2 rows
                DimensionConfig::default(),
                DimensionConfig::default(),
            ],
            IterationConfig::new(1, 3),  // wrap=1 (2 iterations), stepsize=3 (actual=4 words=16 bytes)
        );
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![
            // Iteration 0: base + 0
            0x1000, 0x1004,  // Row 0
            0x1010, 0x1014,  // Row 1
            // Iteration 1: base + 16
            0x1010, 0x1014,  // Row 0 (offset by 16)
            0x1020, 0x1024,  // Row 1 (offset by 16)
        ]);
        assert_eq!(addrs.len(), 8);  // 4 elements * 2 iterations
    }

    #[test]
    fn test_iteration_disabled() {
        // wrap=0 means single iteration (no repeat)
        let gen = AddressGenerator::with_iteration(
            0x1000,
            [
                DimensionConfig::new(4, 4),
                DimensionConfig::default(),
                DimensionConfig::default(),
                DimensionConfig::default(),
            ],
            IterationConfig::new(0, 10),  // wrap=0 (1 iteration), stepsize doesn't matter
        );
        let addrs: Vec<u64> = gen.iter().collect();

        assert_eq!(addrs, vec![0x1000, 0x1004, 0x1008, 0x100C]);
        assert_eq!(addrs.len(), 4);  // No iteration repeat
    }

    #[test]
    fn test_iteration_total_elements() {
        // Verify total_elements includes iterations
        let gen = AddressGenerator::with_iteration(
            0x1000,
            [
                DimensionConfig::new(3, 4),   // 3 elements
                DimensionConfig::new(2, 12),  // 2 rows
                DimensionConfig::default(),
                DimensionConfig::default(),
            ],
            IterationConfig::new(3, 0),  // wrap=3 (4 iterations)
        );

        // 3 * 2 * 4 = 24 elements
        assert_eq!(gen.total_elements(), 24);
    }

    #[test]
    fn test_iteration_reset() {
        let mut gen = AddressGenerator::with_iteration(
            0x1000,
            [
                DimensionConfig::new(2, 4),
                DimensionConfig::default(),
                DimensionConfig::default(),
                DimensionConfig::default(),
            ],
            IterationConfig::new(1, 1),  // 2 iterations
        );

        // Consume some elements
        gen.next();
        gen.next();
        gen.next();
        assert_eq!(gen.remaining(), 1);

        // Reset
        gen.reset();
        assert_eq!(gen.remaining(), 4);  // 2 elements * 2 iterations
        assert_eq!(gen.current(), 0x1000);
    }
}
