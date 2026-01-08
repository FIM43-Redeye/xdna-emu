//! DMA Sparsity Compression/Decompression.
//!
//! AIE-ML supports hardware compression for sparse data (AM020 Ch1 "Sparsity").
//! This is designed for 8-bit data samples where zeros are common (ReLU activations,
//! sparse weights).
//!
//! # Compression Format
//!
//! Input: 256-bit words (32 bytes)
//! Output: 32-bit mask + packed non-zero bytes (padded to 32-bit boundary)
//!
//! ```text
//! Input:  [b0][b1][b2]...[b31]  (32 bytes)
//! Mask:   bit[i] = 1 if b[i] != 0, else 0
//! Output: [mask (4 bytes)][non-zero bytes][padding to 4-byte align]
//! ```
//!
//! # Example
//!
//! ```ignore
//! Input bytes:  [5, 0, 0, 3, 0, 0, 0, 0, 7, 0, ...]
//! Mask:         0b00001001_00000000_00000000_10000000 (bits 0, 3, 8 set)
//! Output:       [mask][5][3][7][padding]
//! ```
//!
//! # Usage
//!
//! - MM2S (Memory to Stream): compress() before sending
//! - S2MM (Stream to Memory): decompress() after receiving

/// Compress a 256-bit word (32 bytes) using sparsity encoding.
///
/// Returns the compressed data as a vector of bytes:
/// - First 4 bytes: 32-bit mask (little-endian)
/// - Following bytes: packed non-zero data bytes
/// - Padding: zeros to 32-bit boundary
///
/// Returns None if input is not exactly 32 bytes.
pub fn compress(input: &[u8]) -> Option<Vec<u8>> {
    if input.len() != 32 {
        return None;
    }

    // Build the 32-bit mask and collect non-zero bytes
    let mut mask: u32 = 0;
    let mut non_zero_bytes = Vec::with_capacity(32);

    for (i, &byte) in input.iter().enumerate() {
        if byte != 0 {
            mask |= 1 << i;
            non_zero_bytes.push(byte);
        }
    }

    // Build output: mask (4 bytes LE) + non-zero bytes + padding
    let mut output = Vec::with_capacity(4 + non_zero_bytes.len() + 4);

    // Mask in little-endian
    output.extend_from_slice(&mask.to_le_bytes());

    // Non-zero bytes
    output.extend_from_slice(&non_zero_bytes);

    // Pad to 32-bit (4-byte) boundary
    let padding_needed = (4 - (output.len() % 4)) % 4;
    output.extend(std::iter::repeat(0u8).take(padding_needed));

    Some(output)
}

/// Decompress sparsity-encoded data back to a 256-bit word (32 bytes).
///
/// Input format:
/// - First 4 bytes: 32-bit mask (little-endian)
/// - Following bytes: packed non-zero data bytes
///
/// Returns the decompressed 32-byte word, or None if input is invalid.
pub fn decompress(input: &[u8]) -> Option<[u8; 32]> {
    if input.len() < 4 {
        return None;
    }

    // Extract mask (little-endian)
    let mask = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);

    // Count expected non-zero bytes
    let non_zero_count = mask.count_ones() as usize;

    // Check we have enough data (mask + non-zero bytes)
    if input.len() < 4 + non_zero_count {
        return None;
    }

    // Decompress
    let mut output = [0u8; 32];
    let mut data_idx = 4; // Start after mask

    for i in 0..32 {
        if (mask >> i) & 1 == 1 {
            output[i] = input[data_idx];
            data_idx += 1;
        }
        // else: output[i] stays 0
    }

    Some(output)
}

/// Get the compressed size for a 32-byte word (without actually compressing).
///
/// Returns the number of bytes the compressed output would be:
/// 4 (mask) + non_zero_count + padding_to_4_byte_boundary
pub fn compressed_size(input: &[u8]) -> usize {
    if input.len() != 32 {
        return 0;
    }

    let non_zero_count = input.iter().filter(|&&b| b != 0).count();
    let total = 4 + non_zero_count;

    // Round up to 4-byte boundary
    (total + 3) & !3
}

/// Check if compression would be beneficial for this data.
///
/// Returns true if the compressed size would be smaller than uncompressed.
pub fn is_compressible(input: &[u8]) -> bool {
    if input.len() != 32 {
        return false;
    }

    // Compression overhead: 4 bytes for mask
    // Beneficial if we save more than 4 bytes by omitting zeros
    let zero_count = input.iter().filter(|&&b| b == 0).count();

    // Need more than 4 zeros to benefit (mask overhead)
    // Plus padding overhead consideration
    zero_count > 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_all_zeros() {
        let input = [0u8; 32];
        let compressed = compress(&input).unwrap();

        // Should be just the mask (all zeros) + no data
        assert_eq!(compressed.len(), 4);
        assert_eq!(&compressed[0..4], &[0, 0, 0, 0]);
    }

    #[test]
    fn test_compress_all_nonzero() {
        let mut input = [0u8; 32];
        for i in 0..32 {
            input[i] = (i + 1) as u8;
        }

        let compressed = compress(&input).unwrap();

        // Mask (all ones) + 32 bytes = 36 bytes
        assert_eq!(compressed.len(), 36);

        // Mask should be all 1s
        let mask = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
        assert_eq!(mask, 0xFFFFFFFF);

        // Data should match input
        for i in 0..32 {
            assert_eq!(compressed[4 + i], input[i]);
        }
    }

    #[test]
    fn test_compress_sparse() {
        let mut input = [0u8; 32];
        input[0] = 5;  // bit 0
        input[3] = 3;  // bit 3
        input[8] = 7;  // bit 8

        let compressed = compress(&input).unwrap();

        // Mask + 3 non-zero bytes + 1 padding = 8 bytes
        assert_eq!(compressed.len(), 8);

        // Check mask: bits 0, 3, 8 set
        let mask = u32::from_le_bytes([compressed[0], compressed[1], compressed[2], compressed[3]]);
        assert_eq!(mask, (1 << 0) | (1 << 3) | (1 << 8));

        // Check non-zero bytes in order
        assert_eq!(compressed[4], 5);
        assert_eq!(compressed[5], 3);
        assert_eq!(compressed[6], 7);
    }

    #[test]
    fn test_decompress_round_trip() {
        let mut input = [0u8; 32];
        input[0] = 42;
        input[15] = 128;
        input[31] = 255;

        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_decompress_all_zeros() {
        let compressed = vec![0u8, 0, 0, 0]; // Mask = 0
        let decompressed = decompress(&compressed).unwrap();

        assert_eq!(decompressed, [0u8; 32]);
    }

    #[test]
    fn test_compressed_size() {
        let zeros = [0u8; 32];
        assert_eq!(compressed_size(&zeros), 4); // Just mask

        let mut sparse = [0u8; 32];
        sparse[0] = 1;
        sparse[1] = 2;
        sparse[2] = 3;
        assert_eq!(compressed_size(&sparse), 8); // mask(4) + 3 bytes + 1 padding

        let full: Vec<u8> = (1..=32).collect();
        assert_eq!(compressed_size(&full), 36); // mask(4) + 32 bytes
    }

    #[test]
    fn test_is_compressible() {
        // All zeros - very compressible
        assert!(is_compressible(&[0u8; 32]));

        // All non-zero - not compressible (overhead > savings)
        let full: Vec<u8> = (1..=32).collect();
        assert!(!is_compressible(&full));

        // Few zeros - borderline
        let mut few_zeros = [1u8; 32];
        few_zeros[0] = 0;
        few_zeros[1] = 0;
        assert!(!is_compressible(&few_zeros)); // Only 2 zeros, not worth it

        // More zeros - compressible
        let mut more_zeros = [0u8; 32];
        more_zeros[0] = 1;
        more_zeros[1] = 2;
        assert!(is_compressible(&more_zeros)); // 30 zeros, definitely worth it
    }

    #[test]
    fn test_wrong_input_size() {
        assert!(compress(&[0u8; 31]).is_none());
        assert!(compress(&[0u8; 33]).is_none());
        assert!(decompress(&[0u8; 3]).is_none()); // Too short for mask
    }
}
