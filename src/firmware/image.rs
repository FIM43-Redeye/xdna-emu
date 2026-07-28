//! `$PS1` PSP-signed firmware container loader.
//!
//! The image is signed but NOT encrypted or compressed (recon
//! `build/experiments/firmware-re/INFODUMP.md`). We bypass signing and expose
//! the plaintext payload as a base-0 addressable byte image: for the base-0
//! `.text`/`.rodata` segment, file offset == link address.

use crate::firmware::error::FirmwareError;

const MAGIC_OFFSET: usize = 0x10;
const MAGIC: &[u8; 4] = b"$PS1";
const SIZE_OFFSET: usize = 0x14;
const HEADER_END: usize = 0x18;
const HEADER_SIZE: usize = 0x100;

#[derive(Debug)]
pub struct FirmwareImage {
    payload: Vec<u8>,
    payload_size: u32,
}

impl FirmwareImage {
    pub fn parse(raw: &[u8]) -> Result<Self, FirmwareError> {
        if raw.len() < HEADER_END {
            return Err(FirmwareError::Truncated { offset: 0, needed: HEADER_END, got: raw.len() });
        }
        let found: [u8; 4] = raw[MAGIC_OFFSET..MAGIC_OFFSET + 4].try_into().unwrap();
        if &found != MAGIC {
            return Err(FirmwareError::BadMagic { offset: MAGIC_OFFSET, found });
        }
        let payload_size = u32::from_le_bytes(raw[SIZE_OFFSET..SIZE_OFFSET + 4].try_into().unwrap());
        let signed_size = HEADER_SIZE.checked_add(payload_size as usize);
        if payload_size == 0 || signed_size.is_none_or(|size| size > raw.len()) {
            return Err(FirmwareError::SizeMismatch { header: payload_size, file: raw.len() });
        }
        // The field is the signed BODY size after the fixed 0x100-byte header,
        // matching PSPTool HeaderFile::get_signed_bytes(). Preserve the whole
        // container for diagnostics; signed_size() excludes only the signature.
        Ok(Self { payload: raw.to_vec(), payload_size })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.payload
    }

    pub fn payload_size(&self) -> u32 {
        self.payload_size
    }

    pub fn signed_size(&self) -> usize {
        HEADER_SIZE + self.payload_size as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal valid $PS1 container: fixed 0x100-byte header, signed
    // body, then an inert signature trailer.
    fn build_image(payload: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; HEADER_SIZE];
        v[0x10..0x14].copy_from_slice(b"$PS1");
        let size = payload.len() as u32;
        v[0x14..0x18].copy_from_slice(&size.to_le_bytes());
        v.extend_from_slice(payload);
        v.extend_from_slice(&[0u8; 0x100]);
        v
    }

    #[test]
    fn parses_valid_container_and_exposes_base0_bytes() {
        let raw = build_image(&[0xde, 0xad, 0xbe, 0xef]);
        let img = FirmwareImage::parse(&raw).expect("valid image");
        // base-0 addressable: the "$PS1" magic is still visible at 0x10
        assert_eq!(&img.bytes()[0x10..0x14], b"$PS1");
        assert_eq!(img.payload_size(), 4);
        assert_eq!(img.signed_size(), 0x104);
    }

    #[test]
    fn ps1_size_field_counts_the_signed_body_after_the_header() {
        let mut raw = vec![0u8; 0x100 + 4 + 0x100];
        raw[0x10..0x14].copy_from_slice(b"$PS1");
        raw[0x14..0x18].copy_from_slice(&4u32.to_le_bytes());
        raw[0x100..0x104].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);

        let img = FirmwareImage::parse(&raw).expect("body-sized PS1 image");

        assert_eq!(img.payload_size(), 4);
        assert_eq!(&img.bytes()[0x100..0x104], &[0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut raw = build_image(&[0x00]);
        raw[0x10] = b'X';
        let err = FirmwareImage::parse(&raw).unwrap_err();
        assert!(matches!(err, FirmwareError::BadMagic { offset: 0x10, .. }), "got {err}");
    }

    #[test]
    fn rejects_truncated_before_header() {
        let err = FirmwareImage::parse(&[0u8; 0x12]).unwrap_err();
        assert!(matches!(err, FirmwareError::Truncated { .. }), "got {err}");
    }

    #[test]
    fn rejects_declared_payload_beyond_the_supplied_file() {
        let mut raw = build_image(&[0u8; 4]);
        let declared = raw.len() as u32 + 1;
        raw[0x14..0x18].copy_from_slice(&declared.to_le_bytes());
        let err = FirmwareImage::parse(&raw).unwrap_err();
        assert!(matches!(err, FirmwareError::SizeMismatch { .. }), "got {err}");
    }

    #[test]
    fn rejects_an_empty_signed_body() {
        let mut raw = build_image(&[]);
        raw[0x14..0x18].copy_from_slice(&0u32.to_le_bytes());
        let err = FirmwareImage::parse(&raw).unwrap_err();
        assert!(matches!(err, FirmwareError::SizeMismatch { .. }), "got {err}");
    }
}
