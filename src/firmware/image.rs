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
        // The whole file (minus the inert signature trailer) is the base-0 image.
        Ok(Self { payload: raw.to_vec(), payload_size })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.payload
    }

    pub fn payload_size(&self) -> u32 {
        self.payload_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a minimal valid $PS1 container: 0x10 bytes of hash, "$PS1",
    // a size field, then payload, then a 0x200-byte signature trailer.
    fn build_image(payload: &[u8]) -> Vec<u8> {
        let mut v = vec![0u8; 0x18];
        v[0x10..0x14].copy_from_slice(b"$PS1");
        let size = (0x18 + payload.len()) as u32; // header+payload, excl. sig
        v[0x14..0x18].copy_from_slice(&size.to_le_bytes());
        v.extend_from_slice(payload);
        v.extend_from_slice(&[0u8; 0x200]); // signature trailer
        v
    }

    #[test]
    fn parses_valid_container_and_exposes_base0_bytes() {
        let raw = build_image(&[0xde, 0xad, 0xbe, 0xef]);
        let img = FirmwareImage::parse(&raw).expect("valid image");
        // base-0 addressable: the "$PS1" magic is still visible at 0x10
        assert_eq!(&img.bytes()[0x10..0x14], b"$PS1");
        assert_eq!(img.payload_size(), (0x18 + 4) as u32);
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
}
