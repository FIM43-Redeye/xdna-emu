//! Where an instruction form's signedness comes from.
//!
//! Several AIE2 vector instructions come in pairs of forms, written `s` and `d` in the
//! mnemonic (`vunpack.s16.s8` and `vunpack.d16.d8`, `vpack.s8.s16` and `vpack.d8.d16`). The
//! `s` form has the signedness fixed by its encoding (`sgn = 1`: signed). The `d` form
//! (`sgn = 0`) takes it at run time from a control register that the TableGen record lists in
//! `Uses` (`crUnpackSign` for unpack, `crPackSign` for pack).
//!
//! The control register is data, not an assumption: AIE2P splits it differently (`unpackSign0`
//! and `unpackSign1`, `packSign0` and `packSign1`), so [`SignSource::Register`] names the
//! register a form reads and callers resolve its value through their own control-register file.

/// Where a form's signedness comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignSource {
    /// Fixed by the encoding.
    Fixed(bool),
    /// Read at run time from a control register: its TableGen name and hardware id.
    Register { name: &'static str, id: u8 },
}

impl SignSource {
    /// Resolve the signedness, reading the control register through `read` when the form
    /// takes it from one. A register value of 1 is signed, 0 unsigned (silicon, unpack and
    /// pack: values 0 and 1 only; higher bits are unobserved and ignored).
    pub fn is_signed(self, read: impl FnOnce(u8) -> u32) -> bool {
        match self {
            SignSource::Fixed(signed) => signed,
            SignSource::Register { id, .. } => read(id) & 1 != 0,
        }
    }

    /// The sign source of a form from its `sgn` encoding bit and the sign control register
    /// its record lists in `Uses`, if any. A register on a form with `sgn = 1` is a table
    /// error: the encoding would not need it.
    pub fn from_encoding(sgn: u8, register: Option<(&'static str, u8)>) -> SignSource {
        match register {
            Some((name, id)) => {
                assert_eq!(sgn, 0, "{name}: a sign register on a form whose encoding fixes the sign");
                SignSource::Register { name, id }
            }
            None => SignSource::Fixed(sgn == 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_is_read_only_for_register_forms() {
        let reads = std::cell::Cell::new(0);
        let read = |v: u32| {
            let reads = &reads;
            move |_id: u8| {
                reads.set(reads.get() + 1);
                v
            }
        };
        assert!(SignSource::Fixed(true).is_signed(read(0)));
        assert!(!SignSource::Fixed(false).is_signed(read(1)));
        assert_eq!(reads.get(), 0, "a fixed-sign form must not read the register");
        let reg = SignSource::Register { name: "crPackSign", id: 5 };
        assert!(reg.is_signed(read(1)));
        assert!(!reg.is_signed(read(0)));
        assert!(!reg.is_signed(read(2)), "only bit 0 decides");
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn from_encoding_follows_sgn_and_uses() {
        assert_eq!(SignSource::from_encoding(1, None), SignSource::Fixed(true));
        assert_eq!(SignSource::from_encoding(0, None), SignSource::Fixed(false));
        assert_eq!(
            SignSource::from_encoding(0, Some(("crPackSign", 5))),
            SignSource::Register { name: "crPackSign", id: 5 }
        );
    }

    #[test]
    #[should_panic(expected = "fixes the sign")]
    fn a_register_on_a_fixed_sign_encoding_is_an_error() {
        SignSource::from_encoding(1, Some(("crPackSign", 5)));
    }
}
