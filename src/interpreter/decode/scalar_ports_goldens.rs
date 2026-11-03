//! Scalar-file write ports of the movd probe instructions (`build/experiments/harness/movd`),
//! decoded from the real instruction bytes.
//!
//! `PortUse::scalar_defs` is *derived* from the resolved TableGen itineraries (`decode::ports`):
//! for every result that lands in a scalar register file, the units the itinerary holds in the
//! cycle before it lands. The expectations below are read off `AIE2Schedule.td` by hand; the
//! tests may name units, the emulator code never does. Note the p/m/d* movers and the vector
//! extracts into them also list `RS_WM_PORT`: the schedule shares that unit between the r and s
//! writers and the others, which silicon does not (see `state::scalar_collision`); the register
//! file of the result keeps them apart.

use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::traits::Decoder;

/// `(assembly, instruction bytes, "File units@land" per scalar result, `;` separated, - = none)`.
const CATALOGUE: &[(&str, &str, &str)] = &[
    ("mov.d1 r1, r2", "59 0e 41 18", "R RS_WM_PORT@2"),
    ("mov.d3 r1, r2", "59 0e 51 18", "R RS_WM_PORT@4"),
    ("mov.d6 r1, r2", "59 1e 61 18", "R RS_WM_PORT@7"),
    ("mov r5, r6", "59 16 43 19", "R P_RM_PORT+RS_WM_PORT@1"),
    ("mov r5, #7", "99 1f 40 19", "R RS_WM_PORT@1"),
    ("movxm r5, #0x55555555", "55 aa 8a 52 55 55", "R RS_WM_PORT@1"),
    ("add.nc r5, r6, #3", "f9 07 43 19", "R RS_WM_PORT@1"),
    ("mov s0, r6", "59 16 b3 18", "S P_RM_PORT+RS_WM_PORT@1"),
    ("mov p4, r6", "59 16 33 1c", "P P_RM_PORT+P_WM_PORT+RS_WM_PORT@1"),
    ("mov p4, #7", "99 1f 30 1c", "P P_WM_PORT+RS_WM_PORT@1"),
    ("mov m1, r6", "59 16 63 18", "M M_WM_PORT+P_RM_PORT+RS_WM_PORT@1"),
    ("mov dn1, r6", "59 16 63 1a", "Dn DN_WM_PORT+P_RM_PORT+RS_WM_PORT@1"),
    ("mov dj1, r6", "59 16 63 1c", "Dj DJ_WM_PORT+P_RM_PORT+RS_WM_PORT@1"),
    ("mov dc1, r6", "59 16 63 1e", "Dc DC_WM_PORT+P_RM_PORT+RS_WM_PORT@1"),
    ("add.nc p4, r6, #3", "f9 07 33 1c", "P P_WM_PORT+RS_WM_PORT@1"),
    ("vextract.s32 r5, x0, r17", "b9 86 4a 19", "R RS_WM_PORT@2"),
    ("vextract.s32 p4, x0, r17", "b9 86 3a 1c", "P P_WM_PORT+RS_WM_PORT@2"),
    ("vlt.s32 r16, x0, x1", "d9 80 0c 18", "R RS_WM_PORT@2"),
    ("veqz.32 r16, x0", "59 80 07 18", "R RS_WM_PORT@2"),
    // the vector result enters through the vector file, only the compare mask is scalar; the
    // vector write unit is held in the same cycle and listed with it
    ("vmax_lt.s32 x2, r16, x0, x1", "d9 80 0a 19", "R RS_WM_PORT+W_WM_PORT@2"),
    ("add r5, r6, r9", "99 90 8a 11", "R R_WX_PORT@1"),
    ("add r5, r6, #3", "19 0f 8a 11", "R R_WX_PORT@1"),
    ("mul r5, r6, r9", "99 9f 8a 11", "R R_WX_PORT@2"),
    ("lda r5, [p0, #0]", "59 8a 02 00", "R R_WA_PORT@7"),
    ("lda p5, [p0, #0]", "d9 ae 02 00", "P P_WM_PORT@7"),
    ("lda m2, [p0, #0]", "59 85 02 00", "M M_WM_PORT@7"),
    ("lda dn2, [p0, #0]", "59 95 02 00", "Dn DN_WM_PORT@7"),
    ("mova r5, #5", "59 0a 05 00", "R R_WA_PORT@1"),
    ("mova p4, #5", "d9 26 05 00", "P P_WM_PORT@1"),
    // a control register is no scalar file
    ("movx crSat, r9", "19 20 64 12", "-"),
];

fn bytes(hex: &str) -> Vec<u8> {
    hex.split_whitespace().map(|h| u8::from_str_radix(h, 16).unwrap()).collect()
}

fn describe(asm: &str, hex: &str) -> String {
    let units = xdna_archspec::aie2::isa::load_from_generated().functional_units;
    let decoder = InstructionDecoder::load_default();
    let bundle = decoder.decode(&bytes(hex), 0).unwrap_or_else(|e| panic!("{asm}: {e:?}"));
    let op = bundle
        .slots()
        .iter()
        .flatten()
        .find(|s| !s.is_nop())
        .unwrap_or_else(|| panic!("{asm}: no op"));
    let defs: Vec<String> = op
        .ports
        .scalar_defs
        .iter()
        .map(|d| {
            let mut names: Vec<&str> = d.units.iter().map(|u| units[*u as usize].as_str()).collect();
            names.sort_unstable();
            format!("{:?} {}@{}", d.file, names.join("+"), d.land)
        })
        .collect();
    if defs.is_empty() {
        "-".into()
    } else {
        defs.join(";")
    }
}

#[test]
fn every_scalar_instruction_has_the_pinned_write_ports() {
    for (asm, hex, want) in CATALOGUE {
        assert_eq!(&describe(asm, hex), want, "{asm}");
    }
}
