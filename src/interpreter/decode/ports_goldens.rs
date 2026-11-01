//! Port facts of the collision-probe instruction catalogue (`tools/gen_collision_probe.py`),
//! decoded from the real instruction bytes.
//!
//! `write_port`, `mover` and the accumulator read ports are *derived* from the resolved TableGen
//! itineraries (`decode::ports`); the sets pinned here are exactly the ones NPU1 silicon showed
//! behaving differently in the collision probes (`build/experiments/collisions`, r1 == r2). The
//! tests may name units: the emulator code never does.

use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::traits::Decoder;

/// `(catalogue key, assembly, instruction bytes)` of every probe instruction with the sample
/// register assignment of `gen_collision_probe.sample_regs`.
const CATALOGUE_BYTES: &[(&str, &str, &str)] = &[
    ("vmov_cm", "vmov cm1, cm0", "d9 06 90 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_bm", "vmov bml1, bml0", "d9 06 a0 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_x_bm", "vmov x8, bml0", "d9 06 00 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_s16_s32", "vsrs.s16.s32 wl8, bml0, s1", "99 41 1d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_s32_s64", "vsrs.s32.s64 x8, cm0, s1", "99 c0 3d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrsm_s16_s32", "vsrsm.s16.s32 wl8, bml0, s1", "59 27 10 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_d16_s32", "vsrs.d16.s32 wl8, bml0, s1", "99 01 1d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vconv_bf16_fp32", "vconv.bf16.fp32 wl8, bml0", "d9 01 00 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vst_amll", "vst amll0, [p1, #-128]", "59 80 f2 09 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vst_amhh", "vst amhh0, [p1, #-128]", "59 83 f2 09 37 88 03 00 00 00 00 00 00 00 00 00"),
    (
        "vst_srs_s16_s32",
        "vst.srs.s16.s32 bml0, s1, [p1, #-128]",
        "99 41 8d 09 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vst_conv_bf16_fp32",
        "vst.conv.bf16.fp32 bml0, [p1, #-128]",
        "d9 01 8d 09 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    ("vadd_i", "vadd cm2, cm0, cm1, r1", "09 48 38 08 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vadd_f", "vadd.f bml2, bml0, bml1, r3", "89 88 00 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vaddmac_i", "vaddmac cm1, cm1, cm0, x0, x1, r1", "09 40 c0 08 37 88 03 00 00 00 00 00 00 00 00 00"),
    (
        "vaddmac_f",
        "vaddmac.f bml1, bml1, bml0, x2, x3, r3",
        "c9 c0 88 18 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vaddmsc_f",
        "vaddmsc.f bml1, bml1, bml0, x2, x3, r3",
        "c9 c1 88 18 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vaddmac_i_sparse",
        "vaddmac cm1, cm1, cm0, x0, qx1, r2",
        "09 40 c3 10 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vaddmac_f_sparse",
        "vaddmac.f bml1, bml1, bml0, y3, qx3, r4",
        "c9 c0 b7 20 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    ("vmul_i", "vmul cm0, x0, x1, r1", "09 40 80 0f 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmul_f", "vmul.f bml0, x2, x3, r3", "49 c0 c8 1f 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmac_i", "vmac cm1, cm0, x0, x1, r1", "09 44 00 08 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmac_f", "vmac.f bml1, bml0, x2, x3, r3", "49 c4 08 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmac_i_sparse", "vmac cm1, cm0, x0, qx1, r2", "09 44 03 10 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmac_f_sparse", "vmac.f bml1, bml0, y3, qx3, r4", "49 c4 37 20 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmsc_f", "vmsc.f bml1, bml0, x2, x3, r3", "49 c5 08 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vclr_cm", "vclr cm0", "09 02 3c 00 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vclr_bm", "vclr bml0", "49 00 fc 07 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vneg_i", "vneg cm1, cm0, r1", "09 05 7c 08 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vneg_f", "vneg.f bml1, bml0, r3", "49 04 3c 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vups", "vups.s32.s16 bml0, wl10, s3", "d9 2f 35 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vlda_ups", "vlda.ups.s32.s16 bml0, s3, [p4, #0]", "19 41 0f 04 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vlda_conv", "vlda.conv.fp32.bf16 bml0, [p4, #0]", "19 80 0c 04 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vconv_fp32_bf16", "vconv.fp32.bf16 bml0, wl10", "d9 2f 00 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_bm_x", "vmov bml0, x10", "d9 56 2c 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_s8_s32", "vsrs.s8.s32 wl8, cm0, s1", "99 40 1d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_d32_s64", "vsrs.d32.s64 x8, cm0, s1", "99 80 3d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_s16_s64", "vsrs.s16.s64 wl8, cm0, s1", "99 c0 1d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_d16_s64", "vsrs.d16.s64 wl8, cm0, s1", "99 80 1d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrsm_d16_s32", "vsrsm.d16.s32 wl8, bml0, s1", "59 07 10 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrsm_s32_s64", "vsrsm.s32.s64 wl8, bml0, s1", "59 67 10 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrsm_d32_s64", "vsrsm.d32.s64 wl8, bml0, s1", "59 47 10 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    (
        "vst_srs_d16_s32",
        "vst.srs.d16.s32 bml0, s1, [p1, #-128]",
        "99 01 8d 09 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vst_srs_s8_s32",
        "vst.srs.s8.s32 cm0, s1, [p1, #-128]",
        "99 40 8d 09 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vst_srs_s16_s64",
        "vst.srs.s16.s64 cm0, s1, [p1, #-128]",
        "99 c0 8d 09 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vst_srs_d16_s64",
        "vst.srs.d16.s64 cm0, s1, [p1, #-128]",
        "99 80 8d 09 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vst_srs_s32_s64",
        "vst.srs.s32.s64 bml0, s1, [p1, #-128]",
        "99 c1 8d 09 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    (
        "vst_srs_d32_s64",
        "vst.srs.d32.s64 bml0, s1, [p1, #-128]",
        "99 81 8d 09 37 88 03 00 00 00 00 00 00 00 00 00",
    ),
    ("vpack_s8_s16", "vpack.s8.s16 wl8, x0", "99 02 3f 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vpack_d8_d16", "vpack.d8.d16 wl8, x0", "99 02 3d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vpack_s4_s8", "vpack.s4.s8 wl8, x0", "99 02 3e 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vlda_w", "vlda wl8, [p0, #32]", "d9 a1 06 00 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vlda_128", "vlda.128 wl8, [p0]", "d9 21 80 00 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vfloor_am", "vfloor.s32.bf16 x8, amll0, s1", "59 00 41 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vfloor_w", "vfloor.s32.bf16 x8, wl0, s1", "59 00 01 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vband", "vband x8, x0, x1", "d9 80 09 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbor", "vbor x8, x0, x1", "d9 80 0b 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vadd_32", "vadd.32 x8, x0, x1", "d9 00 00 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsub_32", "vsub.32 x8, x0, x1", "d9 00 01 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vshift_align", "vshift.align x8, x0, s0, x1, r1", "f9 80 10 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbcst_32", "vbcst.32 x8, r12", "b9 07 46 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_w_w", "vmov wl8, wl0", "59 07 0c 19 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_x_x", "vmov x8, x0", "d9 06 0c 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vshuffle_x", "vshuffle x8, x0, x1, r1", "f9 80 00 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vshuffle_bm", "vshuffle bml0, x0, x1, r1", "f9 80 40 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbcstshfl_x", "vbcstshfl.32 x8, r12, r29", "b9 27 46 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbcstshfl_bm", "vbcstshfl.32 bml0, r12, r29", "b9 67 46 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_am_q", "vmov amll0, q1", "59 87 1e 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_q_am", "vmov q0, amll0", "59 0f 20 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vlda_wh", "vlda wh8, [p0, #32]", "d9 a3 06 00 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vpack_s8_s16_wh", "vpack.s8.s16 wh8, x0", "99 02 7f 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_s16_s32_wh", "vsrs.s16.s32 wh8, bml0, s1", "99 41 5d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrs_s16_s64_wh", "vsrs.s16.s64 wh8, cm0, s1", "99 c0 5d 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vconv_bf16_fp32_wh", "vconv.bf16.fp32 wh8, bml0", "d9 01 40 0c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_w_wh", "vmov wh8, wl0", "59 07 0c 1d 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vsrsm_s16_s32_wh", "vsrsm.s16.s32 wh8, bml0, s1", "59 27 50 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_amlh_q", "vmov amlh0, q1", "59 87 3e 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_amhl_q", "vmov amhl0, q1", "59 87 5e 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vmov_amhh_q", "vmov amhh0, q1", "59 87 7e 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vshuffle_bmh", "vshuffle bmh0, x0, x1, r1", "f9 80 40 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbcstshfl_bmh", "vbcstshfl.32 bmh0, r12, r29", "b9 67 46 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbcstshfl8_bm", "vbcstshfl.8 bml0, r12, r29", "b9 67 06 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbcstshfl16_bm", "vbcstshfl.16 bml0, r12, r29", "b9 67 26 18 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbcstshfl8_x", "vbcstshfl.8 x8, r12, r29", "b9 27 06 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
    ("vbcstshfl16_x", "vbcstshfl.16 x8, r12, r29", "b9 27 26 1c 37 88 03 00 00 00 00 00 00 00 00 00"),
];

fn bytes(hex: &str) -> Vec<u8> {
    hex.split_whitespace().map(|h| u8::from_str_radix(h, 16).unwrap()).collect()
}

/// `key -> "write port | accumulator read port@offset+secondary | mover"` for every catalogue
/// instruction, as decoded; `-` means none.
const EXPECTED: &[(&str, &str)] = &[
    ("vmov_cm", "CM_WM_PORT | CM_RM_PORT@0+- | true"),
    ("vmov_bm", "CM_WM_PORT | CM_RM_PORT@0+- | true"),
    ("vmov_x_bm", "W_WM_PORT | CM_RM_PORT@0+- | true"),
    ("vsrs_s16_s32", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vsrs_s32_s64", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vsrsm_s16_s32", "W_WM_PORT | CM_RM_PORT@0+UPPER_SRS | true"),
    ("vsrs_d16_s32", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vconv_bf16_fp32", "W_WA_PORT | CM_RM_PORT@0+- | false"),
    ("vst_amll", "- | CM_RM_PORT@0+- | false"),
    ("vst_amhh", "- | CM_RM_PORT@0+- | false"),
    ("vst_srs_s16_s32", "- | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vst_conv_bf16_fp32", "- | CM_RM_PORT@0+- | false"),
    ("vadd_i", "CM_WA_PORT | CM_RM_PORT@2+- | false"),
    ("vadd_f", "CM_WA_PORT | CM_RM_PORT@2+- | false"),
    ("vaddmac_i", "CM_WA_PORT | CM_RM_PORT@2+- | false"),
    ("vaddmac_f", "CM_WA_PORT | CM_RM_PORT@2+- | false"),
    ("vaddmsc_f", "CM_WA_PORT | CM_RM_PORT@2+- | false"),
    ("vaddmac_i_sparse", "CM_WA_PORT | CM_RM_PORT@2+- | false"),
    ("vaddmac_f_sparse", "CM_WA_PORT | CM_RM_PORT@2+- | false"),
    ("vmul_i", "CM_WA_PORT | - | false"),
    ("vmul_f", "CM_WA_PORT | - | false"),
    ("vmac_i", "CM_WA_PORT | - | false"),
    ("vmac_f", "CM_WA_PORT | - | false"),
    ("vmac_i_sparse", "CM_WA_PORT | - | false"),
    ("vmac_f_sparse", "CM_WA_PORT | - | false"),
    ("vmsc_f", "CM_WA_PORT | - | false"),
    ("vclr_cm", "CM_WA_PORT | - | false"),
    ("vclr_bm", "CM_WA_PORT | - | false"),
    ("vneg_i", "CM_WA_PORT | - | false"),
    ("vneg_f", "CM_WA_PORT | - | false"),
    ("vups", "CM_WM_PORT | - | false"),
    ("vlda_ups", "CM_WM_PORT | - | false"),
    ("vlda_conv", "CM_WM_PORT | - | false"),
    ("vconv_fp32_bf16", "CM_WM_PORT | - | false"),
    ("vmov_bm_x", "CM_WM_PORT | - | false"),
    ("vsrs_s8_s32", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vsrs_d32_s64", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vsrs_s16_s64", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vsrs_d16_s64", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vsrsm_d16_s32", "W_WM_PORT | CM_RM_PORT@0+UPPER_SRS | true"),
    ("vsrsm_s32_s64", "W_WM_PORT | CM_RM_PORT@0+UPPER_SRS | true"),
    ("vsrsm_d32_s64", "W_WM_PORT | CM_RM_PORT@0+UPPER_SRS | true"),
    ("vst_srs_d16_s32", "- | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vst_srs_s8_s32", "- | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vst_srs_s16_s64", "- | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vst_srs_d16_s64", "- | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vst_srs_s32_s64", "- | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vst_srs_d32_s64", "- | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vpack_s8_s16", "W_WA_PORT | - | false"),
    ("vpack_d8_d16", "W_WA_PORT | - | false"),
    ("vpack_s4_s8", "W_WA_PORT | - | false"),
    ("vlda_w", "W_WA_PORT | - | false"),
    ("vlda_128", "W_WA_PORT | - | false"),
    ("vfloor_am", "W_WA_PORT | CM_RM_PORT@0+- | false"),
    ("vfloor_w", "W_WA_PORT | - | false"),
    ("vband", "W_WM_PORT | - | false"),
    ("vbor", "W_WM_PORT | - | false"),
    ("vadd_32", "W_WM_PORT | - | false"),
    ("vsub_32", "W_WM_PORT | - | false"),
    ("vshift_align", "W_WM_PORT | - | false"),
    ("vbcst_32", "W_WM_PORT | - | false"),
    ("vmov_w_w", "W_WM_PORT | - | false"),
    ("vmov_x_x", "W_WM_PORT | - | false"),
    ("vshuffle_x", "W_WM_PORT | - | false"),
    ("vshuffle_bm", "CM_WM_PORT | - | false"),
    ("vbcstshfl_x", "W_WM_PORT | - | false"),
    ("vbcstshfl_bm", "CM_WM_PORT | - | false"),
    ("vmov_am_q", "CM_WM_PORT | - | false"),
    ("vmov_q_am", "- | CM_RM_PORT@0+- | true"),
    ("vlda_wh", "W_WA_PORT | - | false"),
    ("vpack_s8_s16_wh", "W_WA_PORT | - | false"),
    ("vsrs_s16_s32_wh", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vsrs_s16_s64_wh", "W_WA_PORT | CM_RM_PORT@0+UPPER_SRS | false"),
    ("vconv_bf16_fp32_wh", "W_WA_PORT | CM_RM_PORT@0+- | false"),
    ("vmov_w_wh", "W_WM_PORT | - | false"),
    ("vsrsm_s16_s32_wh", "W_WM_PORT | CM_RM_PORT@0+UPPER_SRS | true"),
    ("vmov_amlh_q", "CM_WM_PORT | - | false"),
    ("vmov_amhl_q", "CM_WM_PORT | - | false"),
    ("vmov_amhh_q", "CM_WM_PORT | - | false"),
    ("vshuffle_bmh", "CM_WM_PORT | - | false"),
    ("vbcstshfl_bmh", "CM_WM_PORT | - | false"),
    ("vbcstshfl8_bm", "CM_WM_PORT | - | false"),
    ("vbcstshfl16_bm", "CM_WM_PORT | - | false"),
    ("vbcstshfl8_x", "W_WM_PORT | - | false"),
    ("vbcstshfl16_x", "W_WM_PORT | - | false"),
];

/// Decode one catalogue instruction and describe its ports with unit names.
fn describe(key: &str) -> String {
    let units = xdna_archspec::aie2::isa::load_from_generated().functional_units;
    let (_, asm, hex) = CATALOGUE_BYTES.iter().find(|(k, _, _)| *k == key).unwrap();
    let decoder = InstructionDecoder::load_default();
    let bundle = decoder.decode(&bytes(hex), 0).unwrap_or_else(|e| panic!("{asm}: {e:?}"));
    let op = bundle
        .slots()
        .iter()
        .flatten()
        .find(|s| !s.is_nop())
        .unwrap_or_else(|| panic!("{asm}: no op"));
    let name = |u: u16| units[u as usize].as_str();
    let write = op.ports.write_port.map(name).unwrap_or("-");
    let read = match op.ports.acc_read {
        Some(r) => format!("{}@{}+{}", name(r.primary), r.offset, r.secondary.map(name).unwrap_or("-")),
        None => "-".into(),
    };
    format!("{write} | {read} | {}", op.ports.mover)
}

#[test]
fn every_catalogue_instruction_has_the_pinned_ports() {
    assert_eq!(EXPECTED.len(), CATALOGUE_BYTES.len());
    for (key, want) in EXPECTED {
        assert_eq!(&describe(key), want, "{key}");
    }
}

/// The instructions that steal a late reader's accumulator read port on silicon (CM_RM steal,
/// probe2/probe3): the moves out of an accumulator and `vsrsm`. The `mover` predicate is
/// derived from the tables (reads an accumulator, writes through the last port of a file); it
/// must select exactly these and none of the non-stealing readers (`vsrs`, `vconv`, `vst*`,
/// `vfloor`), which silicon showed to leave a late reader alone.
#[test]
fn the_mover_predicate_selects_exactly_the_observed_set() {
    let movers: Vec<&str> = EXPECTED
        .iter()
        .filter(|(_, d)| d.ends_with("| true"))
        .map(|(k, _)| *k)
        .collect();
    assert_eq!(
        movers,
        [
            "vmov_cm",
            "vmov_bm",
            "vmov_x_bm",
            "vsrsm_s16_s32",
            "vsrsm_d16_s32",
            "vsrsm_s32_s64",
            "vsrsm_d32_s64",
            "vmov_q_am",
            "vsrsm_s16_s32_wh"
        ]
    );
}

/// The instructions that hold the shifter's second read unit (the wired-OR candidates of
/// probe4/probe10/probe11): every `vsrs`, `vsrsm` and `vst.srs` shape and nothing else.
#[test]
fn the_secondary_read_unit_is_held_by_exactly_the_srs_family() {
    for (key, d) in EXPECTED {
        let has_secondary = !d.split(" | ").nth(1).unwrap().ends_with("+-") && d.contains('@');
        let is_srs = key.starts_with("vsrs") || key.starts_with("vst_srs");
        assert_eq!(has_secondary, is_srs, "{key}: {d}");
    }
}

/// The write-port classification resolves the shuffle family's over-reservation (both a vector
/// and an accumulator port are listed) by the destination register file.
#[test]
fn shuffle_and_broadcast_writes_use_the_port_of_their_destination_file() {
    let get = |k: &str| {
        EXPECTED
            .iter()
            .find(|(key, _)| *key == k)
            .unwrap()
            .1
            .split(" | ")
            .next()
            .unwrap()
    };
    for k in ["vshuffle_x", "vbcstshfl_x", "vbcstshfl8_x", "vbcstshfl16_x"] {
        assert_eq!(get(k), "W_WM_PORT", "{k}");
    }
    for k in
        ["vshuffle_bm", "vshuffle_bmh", "vbcstshfl_bm", "vbcstshfl_bmh", "vbcstshfl8_bm", "vbcstshfl16_bm"]
    {
        assert_eq!(get(k), "CM_WM_PORT", "{k}");
    }
}

/// The write ports derived for each register file, in unit-index order (priority order).
#[test]
fn derived_write_ports_per_register_file() {
    let t = xdna_archspec::aie2::isa::load_from_generated();
    let (vector, accum) = super::ports::PortModel::derive(&t).write_ports();
    let names = |l: Vec<u16>| l.iter().map(|u| t.functional_units[*u as usize].clone()).collect::<Vec<_>>();
    assert_eq!(names(vector), ["W_RS_PORT", "W_WA_PORT", "W_WM_PORT"]);
    assert_eq!(names(accum), ["CM_WA_PORT", "CM_WM_PORT"]);
}
