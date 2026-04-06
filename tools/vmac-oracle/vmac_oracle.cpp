// vmac_oracle.cpp -- Compute exact hardware vmac output using aietools C++ ISS.
//
// Usage: vmac_oracle <A_hex_128B> <B_hex_64B> <mask_hex_16B> <config_hex>
//   A: 256 hex chars (128 bytes, dense operand xs1)
//   B: 128 hex chars (64 bytes, sparse compressed qxs2)
//   mask: 32 hex chars (16 bytes, 128-bit sparsity mask)
//   config: hex config word (e.g. 353)
// Output: 16 accumulator lanes as hex u64 values, one per line.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vbit.h"
#include "me_iss_types.h"

namespace me_primitive {
// Lines 4961-14870 from me_inline_primitives.h
#include "vmac_functions.inc"
} // namespace me_primitive

static void hex_to_bytes(const char *hex, uint8_t *out, size_t nbytes) {
    for (size_t i = 0; i < nbytes; i++) {
        unsigned int val;
        sscanf(hex + 2*i, "%02x", &val);
        out[i] = (uint8_t)val;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        fprintf(stderr, "Usage: vmac_oracle <A_hex> <B_hex> <mask_hex> <config_hex>\n");
        return 1;
    }

    uint8_t A_bytes[128] = {};
    uint8_t B_bytes[64] = {};
    uint8_t mask_bytes[16] = {};
    uint32_t config = (uint32_t)strtoul(argv[4], nullptr, 16);

    hex_to_bytes(argv[1], A_bytes, 128);
    hex_to_bytes(argv[2], B_bytes, 64);
    hex_to_bytes(argv[3], mask_bytes, 16);

    // CRITICAL: Operand mapping (from me_inline_primitives.h mac() at line 14820):
    //   v16w32 al = DENSE operand (xs1, 512 bits) -> zero-extended to v32w32 -> CROSSBAR
    //   w640 mask_b = SPARSE compressed data (upper 512) + mask (lower 128) -> Y-PERM
    //   The DENSE data goes to the CROSSBAR, SPARSE goes to Y-PERM.

    // Build v16w32 for low 64 bytes of DENSE data (integer path uses this)
    me_primitive::v16w32 al_dense;
    for (int i = 0; i < 16; i++) {
        uint32_t word = 0;
        memcpy(&word, A_bytes + i*4, 4);
        al_dense.val.elem(i) = VBit<32, true>((int64_t)(int32_t)word);
    }

    // Build v32w32 for full 128 bytes (bf16 path needs all 64 elements)
    me_primitive::v32w32 a_full;
    for (int i = 0; i < 32; i++) {
        uint32_t word = 0;
        memcpy(&word, A_bytes + i*4, 4);
        a_full.val.elem(i) = VBit<32, true>((int64_t)(int32_t)word);
    }

    // Build w640 = {sparse_data[512], mask[128]} with mask in LOWER 128 bits
    me_primitive::w640 mask_b;
    {
        uint64_t mask_lo, mask_hi;
        memcpy(&mask_lo, mask_bytes, 8);
        memcpy(&mask_hi, mask_bytes + 8, 8);
        VBit<128, true> mask_part = concat(
            VBit<64, true>((int64_t)mask_hi),
            VBit<64, true>((int64_t)mask_lo)
        );
        VBit<512, true> data_part(0);
        for (int i = 0; i < 8; i++) {
            uint64_t qw;
            memcpy(&qw, B_bytes + i*8, 8);
            for (int b = 0; b < 64; b++) {
                data_part.deposit((qw >> b) & 1, i*64 + b);
            }
        }
        // Verify: print first 8 bytes of data_part for debugging
        fprintf(stderr, "data_part byte 0-7: ");
        for (int i = 0; i < 8; i++) {
            uint8_t byte = 0;
            for (int b = 0; b < 8; b++) {
                if (data_part.extract(i*8 + b)) byte |= (1 << b);
            }
            fprintf(stderr, "%02x ", byte);
        }
        fprintf(stderr, "\nB_bytes  byte 0-7: ");
        for (int i = 0; i < 8; i++) fprintf(stderr, "%02x ", B_bytes[i]);
        fprintf(stderr, "\n");
        mask_b.val = concat(data_part, mask_part);
    }

    // Config word
    me_primitive::w32 cfg;
    cfg.val = VBit<32, true>((int64_t)(int32_t)config);

    me_primitive::u1 zero_acc_flag;
    zero_acc_flag.val = VBit<1, false>(1); // force zero acc

    me_primitive::u1 sub0, sub1, sub2;
    sub0.val = VBit<1, false>(0);
    sub1.val = VBit<1, false>(0);
    sub2.val = VBit<1, false>(0);

    // Detect bf16 mode: amode = (config >> 1) & 3; bf16 when amode == 2
    bool is_bfloat = ((config >> 1) & 3) == 2;

    if (is_bfloat) {
        // bf16 uses mul_bf (zero_acc=1 internally) which returns v8w64.
        // mul_bf(v16w32 a, w640 b, w32 cfg, u1 sub0, v5u1 mask, v5u1& of)
        me_primitive::v5u1 flag_mask;
        for (int i = 0; i < 5; i++)
            flag_mask.val.elem(i) = VBit<1, false>(1);
        me_primitive::v5u1 oflags;

        me_primitive::v8w64 result = me_primitive::mul_bf(
            a_full, mask_b, cfg, sub0, flag_mask, oflags
        );

        // Output 8 x u64 (each holds 2 fp32 values: lo32 + hi32)
        // Expand to 16 lines for consistency with integer path
        for (int i = 0; i < 8; i++) {
            uint64_t val = result.val.elem(i).to_unsigned();
            uint32_t lo = (uint32_t)(val & 0xFFFFFFFF);
            uint32_t hi = (uint32_t)(val >> 32);
            // Lane 2*i gets lo32, lane 2*i+1 gets hi32
            printf("%016lx\n", (uint64_t)lo);
            printf("%016lx\n", (uint64_t)hi);
        }
    } else {
        // Integer path: mac() returns v16w64
        me_primitive::v16w64 acc1, acc2;
        for (int i = 0; i < 16; i++) {
            acc1.val.elem(i) = VBit<64, true>(0);
            acc2.val.elem(i) = VBit<64, true>(0);
        }

        me_primitive::v16w64 result = me_primitive::mac(
            al_dense, mask_b, acc1, acc2,
            zero_acc_flag, cfg, sub0, sub1, sub2
        );

        for (int i = 0; i < 16; i++) {
            uint64_t val = result.val.elem(i).to_unsigned();
            printf("%016lx\n", val);
        }
    }

    return 0;
}
