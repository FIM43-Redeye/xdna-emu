// Mode-2 ZOL emission probes.
//
// Goal: discover what loop shapes Peano emits ZLS / MV-LC / JNZD for, so we
// can pick fixture kernels that exercise the LC bit-28 flag without writing
// raw assembly. Compile this with peano clang at -O2 targeting aie2, then
// disassemble the .o to see which functions ended up with a hardware loop.
//
// Build:
//   /home/triple/npu-work/llvm-aie/install/bin/clang \
//       --target=aie2-none-unknown-elf -O2 -c probes.cc -o probes.o
//   /home/triple/npu-work/llvm-aie/install/bin/llvm-objdump -d probes.o

#include <stdint.h>

extern "C" {

// Fixed small count, trivial body - almost certainly fully unrolled.
__attribute__((noinline))
void loop_fixed_8(int32_t *__restrict out) {
  for (int32_t i = 0; i < 8; i++)
    out[i] = i;
}

// Fixed medium count, trivial body - may unroll, may emit ZOL.
__attribute__((noinline))
void loop_fixed_64(int32_t *__restrict out) {
  for (int32_t i = 0; i < 64; i++)
    out[i] = i;
}

// Runtime count - cannot unroll without a peeled prelude. Should be ZOL.
__attribute__((noinline))
int32_t loop_runtime(const int32_t *__restrict in, int32_t n) {
  int32_t acc = 0;
  for (int32_t i = 0; i < n; i++)
    acc += in[i];
  return acc;
}

// Nested loops - does the inner one stay ZOL while the outer is also ZOL?
// (LC is a single register, so nesting needs save/restore around the inner.)
__attribute__((noinline))
int32_t loop_nested(const int32_t *__restrict in, int32_t outer, int32_t inner) {
  int32_t acc = 0;
  for (int32_t i = 0; i < outer; i++) {
    for (int32_t j = 0; j < inner; j++) {
      acc += in[i * inner + j];
    }
  }
  return acc;
}

// Single-iteration edge case - can Peano produce LC=1?
__attribute__((noinline))
int32_t loop_n1(const int32_t *__restrict in) {
  int32_t acc = 0;
  for (int32_t i = 0; i < 1; i++)
    acc += in[i];
  return acc;
}

} // extern "C"
