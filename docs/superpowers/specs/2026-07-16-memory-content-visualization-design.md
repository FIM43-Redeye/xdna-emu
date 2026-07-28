# Memory Content Visualization -- Design

**Date:** 2026-07-16
**Arc:** Visual debugger, architecture view (extends the tile floor-plan)
**Status:** Approved (design), pending implementation

## Problem

The floor-plan's memory block is the biggest block on a tile and currently
shows only a static label ("MEM 64K"). It wastes the space and tells you
nothing about what memory actually holds or how full it is. Memory is where the
"so much time waiting on DMA" story physically lands -- buffers filling and
draining -- and we render none of it.

## Goal

Render a tile's data memory as a **spatial content map**: an RGBA texture where
each u32 word becomes one pixel, laid out in address order. Untouched (zero)
memory stays background; written data splashes its actual bytes as color at its
actual address. Memory "fills up" as the kernel writes, and the color *is* the
content -- a literal binary waterfall.

## Non-Goals (deferred)

- **Class / semantic coloring** (ASCII vs code vs pointer). A great later toggle
  once the raw value map exists; interpretive, not the default.
- **byte3 -> luminance-lift** alternative encoding (see Encoding). We ship the
  alpha encoding first.
- **Core-internals detail view** (showing what executes inside a core, feasible
  because core memory is small). Separate later arc -- viable product first.
- **Locality-preserving layouts** (Hilbert curve). Row-major is the v1.
- **Locks waiter-highlighting** and other floor-plan block upgrades -- unrelated.

## Encoding (approved)

One u32 word -> one pixel. For the four memory bytes at address `4k` in address
order `[b0, b1, b2, b3]`:

```
R = b0
G = b1
B = b2
A = (b0|b1|b2|b3) == 0  ?  0  :  max(b3, FLOOR)     // FLOOR ~ 96, tunable
```

- **Byte -> channel bijection**: all 32 bits represented, nothing discarded.
- **Zero word -> fully transparent -> tile background.** Preserves the "fills
  up" read: untouched memory stays dark.
- **Visibility floor**: any non-zero word is never below `FLOOR` alpha, so real
  data whose high byte is zero (e.g. a small value `0x000000FF`) can never
  vanish. `FLOOR` is a single tunable constant in one place.
- **byte3 becomes an emphasis / magnitude channel**: small values glow faint,
  large-magnitude words (pointers, big ints) render solid. Little-endian byte3
  is the MSB, so small negatives (`0xFFxxxxxx`) go bright -- sign becomes
  visible for free.
- Pure function `word_to_pixel([u8;4], floor: u8) -> Color32`, unit-tested.
- The texture composites over the tile background painted first in the block.

## Layout (power-of-two, address-ordered)

Row-major: address 0 at top-left, increasing left-to-right then top-to-bottom.
Pixel count = `mem_size / 4`. Logical image dimensions are power-of-two on both
axes so data strides align to pixel columns:

- `width  = smallest power of two >= sqrt(pixel_count)`
- `height = pixel_count / width`

Concretely:

- **Compute, 64 KB = 16384 words -> 128 x 128.**
- **Mem, 512 KB = 131072 words -> 512 x 256.**
- **Shim / any tile with `mem_size == 0`**: no texture; keep the existing block
  (DDR/NoC for shim).

Pure function `memory_dims(pixel_count) -> (u32, u32)`, unit-tested (power-of-two
both axes; product == pixel_count for power-of-two inputs).

Rendering uses **nearest-neighbor** filtering (crisp: strides stay razor-sharp
when the GPU scales the logical image into the on-screen block).

## Architecture -- texture state lives outside the paint atom

Textures are stateful: upload once, keep a `TextureHandle`, rebuild only when
memory changes. That does not fit the *stateless* `floorplan()` atom, so:

- **New module `src/visual/memviz.rs`** owns:
  - the pure `word_to_pixel` and `memory_dims` functions + `FLOOR`;
  - `fn build_image(mem: &[u8]) -> egui::ColorImage` (CPU-side, testable);
  - a cache struct, e.g. `MemoryTextures { map: HashMap<(u8,u8), (u64, TextureHandle)> }`
    with `fn texture(&mut self, ctx, col, row, mem: &[u8], gen: u64) -> TextureId`
    that rebuilds via `ctx.load_texture(.., TextureOptions::NEAREST)` only when
    the stored `gen` differs from the tile's `data_memory_gen()`.
- **`DebuggerApp`** owns one `MemoryTextures`. In `detail::show`, for the
  selected tile it fetches `array.get(col,row)` -> `data_memory()` + gen, asks
  the cache for a `TextureId`, and passes it into the floor-plan.
- **`floorplan()`** gains `mem_texture: Option<egui::TextureId>` on
  `FloorplanPresentation`. When present and the tile has a banks block, it paints
  the texture (nearest) filling the banks rect over the tile bg, instead of the
  "MEM {n}K" label. When absent (shim, or first frame), it keeps the label.
  The atom stays stateless -- it only blits a handle it is given.

Data reached with existing accessors only: `array.get(col,row)`,
`Tile::data_memory()`, `Tile::data_memory_gen()`. No snapshot copy of 512 KB;
no new emu accessors; `TileSnapshot` is unchanged.

## Bigger floor-plan (rides along)

The current detail-panel floor-plan is a fixed 180 px tall rect, which starves
the DMA and lock sub-blocks and leaves no room for a memory texture. Raise it
(target ~260-320 px, or responsive to available panel height) so the memory
texture, DMA channels, and lock-map are all legible. This absorbs the "tiles
need to be bigger / DMA looks bodged" feedback.

## Testing

- `word_to_pixel`: zero -> alpha 0; `[0xFF,0,0,0]` -> R=255 with alpha == FLOOR
  (visible, not 0); `[..,..,..,0xC0]` -> alpha 0xC0; a mid word -> exact channels.
- `memory_dims`: 16384 -> (128,128); 131072 -> (512,256); both axes power-of-two.
- `build_image`: for a small synthetic `mem`, `ColorImage.size` matches
  `memory_dims` and a known written word maps to the expected pixel.
- Cache: rebuilds when gen changes, reuses when it doesn't (unit-test the
  gen-compare logic; the actual `load_texture` needs a ctx, cover via the
  `__run_test_ui` harness or factor the decision out as a pure predicate).
- **Gates:** `cargo build` (default / `--features gui` / `--no-default-features`)
  and `cargo test --lib` all green. GUI never launched; `src/debugger` egui-free.

## Human-Only Review (Maya)

Does written data read as a legible waterfall; are strides visibly aligned; is
the visibility floor high enough that small values show but low enough that the
background still reads as empty; nearest-neighbor crispness at the on-screen
block size; and whether the enlarged floor-plan fixes the DMA/lock crunch.
