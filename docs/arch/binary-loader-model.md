# Binary Loader Model -- Design Note

**Subsystem:** 8 (Phase 1b)
**Tag:** `phase1-subsys-parser-arch`
**Spec:** [../superpowers/specs/2026-04-23-subsys8-parser-design.md](../superpowers/specs/2026-04-23-subsys8-parser-design.md)
**Audit:** [subsys8-audit.md](subsys8-audit.md)

This document is the mandatory per-seam design note required by the
parent device-family refactor. It explains where arch-specific data
lives after Stage 8a's migrations, what the `BinaryLoader` trait's
surface is (if any), and what an AIE1 port of the parser would
look like.

Filled in after Task 2 (audit) lands and Task 5 (trait decision)
lands.

---

## What lives where

(Filled in at Task 5. Expected table: per-data-item, which module
owns it after migration -- archspec constants, archspec tables,
xdna-emu types, etc.)

## Trait surface

(Filled in at Task 5. One of: populated with 1--3 methods, empty
anchor mirroring `IsaExecutor`, or "no trait at all" mirroring
`IsaDecoder`. Reasoning tied to audit §9 rejection table.)

## The shape-vs-values rule, applied to parser

(Filled in at Task 5. Applies Subsystem 6's "types from toolchain"
and Subsystem 7's "data vs algorithms" framings to parser concerns.)

## What would AIE1 look like?

(Filled in at Task 5 -- audit §8 supplies the projection as part of
its closing summary; the design note expands to ~200--300 words.)

## Alternatives rejected

(Filled in at Task 5. Expected content: any trait shape the audit
considered and rejected, mirroring isa-execute-model.md §"Alternatives
rejected" structure.)
