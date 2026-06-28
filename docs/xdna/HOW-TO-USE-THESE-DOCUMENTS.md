AI AGENTS: Use the compact folders (am025-compact, am029-compact) and the text
documentation (am020-aie-ml, am027-aie-ml-v2). The compact folders contain
prettified, clean register logic distilled from the AMD register references.

The original AMD register references were HTML (human-readable but terrible for AI
reading) and have been removed from the tree to cut ~27MB of bloat -- they are
redundant with the compact .txt folders and the AM025 register JSON in mlir-aie.
They remain in git history if ever needed; convert_registers.py regenerates the
compact .txt from them.