# Memory and Documentation Cleanup -- Design

**Date**: 2026-04-16
**Status**: Approved, executing

## Problem

Memory and documentation have accumulated into an unwieldy state:

- 88 memory files + 362-line `MEMORY.md` index (most content describes already-shipped fixes)
- 57 plans in `docs/plans/` (Feb-April), 22 in `docs/superpowers/plans/`, 18 specs
- Loose historical audits at `docs/*.md` top level
- Durable user preferences trapped in memory (`feedback_*.md`) that should live in the repo

The repo should be self-documenting. Claude memory should hold only what
genuinely cannot live in source: machine-specific env state, user
collaboration style, truly in-flight context.

## Goals

1. **Clear active context** -- Claude loads a short, focused memory index
2. **Preserve history** -- nothing is deleted; archived content stays on disk
3. **Promote durable knowledge** -- user/project rules move into CLAUDE.md
4. **Set up for refactor** -- a clean slate makes subsequent refactor planning tractable

## Non-goals

- No code refactoring (that comes after, in its own pass)
- No rewriting of `docs/roadmap/` phase files or `docs/xdna/` reference material
- No deletion of any historical content (archive only)

## Design

### Three archive layers

**Layer 1 -- Memory** (`~/.claude/projects/-home-triple-npu-work-xdna-emu/memory/`)

- New `memory/archive/` subfolder
- Files in `archive/` are on disk but NOT referenced from `MEMORY.md`, so
  Claude never loads them in future sessions
- Archive rules:
  - All `*-2026MMDD.md` dated session logs
  - All entries explicitly marked superseded
  - All "SHIPPED" / "RESOLVED" / "FIXED" writeups (git log has the actual fix)
  - Topic references made redundant by CLAUDE.md or code comments
- Keep in memory root: only machine-specific env state and user preferences
  that don't belong in the repo

**Layer 2 -- Repo docs** (`xdna-emu/docs/`)

- New `docs/archive/plans/`, `docs/archive/specs/`, and `docs/archive/` subdirs
- Move all completed plans and specs to archive
- Triage loose `docs/*.md` top-level files; archive historical audits
- Preserve: `docs/roadmap/`, `docs/xdna/` (reference material),
  `docs/investigations/` (cold-case writeups worth keeping at the top level)

**Layer 3 -- CLAUDE.md promotion**

- Extract durable feedback rules from `feedback_*.md` memory files
- Anything not already covered by global or project CLAUDE.md moves into
  `xdna-emu/CLAUDE.md`
- Redundant feedback files are deleted (not archived) since they duplicate
  CLAUDE.md content

### Categorization rules (memory)

| Pattern | Action |
|---|---|
| `*-2026MMDD.md` dated session log | Archive |
| Marked "superseded" in MEMORY.md | Archive |
| "SHIPPED" / "RESOLVED" / "FIXED" writeup | Archive |
| Topic reference already in CLAUDE.md | Archive |
| `feedback_*.md` already in CLAUDE.md | Delete |
| `feedback_*.md` NOT yet in CLAUDE.md | Promote to CLAUDE.md, then delete |
| `todo_*.md` | Archive (TODOs should be issues, not memory) |
| Env/state (HOSTID, DNS, driver module) | Keep in memory |
| User collaboration preferences | Keep in memory |

### New MEMORY.md

Target: ~30 lines. Pure index of living memory, no historical pointers.
Archived content is discoverable via `ls memory/archive/` but not loaded.

## Deliverables

1. `memory/archive/` populated; `MEMORY.md` rewritten to active-only
2. `docs/archive/` populated; active plans/specs left at top level (or
   none, if all are archived)
3. `xdna-emu/CLAUDE.md` updated with any promoted rules
4. 3-4 logical commits:
   - Archive memory files + rewrite index
   - Promote feedback rules to CLAUDE.md and delete redundant files
   - Archive docs/plans and docs/specs
   - Triage top-level docs/ loose files

## Out of scope

- Refactor planning -- that's the next task, after cleanup lands
- Deleting archived content -- the archive is permanent; future cleanup
  passes can reduce it if size becomes a problem

## Success criteria

- `MEMORY.md` fits on one screen
- Every remaining file in `memory/` root has a clear "why this lives here
  and not in the repo" answer
- `ls docs/` and `ls docs/plans/` show focused, current content
- Git history preserves the move (use `git mv`, not delete+add)
