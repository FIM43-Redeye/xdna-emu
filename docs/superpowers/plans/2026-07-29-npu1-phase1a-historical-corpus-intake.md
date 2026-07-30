# NPU1 Phase 1A Historical Corpus Intake Implementation Plan

**Goal:** Freeze the current live NPU1 tuple, account for every family in the
two known historical-corpus roots, and deeply intake one successful legacy
firmware witness without modifying the corpus or promoting old claims.

**Architecture:** Two sequential documentation slices use ordinary read-only
system tools. Slice A creates the current-tuple and family-level census. After
review, Slice B verifies and audits one immutable Chess command-list capture.
There is no scanner, schema, database, sidecar, dependency, hardware rerun, or
emulator change.

**Execution:** The primary agent executes serially in the existing
`firmware-priors` worktree. No subagents, Halo, NPU workload, KVM launch, or
privileged operation. Use `apply_patch` for both findings. Stop for Maya's
review after each committed slice.

**Approved design:**
[Phase 1A historical corpus intake
design](../specs/2026-07-29-npu1-phase1a-historical-corpus-intake-design.md)

## Baseline

The approved design base is:

```text
branch: investigate/firmware-priors
commit: 90435ad39777a02cdcdb53275318ba6abff94321
```

Implementation begins from a clean `investigate/firmware-priors` HEAD that
contains this plan and has the design base as an ancestor.

At planning time:

- both planned findings are absent;
- `cargo test --lib` passes 4,275 tests and ignores 32;
- `repo-experiments` contains firmware, vfio-user, timing, and transaction
  campaign families;
- `workspace-experiments` contains NPU and unrelated BIOS/DKMS families that
  still require explicit dispositions;
- the selected Chess command-list capture occupies roughly 187 MiB and records
  `PHOENIX_FROZEN_PASS chess`;
- a separate direct-execution capture records `PHOENIX_FROZEN_PASS peano`, so
  Peano is not an unrun gate; and
- the successful guest runs retain dma-buf and recursive-locking warnings.

The implementation rechecks every baseline fact rather than copying it from
this plan.

## Shared Read-Only Setup

Run this setup at the start of each slice:

```bash
xdna_checkout=$(git rev-parse --show-toplevel)
git_common_dir=$(git rev-parse --path-format=absolute --git-common-dir)
xdna_common_root=$(dirname "$git_common_dir")
npu_workspace=$(dirname "$xdna_common_root")

repo_corpus="$xdna_checkout/build/experiments"
workspace_corpus="$npu_workspace/experiments"

test -d "$repo_corpus"
test -r "$repo_corpus"
test -d "$workspace_corpus"
test -r "$workspace_corpus"
```

Checked-in reports use only `repo-experiments` and
`workspace-experiments`. The expanded local variables are never copied into
them.

Shell variables and functions are operational conveniences, not persistent
state. Re-run the shared setup in every fresh tool process that consumes them.

For pre/post mutation detection, compute this metadata fingerprint for each
root and record the result in the active report:

```bash
metadata_fingerprint() {
    local corpus_root=$1
    (
        cd "$corpus_root"
        find . -xdev \
            -printf '%P\t%y\t%m\t%s\t%T@\t%D:%i:%n\t%l\0' |
            LC_ALL=C sort -z |
            sha256sum
    )
}

metadata_fingerprint "$repo_corpus"
metadata_fingerprint "$workspace_corpus"
```

The fingerprint deliberately excludes access time. It detects changes during
the work; it is not a content-integrity or replica claim.

---

## Slice A -- Live Tuple and Shallow Census

### Task 1: Preflight and Census Skeleton

**Files:**

- Add:
  `docs/superpowers/findings/2026-07-29-npu1-historical-corpus-census.md`

- [ ] Require the approved base and clean tree:

```bash
test "$(git branch --show-current)" = "investigate/firmware-priors"
git merge-base --is-ancestor \
    90435ad39777a02cdcdb53275318ba6abff94321 \
    HEAD
test -f \
    docs/superpowers/plans/2026-07-29-npu1-phase1a-historical-corpus-intake.md
test -z "$(git status --porcelain)"
```

- [ ] Run the shared setup and capture the initial metadata fingerprint for
  both roots.
- [ ] Independently enumerate the top-level boundary:

```bash
find "$repo_corpus" -xdev -mindepth 1 -maxdepth 1 \
    -printf 'repo-experiments/%f\t%y\n' |
    LC_ALL=C sort

find "$workspace_corpus" -xdev -mindepth 1 -maxdepth 1 \
    -printf 'workspace-experiments/%f\t%y\n' |
    LC_ALL=C sort
```

- [ ] Use `apply_patch` to create the report skeleton before filling it. It
  must contain:
  - scope and scan time;
  - root-alias and measurement semantics;
  - live tuple table with source and confidence columns;
  - top-level family table;
  - preservation hazards;
  - unknowns and contradictions;
  - exemplar selection; and
  - validation evidence.
- [ ] Give every family-table row the machine-checkable form:

```text
| F | `repo-experiments/<name>` | <disposition> | ... |
| F | `workspace-experiments/<name>` | <disposition> | ... |
```

- [ ] Leave absent facts explicitly `Pending`; do not prefill from old
  `tuple.txt` files.

### Task 2: Freeze the Current Live Tuple

**Files:**

- Modify:
  `docs/superpowers/findings/2026-07-29-npu1-historical-corpus-census.md`

- [ ] Record the current xdna-emu source identity from the clean preflight:

```bash
git branch --show-current
git rev-parse HEAD
git status --porcelain
git worktree list --porcelain
```

- [ ] Identify the live NPU1 device without opening it:

```bash
lspci -Dnn -d 1022:1502
lspci -Dnnvv -d 1022:1502
```

- [ ] Require exactly one matching function before deriving its BDF. If zero
  or multiple devices appear, record the ambiguity and stop the physical tuple
  derivation rather than choosing one silently.
- [ ] After confirming the single-device count, derive the BDF:

```bash
npu_bdf_count=$(lspci -Dnn -d 1022:1502 | awk 'END { print NR }')
test "$npu_bdf_count" -eq 1
npu_bdf=$(lspci -Dnn -d 1022:1502 | awk 'NR == 1 { print $1 }')
test -n "$npu_bdf"
```

- [ ] For the single BDF, record current sysfs ownership and IOMMU exposure:

```bash
readlink -f "/sys/bus/pci/devices/$npu_bdf/driver"
readlink -f "/sys/bus/pci/devices/$npu_bdf/iommu_group"
```

- [ ] Record the running kernel and amdxdna module identity:

```bash
uname -r
modinfo amdxdna
amdxdna_module=$(modinfo -F filename amdxdna)
test -f "$amdxdna_module"
sha256sum "$amdxdna_module"
```

- [ ] If amdxdna is loaded, enumerate and read its exposed parameters without
  writing them:

```bash
if test -d /sys/module/amdxdna/parameters; then
    find /sys/module/amdxdna/parameters -maxdepth 1 -type f \
        -printf '%f\n' |
        LC_ALL=C sort
fi
```

Read each listed value individually and cite its sysfs name. Missing parameters
remain `unknown`.

- [ ] Hash the primary firmware and AM025 register database:

```bash
sha256sum /usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin
sha256sum \
    "$npu_workspace/mlir-aie/lib/Dialect/AIE/Util/aie_registers_aie2.json"
```

- [ ] Record package versions and the resolved XRT runtime-library identities:

```bash
dpkg-query -W \
    -f='${binary:Package}\t${Version}\n' \
    xrt-base xrt-npu xrt_plugin-amdxdna

for xrt_name in \
    libxrt_core.so \
    libxrt_coreutil.so \
    libxrt_driver_xdna.so
do
    xrt_library=$(readlink -f "/opt/xilinx/xrt/lib/$xrt_name")
    test -f "$xrt_library"
    printf '%s\t%s\n' "$xrt_name" "$xrt_library"
    sha256sum "$xrt_library"
done
```

The report retains the library basename, version, and hash, not the expanded
home-directory path.

- [ ] Record source revision and dirty state independently for each available
  component:

```bash
for component_name in aie-rt mlir-aie llvm-aie xdna-driver
do
    component_root="$npu_workspace/$component_name"
    printf '%s\n' "$component_name"
    git -C "$component_root" rev-parse HEAD
    git -C "$component_root" branch --show-current
    git -C "$component_root" status --porcelain
done
```

- [ ] Record relevant live address/IOMMU state when safely exposed. Record
  unexposed reset, power, or clock state as `unknown`; do not infer it and do
  not issue a device command.
- [ ] For every tuple row, identify the live command or file that supplied the
  value. Keep loaded module state distinct from nearby source-checkout state.

### Task 3: Census, Classify, Validate, and Commit Slice A

**Files:**

- Modify:
  `docs/superpowers/findings/2026-07-29-npu1-historical-corpus-census.md`

- [ ] For each top-level family in both roots, run this metadata-only summary
  twice:

```bash
family_summary() {
    local family_path=$1

    printf 'types\n'
    find "$family_path" -xdev -printf '%y\n' |
        LC_ALL=C sort |
        uniq -c

    printf 'allocated-bytes\n'
    du -sx --block-size=1 "$family_path"

    printf 'apparent-bytes\n'
    du -sx --apparent-size --block-size=1 "$family_path"

    printf 'time-bounds\n'
    find "$family_path" -xdev -printf '%T@\t%P\n' |
        LC_ALL=C sort -n |
        awk 'NR == 1 { first = $0 } { last = $0 }
             END { print first; if (last != first) print last }'

    printf 'unreadable\n'
    find "$family_path" -xdev ! -readable -printf '%P\n' |
        LC_ALL=C sort

    printf 'broken-links\n'
    find "$family_path" -xdev -xtype l -printf '%P\t%l\n' |
        LC_ALL=C sort

    printf 'multiply-linked-files\n'
    find "$family_path" -xdev -type f -links +1 \
        -printf '%D:%i\t%n\t%P\n' |
        LC_ALL=C sort

    printf 'provenance-markers\n'
    find "$family_path" -xdev -type f \
        \( -name tuple.txt -o -name manifest.json \
           -o -name SHA256SUMS -o -iname '*sha256*' \) \
        -printf '%P\n' |
        LC_ALL=C sort
}
```

- [ ] If the two summaries disagree, mark the family `unstable` and rescan.
  Do not conceal the change by using only the later values.
- [ ] Read only small provenance markers and representative filenames needed
  to distinguish campaign kinds. Do not recursively invoke `file`, hash bulk
  contents, or interpret raw traces in Slice A.
- [ ] Assign exactly one disposition to each top-level family:
  `npu1-relevant`, `mixed`, `excluded-with-reason`, or `unknown`.
- [ ] Record rather than resolve:
  - the linked-worktree location of `repo-experiments`;
  - apparent-versus-allocated size discrepancies;
  - unreadable or broken entries;
  - non-independent link evidence;
  - missing provenance markers; and
  - any family whose relevance cannot be established cheaply.
- [ ] Nominate the approved Chess command-list exemplar and note the distinct
  successful Peano/direct capture.
- [ ] Mechanically prove exact top-level accounting:

```bash
actual_families() {
    find "$repo_corpus" -xdev -mindepth 1 -maxdepth 1 \
        -printf 'repo-experiments/%f\n'
    find "$workspace_corpus" -xdev -mindepth 1 -maxdepth 1 \
        -printf 'workspace-experiments/%f\n'
}

reported_families() {
    awk -F'|' '
        $2 ~ /^[[:space:]]*F[[:space:]]*$/ {
            value = $3
            gsub(/[`[:space:]]/, "", value)
            print value
        }
    ' docs/superpowers/findings/2026-07-29-npu1-historical-corpus-census.md
}

comm -3 \
    <(actual_families | LC_ALL=C sort) \
    <(reported_families | LC_ALL=C sort)

reported_families | LC_ALL=C sort | uniq -d
```

Both commands must print nothing.

- [ ] Re-run the shared metadata fingerprints and require exact agreement with
  the values recorded at preflight. A disagreement blocks the commit until it
  is explained.
- [ ] Reject leaked home-directory paths:

```bash
if rg -n '/home/triple/' \
    docs/superpowers/findings/2026-07-29-npu1-historical-corpus-census.md
then
    exit 1
fi
```

- [ ] Run the repository gates:

```bash
git add --intent-to-add \
    docs/superpowers/findings/2026-07-29-npu1-historical-corpus-census.md
git diff --check
cargo test --lib
```

- [ ] Review every `unknown`, exclusion reason, and preservation hazard.
- [ ] Confirm `git diff --name-only` lists only the census report.
- [ ] Commit:

```text
docs(reserve): census NPU1 historical evidence
```

### Slice A Review Checkpoint

Stop after the census commit. Give Maya the report, exact validation counts,
and any preservation hazard that may require action before deeper intake.
Do not begin Slice B until she approves the census.

---

## Slice B -- Deep Intake of the Chess Command-List Witness

### Task 4: Integrity Audit and Intake Skeleton

**Files:**

- Add:
  `docs/superpowers/findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md`

- [ ] Require a clean tree at the approved Slice A commit.
- [ ] Run the shared setup and define the immutable exemplar:

```bash
witness_dir="$repo_corpus/phoenix-vfio-user/20260729T171244Z-3136359"
test -d "$witness_dir"
test -r "$witness_dir"
```

- [ ] Capture its pre-intake metadata fingerprint, allocated/apparent size,
  regular-file count, and symlink inventory:

```bash
metadata_fingerprint "$witness_dir"
du -sh "$witness_dir"
du -sh --apparent-size "$witness_dir"
find "$witness_dir" -xdev -type f -printf . | wc -c
find "$witness_dir" -xdev -type l -printf '%P\t%l\n' |
    LC_ALL=C sort
```

- [ ] Attempt every checksum already recorded in `tuple.txt`:

```bash
rg '^[0-9a-f]{64}  ' "$witness_dir/tuple.txt" |
    sha256sum --check --strict -
```

A missing or mismatched external reference is report evidence, not permission
to substitute or repair it.

- [ ] Generate the deterministic root-relative checksum listing without
  writing into the capture:

```bash
(
    cd "$witness_dir"
    find . -xdev -type f -print0 |
        LC_ALL=C sort -z |
        xargs -0 -r sha256sum
)
```

- [ ] Use `apply_patch` to create the intake skeleton with:
  - scope and immutable source alias;
  - recovered platform and stimulus;
  - artifact integrity;
  - observed outcomes;
  - lifecycle and warning audit;
  - candidate facts and explicit non-claims;
  - redistributability;
  - missing canonical fields and rerun requirements; and
  - a root-relative checksum appendix.
- [ ] Embed the complete checksum listing between explicit
  `CHECKSUMS-BEGIN` and `CHECKSUMS-END` comments. Do not add a sidecar to the
  witness.

### Task 5: Provenance and Claim Audit

**Files:**

- Modify:
  `docs/superpowers/findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md`

- [ ] Read the evidence path from the assertion backward:
  `guest.log`, `dmesg.log`, `server.log`, `msix.log`, `tuple.txt`,
  `qemu-command.txt`, build logs, library inventory, frozen workload, driver
  source, and initramfs contents.
- [ ] Mechanically verify the guest result sequence and pass marker:

```bash
awk '
    BEGIN { expected = 2; pass_count = 0 }
    /^Correct output [0-9]+ == [0-9]+$/ {
        if ($3 != expected || $5 != expected) {
            printf "unexpected output at %d: %s\n", expected, $0 > "/dev/stderr"
            exit 1
        }
        expected++
    }
    /^PHOENIX_FROZEN_PASS chess$/ { pass_count++ }
    END {
        if (expected != 66 || pass_count != 1)
            exit 1
        print "ordered outputs 2..65 and one Chess pass marker"
    }
' "$witness_dir/guest.log"
```

- [ ] Locate and cite exact relative lines for:
  - compiler and command-list stimulus;
  - firmware and driver identities;
  - firmware command completion;
  - context interrupt publication;
  - driver teardown;
  - dma-buf warnings;
  - recursive-locking warnings; and
  - any recovery or unexplained anomaly.
- [ ] Label each material report statement `Observed`, `Derived`, or
  `Unknown`. A successful marker remains an observation about that run, not a
  generalized hardware fact.
- [ ] Keep these questions separate:
  - Did the guest produce the correct result?
  - Did firmware publish completion?
  - Did the interrupt and driver lifecycle finish?
  - Was teardown warning-free?
- [ ] Distinguish contained artifacts from absolute external references.
  Verify what still exists, but never replace a missing historical object with
  a current namesake.
- [ ] Record redistribution only when supported by a license or source
  classification. Proprietary firmware and runtime payloads default to
  non-redistributable; uncertain material remains `unknown`.
- [ ] List candidate emulator contracts without promoting them. Include every
  missing field needed for a canonical rerun.
- [ ] Record the Peano/direct success only as a companion candidate. Do not
  merge its distinct stimulus into this witness.

### Task 6: Validate and Commit Slice B

**Files:**

- Modify:
  `docs/superpowers/findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md`

- [ ] Reproduce the embedded checksum appendix exactly:

```bash
generated_checksums() {
    (
        cd "$witness_dir"
        find . -xdev -type f -print0 |
            LC_ALL=C sort -z |
            xargs -0 -r sha256sum
    )
}

reported_checksums() {
    sed -n \
        '/CHECKSUMS-BEGIN/,/CHECKSUMS-END/p' \
        docs/superpowers/findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md |
        sed '1d;$d;/^```/d'
}

diff -u \
    <(generated_checksums) \
    <(reported_checksums)
```

- [ ] Re-run the guest-result verifier from Task 5.
- [ ] Re-run the witness metadata fingerprint and require exact agreement with
  the pre-intake value.
- [ ] Re-run both corpus-root fingerprints and compare them with Slice A.
  Any drift is reported before continuing; Phase 1A does not conceal it.
- [ ] Reject leaked home-directory paths:

```bash
if rg -n '/home/triple/' \
    docs/superpowers/findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md
then
    exit 1
fi
```

- [ ] Confirm every candidate fact has a cited observation and caveat, and
  every missing field is `Unknown`.
- [ ] Run:

```bash
git add --intent-to-add \
    docs/superpowers/findings/2026-07-29-phoenix-vfio-user-cmdlist-intake.md
git diff --check
cargo test --lib
```

- [ ] Confirm `git diff --name-only` lists only the intake report.
- [ ] Commit:

```text
docs(reserve): intake Phoenix command-list witness
```

- [ ] Require a clean worktree and report the two Slice A/B commit IDs and
  exact validation counts.

### Slice B Review Checkpoint

Stop after the intake commit. Maya reviews the recovered tuple, functional
claim, warnings, unknowns, and canonical-rerun requirements. Phase 2 catalogue
design begins only after that review.

## Explicit Non-Actions

- Do not modify, rename, move, delete, deduplicate, or annotate either corpus.
- Do not create a sidecar, manifest, checksum file, or bundle in a legacy
  capture.
- Do not write an inventory scanner or retain an ad hoc script.
- Do not add a schema, database, service, dependency, or storage system.
- Do not run NPU hardware, KVM, vfio-user, QEMU, Halo, or privileged commands.
- Do not touch emulator, firmware, driver, or toolchain source.
- Do not call a historical `PASS` retirement-qualified.
- Do not begin Phase 2 or a canonical rerun within this plan.
