#!/usr/bin/env python3
"""Mutation check of the register-port collision model.

Each mutation breaks one rule in the source (an exact-text replacement), runs the tests that must
catch it and restores the file. A mutation that no test notices is reported as SURVIVED and the
script exits non-zero. Run from the repository root; each case costs one incremental build.

  tools/mutation_check_collisions.py [name ...]
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WC = "src/interpreter/state/write_collision.rs"
CTX = "src/interpreter/state/context.rs"
REG = "src/interpreter/state/registers.rs"
SRS = "src/interpreter/execute/vector_srs.rs"
PORTS = "src/interpreter/decode/ports.rs"
GOLD = "interpreter::execute::collision_goldens"

# (name, file, old text, new text, test filter that must fail)
MUTATIONS = [
    ("lane-rule-ignores-bank", WC, "out[i].data = lane(c.bank).or_else(|| lane(0)).unwrap_or(c.data);",
     "out[i].data = lane(0).unwrap_or(c.data);", "write_collision"),
    ("earliest-writer-wins-bank", WC,
     "let latest_on_bank = on_port().filter(|(_, o)| o.bank == c.bank).map(|(_, o)| o.writer).max();",
     "let latest_on_bank = on_port().filter(|(_, o)| o.bank == c.bank).map(|(_, o)| o.writer).min();",
     "write_collision"),
    ("single-writer-takes-lane-rule", WC, "if writers_of(chunks, port).len() < 2 {",
     "if writers_of(chunks, port).is_empty() {", "write_collision"),
    ("cross-port-higher-index-wins", WC, "&& chunks[j].port < chunks[i].port", "&& chunks[j].port > chunks[i].port",
     "write_collision"),
    ("cross-port-any-register", WC, "                && chunks[j].reg == chunks[i].reg\n", "", "write_collision"),
    ("cross-port-any-bank", WC, "                && chunks[j].bank == chunks[i].bank\n", "", "write_collision"),
    ("mover-is-any-write-port", PORTS, "out.mover = at_def.iter().any(|u| self.is_last_port(*u));",
     "out.mover = !at_def.is_empty();", "decode::ports_goldens"),
    ("secondary-read-unit-is-lowest", PORTS, "secondary: (units.len() >= 2).then(|| *units.last().unwrap()),",
     "secondary: (units.len() >= 2).then(|| units[0]),", "decode::ports_goldens"),
    ("writes-not-tagged", CTX, "        self.write_seq += 1;\n        let tag = write_port",
     "        self.write_seq += 1;\n        let write_port: Option<u16> = None;\n        let tag = write_port", GOLD),
    ("acc-add-tag-lost", CTX, "            self.set_write_tag(pa.tag);", "            self.set_write_tag(None);", GOLD),
    ("ups-load-tag-lost", CTX, "            self.set_write_tag(pu.tag);\n            self.queue_acc_write_timed(",
     "            self.set_write_tag(None);\n            self.queue_acc_write_timed(", GOLD),
    ("matmul-tag-lost", "src/interpreter/execute/vector_matmul/mod.rs", "    ctx.set_write_tag(p.tag);",
     "    ctx.set_write_tag(None);", GOLD),
    ("vector-load-not-registered", CTX, "Operand::VectorReg(r) => self.vector.register_phantom(r, value),",
     "Operand::VectorReg(_) => None,", GOLD),
    ("steal-never-applies", CTX, "        let cycle = issue + port.offset as u64;\n",
     "        let cycle = issue + port.offset as u64;\n        if cycle > 0 {\n            return;\n        }\n", GOLD),
    ("steal-covers-all-banks", CTX, "                if c.banks & (1 << bank) == 0 {\n                    continue;\n                }\n",
     "", GOLD),
    ("wired-or-off", SRS, "ctx.accumulator.arm_wired_or(v.slot as u8, rv, p_high);",
     "ctx.accumulator.arm_wired_or(v.slot as u8, rv, [0; 8]);", GOLD),
    ("wired-or-patches-every-reader", REG, "if Some(*slot) == self.reader_slot && ", "if ", GOLD),
    ("s8-units-off", SRS, "if p.element_type == Some(ElementType::Int8) {", "if false && p.element_type == Some(ElementType::Int8) {",
     GOLD),
    ("lane-mode-not-widened", SRS, "if from.bits() < 64 && p.from_type.is_some_and(|t| t.bits() == 64) {",
     "if false && p.from_type.is_some_and(|t| t.bits() == 64) {", GOLD),
]


def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def main():
    wanted = set(sys.argv[1:])
    survived = []
    for name, path, old, new, flt in MUTATIONS:
        if wanted and name not in wanted:
            continue
        full = os.path.join(ROOT, path)
        src = open(full).read()
        if old not in src:
            print(f"{name}: PATTERN NOT FOUND in {path}")
            survived.append(name)
            continue
        open(full, "w").write(src.replace(old, new, 1))
        try:
            r = run(["cargo", "test", "--lib", flt])
        finally:
            open(full, "w").write(src)
        failed = r.returncode != 0 and "test result: FAILED" in r.stdout + r.stderr
        compile_err = "error[" in r.stderr or "could not compile" in r.stderr
        status = "KILLED" if failed and not compile_err else ("COMPILE-ERROR" if compile_err else "SURVIVED")
        print(f"{name}: {status}", flush=True)
        if status != "KILLED":
            survived.append(name)
    print("survivors:", survived or "none")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
