// SPDX-License-Identifier: MIT
//
// SeedFunctions.java -- Ghidra headless postScript that disassembles every
// Xtensa function the call-graph walk missed.
//
// Ghidra's auto-analysis is call-graph driven: it disassembles from known
// entry points and follows calls.  The NPU firmware reaches a large amount
// of code only indirectly -- interrupt/exception vectors, dispatch-table
// handlers, callx targets -- so those functions are never disassembled and
// their inbound calls never register (e.g. FUN_08ad8190, the suspend/halt
// routine, showed "0 callers" because its caller lived in an un-analyzed
// gap).
//
// Every windowed-ABI Xtensa function begins with an 'entry' instruction
// (byte 0x36, target register a1).  This script scans the whole image for
// that prologue, disassembles each candidate, and -- only if it genuinely
// decodes as 'entry' -- creates a function there.  A final analyzeChanges()
// pass lets the normal analyzers (constant propagation, l32r literal
// resolution, string detection) flow over the newly recovered code.
//
// Run as a postScript BEFORE DumpNpuFw.java so the dump sees the full set.
//@category NPU

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.mem.MemoryBlock;

public class SeedFunctions extends GhidraScript {

    @Override
    public void run() throws Exception {
        int seeded = 0, scanned = 0;

        for (MemoryBlock blk : currentProgram.getMemory().getBlocks()) {
            if (!blk.isInitialized()) {
                continue;
            }
            Address a = blk.getStart();
            Address end = blk.getEnd();
            while (a.compareTo(end) > 0 == false && a.add(2).compareTo(end) <= 0) {
                int b0 = getByte(a) & 0xFF;
                int b1 = getByte(a.add(1)) & 0xFF;
                // 'entry a1, imm': op0 byte 0x36, low nibble of byte 1 == a1.
                if (b0 == 0x36 && (b1 & 0x0F) == 0x01) {
                    scanned++;
                    if (getFunctionContaining(a) == null
                            && getInstructionContaining(a) == null) {
                        try {
                            disassemble(a);
                            Instruction insn = getInstructionAt(a);
                            if (insn != null
                                    && insn.getMnemonicString().equalsIgnoreCase("entry")) {
                                Function f = createFunction(a, null);
                                if (f != null) {
                                    seeded++;
                                }
                            }
                        } catch (Exception e) {
                            // Garbage that happened to match the prologue
                            // bytes; skip it.
                        }
                    }
                }
                a = a.add(1);
            }
        }

        println("SeedFunctions: " + scanned + " prologue candidates, "
                + seeded + " new functions created");

        if (seeded > 0) {
            println("SeedFunctions: running analyzeChanges() over recovered code");
            analyzeChanges(currentProgram);
        }
    }
}
