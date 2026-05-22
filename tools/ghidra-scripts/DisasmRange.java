// SPDX-License-Identifier: MIT
//
// DisasmRange.java -- Ghidra headless postScript that force-disassembles an
// arbitrary address range and dumps the instructions.
//
// The NPU firmware's exception/interrupt vectors and reset stub live below
// the first call-graph-reachable function, in a region the normal analysis
// and SeedFunctions.java (which keys off the windowed-ABI 'entry' prologue)
// never touch -- vector handlers have no 'entry'.  To read the RTOS syscall
// dispatcher (the 'syscall' instruction traps here) the raw region must be
// disassembled directly.
//
// Args: <start-hex> <end-hex> <output-file>
//   addresses are image-absolute (e.g. 08ad3000), no 0x prefix needed.
//
// Run with:  analyzeHeadless <proj-loc> <proj-name> -process npu-fw-body.bin \
//              -noanalysis -scriptPath <dir> \
//              -postScript DisasmRange.java <start> <end> <out>
//@category NPU

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Instruction;

import java.io.PrintWriter;

public class DisasmRange extends GhidraScript {

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 3) {
            println("DisasmRange: need <start-hex> <end-hex> <output-file>");
            return;
        }
        Address start = currentProgram.getAddressFactory()
                .getDefaultAddressSpace().getAddress(Long.parseLong(args[0], 16));
        Address end = currentProgram.getAddressFactory()
                .getDefaultAddressSpace().getAddress(Long.parseLong(args[1], 16));
        String outPath = args[2];

        // First pass: linear-sweep disassembly across the range.  Xtensa
        // instructions are 2 or 3 bytes; a misaligned start inside a literal
        // pool decodes as garbage but self-resynchronises quickly.
        int created = 0;
        Address a = start;
        while (a.compareTo(end) <= 0) {
            if (getInstructionContaining(a) == null) {
                try {
                    disassemble(a);
                    if (getInstructionAt(a) != null) {
                        created++;
                    }
                } catch (Exception e) {
                    // undecodable byte; step over it
                }
            }
            Instruction insn = getInstructionAt(a);
            a = (insn != null) ? a.add(insn.getLength()) : a.add(1);
        }

        // Second pass: dump every instruction in the range.
        int dumped = 0;
        try (PrintWriter w = new PrintWriter(outPath)) {
            w.printf("# DisasmRange %s .. %s%n", start, end);
            Instruction insn = getInstructionAt(start);
            if (insn == null) {
                insn = getInstructionAfter(start);
            }
            while (insn != null && insn.getAddress().compareTo(end) <= 0) {
                Address ia = insn.getAddress();
                StringBuilder bytes = new StringBuilder();
                for (byte b : insn.getBytes()) {
                    bytes.append(String.format("%02x", b & 0xFF));
                }
                w.printf("  %s   %-10s   %s%n", ia, bytes, insn.toString());
                dumped++;
                insn = getInstructionAfter(ia);
            }
        }

        println("DisasmRange: disassembled " + created + " new instructions, "
                + "dumped " + dumped + " to " + outPath);
    }
}
