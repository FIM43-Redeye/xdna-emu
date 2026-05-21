// SPDX-License-Identifier: MIT
//
// DumpNpuFw.java -- Ghidra headless postScript for the Phoenix NPU firmware.
//
// Runs after auto-analysis and dumps the program to plain-text files so the
// analysis can be driven and reviewed from the CLI (Ghidra's GUI is not
// scriptable for our workflow).  Output, into the directory given as the
// first script argument (default: ./analysis):
//
//   functions.tsv -- addr, name, size, caller count, callee count
//   strings.tsv   -- addr, length, xref count, referencing function addrs,
//                    string value
//   disasm.txt    -- per-function disassembly, each instruction annotated
//                    with raw bytes and any data/string it references
//
// Grep these (e.g. for "smu", "power", "XAie", mailbox opcodes) to locate
// the SMU dispatcher / mailbox handler / shutdown state machine.
//@category NPU

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Data;
import ghidra.program.model.listing.DataIterator;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.listing.Listing;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceManager;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class DumpNpuFw extends GhidraScript {

    @Override
    public void run() throws Exception {
        String outDir = "analysis";
        String[] args = getScriptArgs();
        if (args.length > 0) {
            outDir = args[0];
        }
        new File(outDir).mkdirs();

        Listing listing = currentProgram.getListing();
        FunctionManager fm = currentProgram.getFunctionManager();
        ReferenceManager rm = currentProgram.getReferenceManager();

        // --- functions.tsv ----------------------------------------------
        int fcount = 0;
        try (PrintWriter w = new PrintWriter(new FileWriter(outDir + "/functions.tsv"))) {
            w.println("addr\tname\tsize_bytes\tnum_callers\tnum_callees");
            for (Function f : fm.getFunctions(true)) {
                long size = f.getBody().getNumAddresses();
                int callers = f.getCallingFunctions(monitor).size();
                int callees = f.getCalledFunctions(monitor).size();
                w.printf("%s\t%s\t%d\t%d\t%d%n",
                        f.getEntryPoint(), f.getName(), size, callers, callees);
                fcount++;
            }
        }

        // --- strings.tsv ------------------------------------------------
        int scount = 0;
        try (PrintWriter w = new PrintWriter(new FileWriter(outDir + "/strings.tsv"))) {
            w.println("addr\tlen\txref_count\txref_funcs\tstring");
            DataIterator di = listing.getDefinedData(true);
            while (di.hasNext()) {
                Data d = di.next();
                if (!d.hasStringValue()) {
                    continue;
                }
                Address sa = d.getAddress();
                StringBuilder xf = new StringBuilder();
                int xc = 0;
                for (Reference r : rm.getReferencesTo(sa)) {
                    xc++;
                    Function cf = fm.getFunctionContaining(r.getFromAddress());
                    if (cf != null) {
                        if (xf.length() > 0) {
                            xf.append(",");
                        }
                        xf.append(cf.getEntryPoint());
                    }
                }
                String val = d.getDefaultValueRepresentation();
                if (val != null) {
                    val = val.replace("\t", "\\t").replace("\n", "\\n");
                }
                w.printf("%s\t%d\t%d\t%s\t%s%n", sa, d.getLength(), xc, xf, val);
                scount++;
            }
        }

        // --- disasm.txt -------------------------------------------------
        try (PrintWriter w = new PrintWriter(new FileWriter(outDir + "/disasm.txt"))) {
            for (Function f : fm.getFunctions(true)) {
                w.println("==== " + f.getName() + " @ " + f.getEntryPoint()
                        + "  (size " + f.getBody().getNumAddresses() + " bytes, "
                        + f.getCallingFunctions(monitor).size() + " callers) ====");
                InstructionIterator it = listing.getInstructions(f.getBody(), true);
                while (it.hasNext()) {
                    Instruction insn = it.next();
                    StringBuilder bytes = new StringBuilder();
                    for (byte b : insn.getBytes()) {
                        bytes.append(String.format("%02x", b & 0xFF));
                    }
                    StringBuilder ann = new StringBuilder();
                    for (Reference r : insn.getReferencesFrom()) {
                        Data rd = listing.getDefinedDataAt(r.getToAddress());
                        if (rd != null) {
                            String rv = rd.getDefaultValueRepresentation();
                            if (rv != null) {
                                ann.append("  ; -> ").append(rv.replace("\n", "\\n"));
                            }
                        }
                    }
                    w.printf("  %-10s %-12s %s%s%n",
                            insn.getAddress(), bytes, insn.toString(), ann);
                }
                w.println();
            }
        }

        println("DumpNpuFw: " + fcount + " functions, " + scount
                + " strings -> " + outDir);
    }
}
