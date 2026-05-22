// SPDX-License-Identifier: MIT
//
// DecompileNpuFw.java -- Ghidra headless postScript that runs the
// decompiler over every function in the Phoenix NPU firmware and dumps
// the recovered C to a single grep-friendly file.
//
// DumpNpuFw.java emits raw Xtensa disassembly; that is exact but slow to
// read.  This script complements it: it runs Ghidra's DecompInterface on
// each function and writes pseudo-C.  The C is approximate (Xtensa
// windowed-ABI register windows and l32r literal pools confuse the
// decompiler more than a stock target would) but turns a multi-hour
// disasm grind into a readable skim.  Always cross-check a load-bearing
// claim against disasm.txt.
//
// Output, into the directory given as the first script argument
// (default: ./analysis):
//
//   decompiled.c  -- every function as pseudo-C, each preceded by a
//                    "// ==== name @ addr (N callers) ====" banner so the
//                    file greps the same way disasm.txt does.
//
// Run as a postScript AFTER SeedFunctions.java so the decompiler sees the
// full recovered function set.
//@category NPU

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

public class DecompileNpuFw extends GhidraScript {

    // Per-function decompiler timeout.  The firmware's largest functions
    // decompile in well under a second; 60s is a generous ceiling that
    // only trips on pathological cases.
    private static final int DECOMP_TIMEOUT_SECS = 60;

    @Override
    public void run() throws Exception {
        String outDir = "analysis";
        String[] args = getScriptArgs();
        if (args.length > 0) {
            outDir = args[0];
        }
        new File(outDir).mkdirs();

        FunctionManager fm = currentProgram.getFunctionManager();

        DecompInterface ifc = new DecompInterface();
        DecompileOptions opts = new DecompileOptions();
        ifc.setOptions(opts);
        if (!ifc.openProgram(currentProgram)) {
            println("DecompileNpuFw: FATAL -- could not open program for "
                    + "decompilation: " + ifc.getLastMessage());
            return;
        }

        int ok = 0, failed = 0;
        try (PrintWriter w = new PrintWriter(new FileWriter(outDir + "/decompiled.c"))) {
            w.println("// Decompiled Phoenix NPU firmware (Ghidra pseudo-C).");
            w.println("// Approximate -- cross-check load-bearing claims against disasm.txt.");
            w.println();
            for (Function f : fm.getFunctions(true)) {
                int callers = f.getCallingFunctions(monitor).size();
                w.println("// ==== " + f.getName() + " @ " + f.getEntryPoint()
                        + "  (size " + f.getBody().getNumAddresses() + " bytes, "
                        + callers + " callers) ====");

                DecompileResults res =
                        ifc.decompileFunction(f, DECOMP_TIMEOUT_SECS, monitor);
                if (res != null && res.decompileCompleted()
                        && res.getDecompiledFunction() != null) {
                    w.println(res.getDecompiledFunction().getC());
                    ok++;
                } else {
                    String msg = (res != null) ? res.getErrorMessage() : "null result";
                    w.println("// DECOMPILE FAILED: " + msg);
                    w.println();
                    failed++;
                }
            }
        } finally {
            ifc.dispose();
        }

        println("DecompileNpuFw: " + ok + " functions decompiled, "
                + failed + " failed -> " + outDir + "/decompiled.c");
    }
}
