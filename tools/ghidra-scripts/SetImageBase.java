// SPDX-License-Identifier: MIT
//
// SetImageBase.java -- Ghidra headless preScript that relocates the program
// image base before auto-analysis runs.
//
// The NPU firmware is a raw blob with no headers, so Ghidra imports it at
// address 0.  But the code addresses absolute targets (string literals,
// function pointers) through Xtensa l32r literal pools, encoded as
// (load_base + offset).  Recovered load base: 0x08ad3000 (see
// tools/fw-find-base.py).  Setting the image base here -- before analysis
// -- lets the analyzer resolve every l32r literal to its real target, so
// string xrefs and pointer tables light up.
//
// Usage: -preScript SetImageBase.java <hex-base>
//@category NPU

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;

public class SetImageBase extends GhidraScript {

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 1) {
            println("SetImageBase: no base argument given; leaving image base unchanged.");
            return;
        }
        long base = Long.parseLong(args[0].replace("0x", ""), 16);
        Address newBase = currentProgram.getAddressFactory()
                .getDefaultAddressSpace().getAddress(base);
        currentProgram.setImageBase(newBase, true);
        println("SetImageBase: image base set to " + newBase);
    }
}
