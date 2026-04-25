# Vitis Unified IDE

The AMD Unified Installer ships an **Eclipse Theia 1.62.2 IDE** ("Vitis
Unified IDE", project codename "Rigel") with both an Electron desktop
and a browser flavour.  The IDE talks over gRPC to a Java backend
(`vitis-server`).  Both components live inside `amd-unified-software/`
and run on Linux out of the box once two environment knobs are set.

This doc captures what we found while making it work, so the next
person doesn't have to chase the same thread.

## Components

```
amd-unified-software/
  ide/
    electron-app/lnx64/vitis-ide          # Theia/Electron binary
    browser-app/lnx64/                    # Same Theia frontend, Node-served
  vitis-server/
    bin/vitis-server                      # Gradle-generated Java launcher
    lib/*.jar                             # Java + gRPC + Netty + custom Xilinx jars
    scripts/                              # Tcl helpers, cmake toolchains, AIE templates
  lib/lnx64.o/
    libxv_common.so                       # JNI native libs the server loads
    libLLVM-9*.so, libboost_*.so.1.72.0   # OLD versions that shadow modern ones
```

* The Electron app's main process is in
  `ide/electron-app/lnx64/resources/app/scripts/rigel-electron-main.js`.
* The Theia backend (Node.js) is at `resources/app/lib/backend/main.js`
  (minified bundle).  It contains a function `findRigelServerExecutable`
  that resolves the server path from `$XILINX_VITIS/bin/vitis-server`.
* The Java server's main class is `com.xilinx.rigel.app.RigelApp`.  CLI
  flags exposed by `vitis-server --help`:
  `-p PORT`, `-t TIMEOUT_SECONDS`, `-w WORKSPACE_DIR`, `-V`, `-h`.

## How it starts up

When you launch `vitis-ide`:

1. The Electron main process logs to
   `~/.Xilinx/Vitis/2025.2/.vitis/<App Name>/logs/config.log`.
2. The Theia backend boots, reads contributions, then invokes
   `findRigelServerExecutable()` and `spawnServer()`.
3. The server JVM loads `libxv_common.so` from `java.library.path`
   (i.e. `LD_LIBRARY_PATH`).  Without it: the server dies with
   `UnsatisfiedLinkError: no xv_common in java.library.path`.
4. The server picks an ephemeral port (or honours `-p`) and prints
   `{"serverPort":NNNNN}`; the IDE backend reads the line, opens a
   gRPC channel, and exposes services to the frontend:
   `Rigel Utils`, `Debugger`, `Workspace`, `Repository`, `Build`,
   `AIE`, `Logger`, `Hls Flow`, `Platform`, `Program Flash`,
   `Elf Reader`.
5. Frontend connects, fires `listAppRepos` / `listAieExampleRepos`
   etc., and the IDE is live.

Verbose log to watch:
`~/.Xilinx/Vitis/2025.2/.wsdata/ide_verbose.log`.

## What you need to set

Two pieces -- both missing from a stock launch:

| Env var          | Value                                                      | Why                                                 |
|------------------|------------------------------------------------------------|-----------------------------------------------------|
| `XILINX_VITIS`   | `<NPU_WORK_DIR>/amd-unified-software/vitis-server`         | IDE backend resolves `$XILINX_VITIS/bin/vitis-server` |
| `LD_LIBRARY_PATH`| prepend `<NPU_WORK_DIR>/amd-unified-software/lib/lnx64.o`  | JVM `loadLibrary("xv_common")` and friends          |

### Env-var conflict with aietools

`activate-npu-env.sh` already sets `XILINX_VITIS=$AIETOOLS_DIR` because
**aietools' Python loader uses that path to find its bundled
`tps/lnx64/python-3.13.0/`**.  The two pointings are incompatible --
aietools needs `XILINX_VITIS=...amd-unified-software/aietools`, the IDE
needs `XILINX_VITIS=...amd-unified-software/vitis-server`.

We resolve this with a wrapper: `toolchain-build/launch-vitis-ide.sh`
overrides both env vars **only for the IDE subprocess**, leaving the
parent shell's aietools-friendly env intact.

### Why we don't add the native lib dir to global `LD_LIBRARY_PATH`

`amd-unified-software/lib/lnx64.o/` ships:

* `libLLVM-9.so`, `libLLVM-9.0.1.so`, `libLLVM.so` -- LLVM 9 from the
  Xilinx-shipped Vitis runtime; would shadow LLVM 21 that we use via
  llvm-aie / Peano.
* `libboost_*.so.1.72.0` -- Boost 1.72; would shadow whatever the
  system has (typically 1.83+).
* `libJudy.so.1` -- collides with the Ubuntu package's library.

Putting this directory on a user's global `LD_LIBRARY_PATH` would break
the rest of the toolchain.  The wrapper adds it only inside the IDE
process tree, where the JVM consumes it on the way in.

## How to launch

After `source toolchain-build/activate-npu-env.sh`, a `vitis-ide`
function is defined that calls the wrapper:

```bash
vitis-ide                 # opens the IDE
vitis-ide /path/to/ws     # opens the IDE with workspace argument
```

Or call the wrapper directly without sourcing the activator:

```bash
toolchain-build/launch-vitis-ide.sh
```

The wrapper validates that the IDE binary, `vitis-server` launcher and
`libxv_common.so` all exist before invoking Electron, and prints the
two env vars it sets so you can see what scope it touched.

## Known harmless warnings on first launch

These appear in `ide_verbose.log` on a clean run; they don't block the
IDE from working:

* `Error in handling the notification ... org.eclipse.tcf.protocol.Protocol.event_queue is null`
  -- Eclipse TCF debug protocol's event queue isn't initialized until a
  debug session starts.
* `Invalid ESW Repo Directory path 'null'` -- embedded software
  examples repo isn't configured (uncommon on a workstation install).
* `Reading early access data from ...vitis-server/vitis-server/scripts/early-access/...`
  -- the path has `vitis-server` doubled because the Feature Registry
  appends `vitis-server/` to `$XILINX_VITIS` (which already points at
  `vitis-server/`).  Cosmetic; the file is still found via the doubled
  segment because it doesn't exist there, and the loader falls back.

## Why this matters

The Vitis IDE bundles a lot of useful surface for AIE/NPU work that
isn't easily reachable elsewhere: a real debug stack via gRPC, the
HLS Flow Server, an AIE Server with build/repository plumbing, a
debugger and an ELF reader, plus the live Theia frontend that can be
extended with VSCode-flavoured plugins.  An Electron client we control
can talk to the same gRPC backend if we want our own UI surface.

## Quick troubleshooting

| Symptom                                                            | Likely cause                          |
|--------------------------------------------------------------------|---------------------------------------|
| IDE opens but `Could not initialize contribution` re XILINX_VITIS  | Forgot to use the wrapper / activate  |
| `UnsatisfiedLinkError: no xv_common in java.library.path`          | `LD_LIBRARY_PATH` missing native libs |
| `Invalid workspace`                                                | Pass an existing dir to `-w`          |
| IDE hangs at `Initializing the frontend client`                    | Server died -- check verbose log      |
