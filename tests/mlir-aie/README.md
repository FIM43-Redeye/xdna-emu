# mlir-aie Test Integration

This directory contains tools for discovering, querying, and running mlir-aie tests against xdna-emu.

## Quick Start

```bash
# Discover tests from mlir-aie repository
./discover_tests.py --summary

# Query tests by architecture
./query_tests.py --arch aie2

# Run AIE2 tests through emulator
./run_tests.py --arch aie2 --limit 10
```

## Tools

### discover_tests.py

Scans the mlir-aie repository and generates a manifest of all tests with metadata.

```bash
# Generate manifest with summary
./discover_tests.py --mlir-aie /path/to/mlir-aie --summary

# Filter by architecture
./discover_tests.py --filter-arch aie2
```

Output: `manifest.json` containing test metadata including:
- Target architecture (aie1, aie2, aie2p, multi)
- Supported devices (npu1, npu2, xcvc1902, etc.)
- Category (integration, unit_test, example, tutorial, benchmark)
- Features used (dma, locks, objectfifo, memtile, etc.)
- Source type (mlir, python, both)

### query_tests.py

Filter and search the manifest for specific tests.

```bash
# List all AIE2 integration tests
./query_tests.py --arch aie2 --category integration

# Find tests using DMA feature
./query_tests.py --feature dma

# Search by name
./query_tests.py --name passthrough

# Output paths for scripting
./query_tests.py --arch aie2 --paths

# Show summary statistics
./query_tests.py --summary

# Verbose output with all details
./query_tests.py --arch aie2 --verbose
```

### run_tests.py

Run tests through the xdna-emu emulator.

```bash
# Dry run - show what would be executed
./run_tests.py --arch aie2 --dry-run

# Run AIE2 tests
./run_tests.py --arch aie2

# Run specific tests
./run_tests.py --name add_one

# Limit number of tests
./run_tests.py --arch aie2 --limit 5

# Save results to JSON
./run_tests.py --arch aie2 --json-output results.json
```

## Architecture Support

| Architecture | Driver IDs | Status in xdna-emu |
|--------------|------------|-------------------|
| aie1 | xcvc1902 (Versal) | Not supported |
| aie2 | npu1, xcve2802 | **Primary target** |
| aie2p | npu2 (maps to NPU4-6) | Planned |
| multi | Multiple targets | Depends on device |

**Note on naming**: mlir-aie uses `npu2` to refer to Strix (AIE2P), which is driver ID NPU4 in the xdna-driver. See CLAUDE.md for the full mapping.

## Test Categories

- **integration**: Full end-to-end tests from `test/npu-xrt/`
- **unit_test**: Focused tests from `test/unit_tests/`
- **example**: Programming examples from `programming_examples/`
- **tutorial**: Tutorial code from `programming_guide/`
- **benchmark**: Performance benchmarks

## Manifest Schema

See `schema.json` for the full JSON Schema definition. Key fields:

```json
{
  "id": "test_npu-xrt_add_one_using_dma",
  "name": "add_one_using_dma",
  "source_path": "test/npu-xrt/add_one_using_dma",
  "architecture": "aie2",
  "devices": ["npu1"],
  "source_type": "mlir",
  "has_host_test": true,
  "category": "integration",
  "features": ["dma", "locks", "shim_dma", "core_kernel"]
}
```

## Current Statistics

As of last discovery:
- **203 total tests** discovered
- 51 AIE1, 82 AIE2, 2 AIE2P, 68 multi-device
- 183 tests have host test (test.cpp)
- 65 integration, 61 unit_test, 55 example, 14 benchmark, 8 tutorial

## Workflow

1. Build mlir-aie tests: `cd mlir-aie && ninja -C build`
2. Discover tests: `./discover_tests.py --summary`
3. Query what's available: `./query_tests.py --arch aie2 --has-host-test`
4. Run tests: `./run_tests.py --arch aie2`

## Known Limitations

- Tests showing "PASS (1 cycles)" only verify loading, not execution correctness
- Execution validation requires matching test.cpp expected values
- Some tests require pre-built kernel objects (.o files)
- AIE1 and AIE2P tests cannot run on the current emulator (AIE2 only)
