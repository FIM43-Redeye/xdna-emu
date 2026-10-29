// SPDX-License-Identifier: Apache-2.0
// Simple test for mock XRT + xdna-emu integration
// This test verifies the FFI integration works end-to-end

#include "xrt/xrt_device.h"
#include "xrt/xrt_xclbin.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <test_dir>" << std::endl;
        std::cerr << "  test_dir should contain: aie.xclbin, insts.bin" << std::endl;
        return 1;
    }

    std::string test_dir = argv[1];
    std::string xclbin_path = test_dir + "/aie.xclbin";
    std::string insts_path = test_dir + "/insts.bin";

    std::cout << "=== Mock XRT + xdna-emu Integration Test ===" << std::endl;
    std::cout << "Test directory: " << test_dir << std::endl;

    try {
        // Step 1: Create device
        std::cout << "\n1. Creating device..." << std::endl;
        xrt::device device(0);
        std::cout << "   Device created" << std::endl;

        // Step 2: Load xclbin
        std::cout << "\n2. Loading xclbin..." << std::endl;
        xrt::xclbin xclbin(xclbin_path);
        xrt::uuid xclbin_uuid = device.register_xclbin(xclbin);
        std::cout << "   Xclbin loaded: " << xclbin_uuid.to_string() << std::endl;

        // Step 3: Create hardware context
        std::cout << "\n3. Creating hardware context..." << std::endl;
        xrt::hw_context ctx(device, xclbin_uuid);
        std::cout << "   Context created" << std::endl;

        // Step 4: Create kernel
        std::cout << "\n4. Creating kernel..." << std::endl;
        xrt::kernel kernel(ctx, "MLIR_AIE");
        std::cout << "   Kernel created" << std::endl;

        // Step 5: Load instruction binary
        std::cout << "\n5. Loading instructions from " << insts_path << "..." << std::endl;
        std::ifstream insts_file(insts_path, std::ios::binary);
        if (!insts_file) {
            std::cerr << "   ERROR: Cannot open " << insts_path << std::endl;
            return 1;
        }
        insts_file.seekg(0, std::ios::end);
        size_t insts_size = insts_file.tellg();
        insts_file.seekg(0, std::ios::beg);
        std::vector<uint32_t> insts(insts_size / 4);
        insts_file.read(reinterpret_cast<char*>(insts.data()), insts_size);
        std::cout << "   Loaded " << insts.size() << " instruction words" << std::endl;

        // Step 6: Create buffer objects
        // Match the original test structure: bo_instr, bo_inA, bo_inB, bo_out
        constexpr size_t DATA_SIZE = 64 * sizeof(int32_t);  // 64 elements
        constexpr size_t INSTR_SIZE_BYTES = 4096;  // Max instruction buffer

        std::cout << "\n6. Creating buffer objects..." << std::endl;
        xrt::bo bo_instr(device, INSTR_SIZE_BYTES, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        xrt::bo bo_inA(device, DATA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        xrt::bo bo_inB(device, DATA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        xrt::bo bo_out(device, DATA_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

        std::cout << "   Instruction buffer: " << bo_instr.size() << " bytes at 0x"
                  << std::hex << bo_instr.address() << std::dec << std::endl;
        std::cout << "   Input A buffer: " << bo_inA.size() << " bytes at 0x"
                  << std::hex << bo_inA.address() << std::dec << std::endl;
        std::cout << "   Input B buffer: " << bo_inB.size() << " bytes at 0x"
                  << std::hex << bo_inB.address() << std::dec << std::endl;
        std::cout << "   Output buffer: " << bo_out.size() << " bytes at 0x"
                  << std::hex << bo_out.address() << std::dec << std::endl;

        // Step 7: Initialize buffers
        std::cout << "\n7. Initializing buffers..." << std::endl;

        // Copy instructions
        void* instr_ptr = bo_instr.map<void*>();
        std::memcpy(instr_ptr, insts.data(), std::min(insts_size, INSTR_SIZE_BYTES));

        // Initialize input A with sequential values (1, 2, 3, ..., 64)
        int32_t* inA_ptr = bo_inA.map<int32_t*>();
        for (int i = 0; i < 64; i++) {
            inA_ptr[i] = i + 1;
        }

        // Input B is unused but initialize anyway
        int32_t* inB_ptr = bo_inB.map<int32_t*>();
        std::memset(inB_ptr, 0, DATA_SIZE);

        // Clear output
        int32_t* out_ptr = bo_out.map<int32_t*>();
        std::memset(out_ptr, 0, DATA_SIZE);

        std::cout << "   Input (first 8): ";
        for (int i = 0; i < 8; i++) {
            std::cout << inA_ptr[i] << " ";
        }
        std::cout << std::endl;

        // Step 8: Sync buffers to device
        std::cout << "\n8. Syncing buffers to device..." << std::endl;
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Step 9: Execute kernel
        // Use opcode 3 to match original test
        std::cout << "\n9. Executing kernel..." << std::endl;
        auto run = kernel(3, bo_instr, insts.size(), bo_inA, bo_inB, bo_out);
        ert_cmd_state state = run.wait();

        std::cout << "   Kernel completed with state: " << state << std::endl;

        // Step 10: Sync output from device
        std::cout << "\n10. Syncing output from device..." << std::endl;
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Step 11: Check results
        std::cout << "\n11. Checking results..." << std::endl;
        std::cout << "   Output (first 8): ";
        for (int i = 0; i < 8; i++) {
            std::cout << out_ptr[i] << " ";
        }
        std::cout << std::endl;

        // add_one_objFifo actually adds 41 (not 1), so expected is input + 41
        // which means for input 1,2,3... the output is 42,43,44... = i + 42
        int correct = 0;
        int zeros = 0;
        int wrong = 0;
        for (int i = 0; i < 64; i++) {
            int32_t expected = i + 42;  // This test adds 41 to i+1 = i+42
            if (out_ptr[i] == expected) {
                correct++;
            } else if (out_ptr[i] == 0) {
                zeros++;
            } else {
                wrong++;
                if (wrong <= 5) {
                    std::cout << "   Mismatch[" << i << "]: got " << out_ptr[i]
                              << " expected " << expected << std::endl;
                }
            }
        }

        std::cout << "\n=== Results ===" << std::endl;
        std::cout << "Correct: " << correct << "/64" << std::endl;
        std::cout << "Zeros: " << zeros << std::endl;
        std::cout << "Wrong: " << wrong << std::endl;

        if (correct == 64) {
            std::cout << "TEST PASSED!" << std::endl;
            return 0;
        } else if (zeros > 0) {
            std::cout << "TEST PARTIAL: Some outputs not computed" << std::endl;
            return 1;
        } else {
            std::cout << "TEST FAILED: Incorrect outputs" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
