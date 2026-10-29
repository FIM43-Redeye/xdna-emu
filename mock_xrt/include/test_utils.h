// SPDX-License-Identifier: Apache-2.0
// Minimal test utilities for mock XRT
// Drop-in replacement for mlir-aie's test_utils.h that works with our mock

#ifndef MOCK_XRT_TEST_UTILS_H_
#define MOCK_XRT_TEST_UTILS_H_

#include "cxxopts.hpp"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_utils {

/// Add default command-line options expected by mlir-aie tests
inline void add_default_options(cxxopts::Options& options) {
    options.add_options()
        ("x,xclbin", "Path to xclbin file", cxxopts::value<std::string>())
        ("k,kernel", "Kernel name", cxxopts::value<std::string>()->default_value("MLIR_AIE"))
        ("i,instr", "Path to instruction binary", cxxopts::value<std::string>())
        ("v,verbosity", "Verbosity level", cxxopts::value<int>()->default_value("0"))
        ("verify", "Verify results (default: true)", cxxopts::value<bool>()->default_value("true"))
        ("iters", "Number of iterations", cxxopts::value<int>()->default_value("1"))
        ("warmup", "Number of warmup iterations", cxxopts::value<int>()->default_value("0"))
        ("trace_sz", "Trace buffer size", cxxopts::value<int>()->default_value("0"))
        ("trace_file", "Trace output file", cxxopts::value<std::string>()->default_value(""))
        ("h,help", "Print help");
}

/// Parse command-line options
inline void parse_options(int argc, const char** argv,
                         cxxopts::Options& options,
                         cxxopts::ParseResult& result) {
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::exceptions::exception& e) {
        std::cerr << "Error parsing options: " << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (!result.count("xclbin")) {
        std::cerr << "Error: --xclbin is required" << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }

    if (!result.count("instr")) {
        std::cerr << "Error: --instr is required" << std::endl;
        std::cerr << options.help() << std::endl;
        exit(1);
    }
}

/// Load instruction binary file
inline std::vector<uint32_t> load_instr_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open instruction file: " + path);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size % 4 != 0) {
        throw std::runtime_error("Instruction file size not aligned to 4 bytes");
    }

    std::vector<uint32_t> instructions(size / 4);
    if (!file.read(reinterpret_cast<char*>(instructions.data()), size)) {
        throw std::runtime_error("Failed to read instruction file: " + path);
    }

    return instructions;
}

/// Load instruction sequence from text file (one hex value per line)
inline std::vector<uint32_t> load_instr_sequence(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open instruction file: " + path);
    }

    std::vector<uint32_t> instructions;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        uint32_t value = std::stoul(line, nullptr, 16);
        instructions.push_back(value);
    }

    return instructions;
}

} // namespace test_utils

#endif // MOCK_XRT_TEST_UTILS_H_
