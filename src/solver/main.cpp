// SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
//
// SPDX-License-Identifier: GPL-2.0-only

#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

void test_connection() {
    std::cout << "Test: The C++ code works!" << std::endl;
}

PYBIND11_MODULE(_solver_ext, m) {
    m.def("test_connection", &test_connection, "Prints a test message from C++");
}