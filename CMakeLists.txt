# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

cmake_minimum_required(VERSION 3.15)
project(dissertation_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_solver_ext src/solver/main.cpp)

if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries(_solver_ext PRIVATE "-static")
endif()

install(TARGETS _solver_ext DESTINATION src/solver)