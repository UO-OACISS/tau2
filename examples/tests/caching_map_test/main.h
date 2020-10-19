#pragma once

//constexpr size_t numThreads{16};
const size_t numThreads = thread::hardware_concurrency() / 4;
//const size_t numThreads = thread::hardware_concurrency();
constexpr size_t iterations{1000000};
constexpr size_t maxValue{100};
bool pin_threads_to_cores{true};
