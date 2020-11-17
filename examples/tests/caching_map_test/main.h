#pragma once

//constexpr size_t numThreads{16};
#if defined(__APPLE__)
const size_t numThreads = thread::hardware_concurrency();
#else
const size_t numThreads = thread::hardware_concurrency() / 4;
#endif
constexpr size_t iterations{1000000};
constexpr size_t maxValue{100};
bool pin_threads_to_cores{true};
