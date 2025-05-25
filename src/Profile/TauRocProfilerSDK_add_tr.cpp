// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//https://github.com/ROCm/rocprofiler-sdk/blob/ccd1e54293768a756fb95c21bff51d95d5f6b20c/tests/pc_sampling/address_translation.cpp

#include "Profile/RocProfilerSDK/TauRocProfilerSDK_add_tr.hpp"
#ifdef SAMPLING_SDKADD
#include <atomic>



namespace sdk_pc_sampling
{

std::atomic<uint64_t> total_samples_num = 0;

std::atomic<uint64_t>& get_total_samples_num()
{
    return total_samples_num;
}

void inc_total_samples_num()
{
    total_samples_num++;
}

namespace address_translation
{
namespace
{
struct FlatProfiler
{
public:
    FlatProfiler()  = default;
    ~FlatProfiler() = default;

    CodeobjAddressTranslate translator = {};
    KernelObjectMap         kernel_object_map = {};
    FlatProfile             flat_profile = {};
    std::mutex              global_mut = {};
};
}  // namespace

// Raw pointer to prevent early destruction of static objects
FlatProfiler* flat_profiler = nullptr;

void
init()
{
    flat_profiler = new FlatProfiler();
}

void
fini()
{
    delete flat_profiler;
}

CodeobjAddressTranslate&
get_address_translator()
{
    return flat_profiler->translator;
}

KernelObjectMap&
get_kernel_object_map()
{
    return flat_profiler->kernel_object_map;
}

FlatProfile&
get_flat_profile()
{
    return flat_profiler->flat_profile;
}

std::mutex&
get_global_mutex()
{
    return flat_profiler->global_mut;
}

KernelObject::KernelObject(uint64_t    code_object_id,
                           std::string kernel_name,
                           uint64_t    begin_address,
                           uint64_t    end_address)
: code_object_id_(code_object_id)
, kernel_name_(kernel_name)
, begin_address_(begin_address)
, end_address_(end_address)
{
    auto&    translator = get_address_translator();
    uint64_t vaddr      = begin_address;
    while(vaddr < end_address)
    {
        auto inst = translator.get(code_object_id, vaddr);
        vaddr += inst->size;
        this->add_instruction(std::move(inst));
    }
}

}  // namespace address_translation
}  // namespace sdk_pc_sampling
#endif //SAMPLING_SDKADD