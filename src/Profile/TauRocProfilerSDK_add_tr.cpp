
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

    CodeobjAddressTranslate translator;
    KernelObjectMap         kernel_object_map;
    FlatProfile             flat_profile;
    std::mutex              global_mut;
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

void
dump_flat_profile(const char* output_filename)
{
    // It seems that an instruction can be part of multiple
    // instances of the same kernel loaded on two different devices.
    // We need to prevent counting the same instruction multiple times.
    std::unordered_set<Instruction*> visited_instructions;

    const auto& kernel_object_map = get_kernel_object_map();
    const auto& flat_profile      = get_flat_profile();

    std::stringstream ss;
    uint64_t          samples_num = 0;
    kernel_object_map.iterate_kernel_objects([&](const KernelObject* kernel_obj) {
        ss << "\n====================================";
        ss << "The kernel: " << kernel_obj->kernel_name()
           << " with the begin address: " << kernel_obj->begin_address()
           << " from code object with id: " << kernel_obj->code_object_id() << std::endl;
        kernel_obj->iterate_instrunctions([&](const Instruction& inst) {
            ss << kernel_obj->kernel_name() ;
            ss << "\t";
            ss << inst.inst << "\t";
            ss << inst.comment << "\t";
            ss << inst.faddr << "\t";
            ss << inst.vaddr << "\t";
            ss << inst.ld_addr << "\t";
            ss << inst.codeobj_id << "\t";
            const auto* _sample_instruction = flat_profile.get_sample_instruction(inst);

            if(_sample_instruction != nullptr)
            {
                _sample_instruction->process([&](const SampleInstruction& sample_instruction) {

                    ss << "samples: ";
                    ss << sample_instruction.sample_count();

                    // Each instruction should be visited exactly once.
                    // Otherwise, code object loading/unloading and relocations
                    // are not handled properly.
                    assert(visited_instructions.count(sample_instruction.inst()) == 0);
                    // Assure that each instruction is counted once.
                    if(visited_instructions.count(sample_instruction.inst()) == 0)
                    {
                        samples_num += sample_instruction.sample_count();
                        visited_instructions.insert(sample_instruction.inst());
                    }

                    if(sample_instruction.exec_mask_counts().size() <= 1)
                    {
                        ss << ", exec_mask: " << std::hex;
                        ss << sample_instruction.exec_mask_counts().begin()->first;
                        ss << std::dec;
                        assert(sample_instruction.sample_count() ==
                               sample_instruction.exec_mask_counts().begin()->second);
                    }
                    else
                    {
                        uint64_t num_samples_sum = 0;
                        // More than one exec_mask
                        for(auto& [exec_mask, samples_per_exec] :
                            sample_instruction.exec_mask_counts())
                        {
                            ss << std::endl;
                            ss << "\t\t"
                               << "exec_mask: " << std::hex << exec_mask;
                            ss << "\t"
                               << "samples: " << std::dec << samples_per_exec;
                            num_samples_sum += samples_per_exec;
                            ss << std::endl;
                        }
                        assert(sample_instruction.sample_count() == num_samples_sum);
                    }
                });
            }
            ss << std::endl;
        });
        ss << "====================================\n" << std::endl;
    });

    ss << "The total number of decoded   samples: " << samples_num << std::endl;
    ss << "The total number of collected samples: " << sdk_pc_sampling::get_total_samples_num()
       << std::endl;

    //std::cout << ss.str() << std::endl;
    
    std::ofstream out(output_filename);
    out << ss.str();
    out.close();

    assert(samples_num == get_total_samples_num());
    // We expect at least one PC sample to be decoded/delivered;
    assert(samples_num > 0);
}

}  // namespace address_translation
}  // namespace sdk_pc_sampling
#endif //SAMPLING_SDKADD