// mpi_ze_vecadd.cpp
//
// A simple application demonstrating the use of MPI with Intel Level Zero.
//
// 1. Rank 0 initializes two vectors, A and B.
// 2. Data is distributed: A is scattered, B is broadcast to all ranks.
// 3. Each rank uses Level Zero to offload its portion of the vector addition
//    (C[i] = A[i] + B[i]) to a local GPU.
// 4. The results are gathered back to Rank 0.
// 5. Rank 0 verifies the result.
//
#include <mpi.h>
#include <ze_api.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <numeric>
#include <cmath>
#include <stdexcept>

// Simple macro for checking Level Zero API call results
#define ZE_CHECK(status, action) \
    if (status != ZE_RESULT_SUCCESS) { \
        throw std::runtime_error("Level Zero Error: " + std::string(action) + " failed with code " + std::to_string(status)); \
    }

// Reads a binary file (like a .spv kernel) into a vector
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

int main(int argc, char** argv) {
    // --- MPI Initialization ---
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // --- Problem Setup ---
    const size_t N = 1024 * 1024; // Total number of elements
    if (N % world_size != 0) {
        if (world_rank == 0) {
            std::cerr << "Error: Total elements N must be divisible by the number of MPI ranks." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    const size_t local_N = N / world_size; // Elements per rank

    std::vector<float> h_A, h_B; // Host vectors on Rank 0
    std::vector<float> local_h_A(local_N); // Local data for each rank
    std::vector<float> local_h_B(local_N); // Local data for each rank
    std::vector<float> local_h_C(local_N); // Local result for each rank

    if (world_rank == 0) {
        std::cout << "Running on " << world_size << " MPI ranks." << std::endl;
        std::cout << "Total vector size: " << N << ", size per rank: " << local_N << std::endl;
        h_A.resize(N);
        h_B.resize(N);
        // Initialize host data
        for (size_t i = 0; i < N; ++i) {
            h_A[i] = static_cast<float>(i);
            h_B[i] = static_cast<float>(N - i);
        }
    }
    
    // --- MPI Data Distribution ---
    // Scatter vector A from Rank 0 to all ranks
    MPI_Scatter(h_A.data(), local_N, MPI_FLOAT,
                local_h_A.data(), local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Broadcast the first part of vector B to all ranks (for this simple vecadd)
    // In a real mat-mul, you'd broadcast the whole matrix B. Here we only need the matching slice.
    // To keep it simple, we just create the B data locally. A broadcast would look like this:
    // MPI_Bcast(h_B.data(), N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // For simplicity here, each rank generates its part of B.
    size_t offset = world_rank * local_N;
    for (size_t i = 0; i < local_N; ++i) {
        local_h_B[i] = static_cast<float>(N - (offset + i));
    }


    try {
        // --- Intel Level-Zero Initialization ---
        ZE_CHECK(zeInit(0), "zeInit");

        // 1. Discover driver and device
        uint32_t driver_count = 0;
        ZE_CHECK(zeDriverGet(&driver_count, nullptr), "zeDriverGet count");
        ze_driver_handle_t driver_handle;
        ZE_CHECK(zeDriverGet(&driver_count, &driver_handle), "zeDriverGet handle");

        uint32_t device_count = 0;
        ZE_CHECK(zeDeviceGet(driver_handle, &device_count, nullptr), "zeDeviceGet count");
        ze_device_handle_t device_handle;
        ZE_CHECK(zeDeviceGet(driver_handle, &device_count, &device_handle), "zeDeviceGet handle");
        
        // Let user know which device is being used by this rank
        ze_device_properties_t device_properties = {ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
        ZE_CHECK(zeDeviceGetProperties(device_handle, &device_properties), "zeDeviceGetProperties");
        std::cout << "Rank " << world_rank << " using Device: " << device_properties.name << std::endl;

        // 2. Create context and command queue
        ze_context_handle_t context;
        ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC};
        ZE_CHECK(zeContextCreate(driver_handle, &context_desc, &context), "zeContextCreate");
        
        ze_command_queue_handle_t cmd_queue;
        ze_command_queue_desc_t cmd_queue_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
        cmd_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        ZE_CHECK(zeCommandQueueCreate(context, device_handle, &cmd_queue_desc, &cmd_queue), "zeCommandQueueCreate");

        ze_command_list_handle_t cmd_list;
        ze_command_list_desc_t cmd_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC};
        ZE_CHECK(zeCommandListCreate(context, device_handle, &cmd_list_desc, &cmd_list), "zeCommandListCreate");

        // 3. Allocate memory on the device
        void *d_A, *d_B, *d_C;
        size_t buffer_size = local_N * sizeof(float);
        ze_device_mem_alloc_desc_t mem_alloc_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC};

        ZE_CHECK(zeMemAllocDevice(context, &mem_alloc_desc, buffer_size, 1, device_handle, &d_A), "zeMemAllocDevice A");
        ZE_CHECK(zeMemAllocDevice(context, &mem_alloc_desc, buffer_size, 1, device_handle, &d_B), "zeMemAllocDevice B");
        ZE_CHECK(zeMemAllocDevice(context, &mem_alloc_desc, buffer_size, 1, device_handle, &d_C), "zeMemAllocDevice C");

        // 4. Load SPIR-V Kernel
        auto spirv_code = readFile("vector_add.spv");
        ze_module_handle_t module;
        ze_kernel_handle_t kernel;

        ze_module_desc_t module_desc = {ZE_STRUCTURE_TYPE_MODULE_DESC};
        module_desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
        module_desc.inputSize = spirv_code.size();
        module_desc.pInputModule = reinterpret_cast<const uint8_t*>(spirv_code.data());
        ZE_CHECK(zeModuleCreate(context, device_handle, &module_desc, &module, nullptr), "zeModuleCreate");
        
        ze_kernel_desc_t kernel_desc = {ZE_STRUCTURE_TYPE_KERNEL_DESC};
        kernel_desc.pKernelName = "vector_add";
        ZE_CHECK(zeKernelCreate(module, &kernel_desc, &kernel), "zeKernelCreate");
        
        // 5. Set kernel arguments
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &d_A), "zeKernelSetArgumentValue A");
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &d_B), "zeKernelSetArgumentValue B");
        ZE_CHECK(zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &d_C), "zeKernelSetArgumentValue C");
        
        // 6. Set thread group size and launch grid
        uint32_t group_size_x = 32;
        ZE_CHECK(zeKernelSetGroupSize(kernel, group_size_x, 1, 1), "zeKernelSetGroupSize");

        ze_group_count_t dispatch_traits;
        dispatch_traits.groupCountX = (uint32_t)ceil((float)local_N / group_size_x);
        dispatch_traits.groupCountY = 1;
        dispatch_traits.groupCountZ = 1;
        
        // 7. Build command list
        ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, d_A, local_h_A.data(), buffer_size, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy A");
        ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, d_B, local_h_B.data(), buffer_size, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy B");
        ZE_CHECK(zeCommandListAppendBarrier(cmd_list, nullptr, 0, nullptr), "zeCommandListAppendBarrier after copy");
        ZE_CHECK(zeCommandListAppendLaunchKernel(cmd_list, kernel, &dispatch_traits, nullptr, 0, nullptr), "zeCommandListAppendLaunchKernel");
        ZE_CHECK(zeCommandListAppendBarrier(cmd_list, nullptr, 0, nullptr), "zeCommandListAppendBarrier after kernel");
        ZE_CHECK(zeCommandListAppendMemoryCopy(cmd_list, local_h_C.data(), d_C, buffer_size, nullptr, 0, nullptr), "zeCommandListAppendMemoryCopy C");

        // 8. Execute and Synchronize
        ZE_CHECK(zeCommandListClose(cmd_list), "zeCommandListClose");
        ZE_CHECK(zeCommandQueueExecuteCommandLists(cmd_queue, 1, &cmd_list, nullptr), "zeCommandQueueExecuteCommandLists");
        ZE_CHECK(zeCommandQueueSynchronize(cmd_queue, UINT64_MAX), "zeCommandQueueSynchronize");

        // 9. Cleanup Level Zero resources
        ZE_CHECK(zeMemFree(context, d_C), "zeMemFree C");
        ZE_CHECK(zeMemFree(context, d_B), "zeMemFree B");
        ZE_CHECK(zeMemFree(context, d_A), "zeMemFree A");
        ZE_CHECK(zeKernelDestroy(kernel), "zeKernelDestroy");
        ZE_CHECK(zeModuleDestroy(module), "zeModuleDestroy");
        ZE_CHECK(zeCommandListDestroy(cmd_list), "zeCommandListDestroy");
        ZE_CHECK(zeCommandQueueDestroy(cmd_queue), "zeCommandQueueDestroy");
        ZE_CHECK(zeContextDestroy(context), "zeContextDestroy");

    } catch (const std::runtime_error& e) {
        std::cerr << "Rank " << world_rank << " caught an exception: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // --- MPI Data Aggregation ---
    std::vector<float> h_C;
    if (world_rank == 0) {
        h_C.resize(N);
    }
    MPI_Gather(local_h_C.data(), local_N, MPI_FLOAT,
               h_C.data(), local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // --- Verification on Rank 0 ---
    if (world_rank == 0) {
        bool success = true;
        for (size_t i = 0; i < N; ++i) {
            // Expected result of A[i] + B[i] is (i) + (N - i) = N
            if (std::abs(h_C[i] - N) > 1e-5) {
                std::cerr << "Verification FAILED at index " << i << "! ";
                std::cerr << "Expected: " << N << ", Got: " << h_C[i] << std::endl;
                success = false;
                break;
            }
        }

        if (success) {
            std::cout << "\n----------------------------------------" << std::endl;
            std::cout << "VERIFICATION SUCCESSFUL!" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        } else {
            std::cout << "\n----------------------------------------" << std::endl;
            std::cout << "VERIFICATION FAILED." << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }

    // --- MPI Finalization ---
    MPI_Finalize();
    return 0;
}
