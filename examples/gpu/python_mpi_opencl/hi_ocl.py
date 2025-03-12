# MPI with OpenCL Python example
from mpi4py import MPI
import pyopencl as cl
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# OpenCL setup
platform = cl.get_platforms()[0]  # Select the first platform
device = platform.get_devices()[0]  # Select the first device
context = cl.Context([device])
queue = cl.CommandQueue(context)

# OpenCL kernel for element-wise squaring
kernel_code = """
__kernel void square(__global const float *input, __global float *output) {
    int i = get_global_id(0);
    output[i] = input[i] * input[i];
}
"""
program = cl.Program(context, kernel_code).build()

# Data preparation (only rank 0 initializes the full data array)
if rank == 0:
    data = np.linspace(1, 100, 100, dtype=np.float32)  # Example data: 100 elements
    chunk_size = len(data) // size
else:
    data = None
    chunk_size = None

# Broadcast chunk size to all processes
chunk_size = comm.bcast(chunk_size, root=0)

# Scatter data to all processes
local_data = np.empty(chunk_size, dtype=np.float32)
comm.Scatter(data, local_data, root=0)

# Allocate buffers for OpenCL computation
mf = cl.mem_flags
input_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=local_data)
output_buffer = cl.Buffer(context, mf.WRITE_ONLY, local_data.nbytes)

# Execute the kernel on the GPU
global_size = (chunk_size,)
program.square(queue, global_size, None, input_buffer, output_buffer)

# Retrieve results from the GPU
local_result = np.empty_like(local_data)
cl.enqueue_copy(queue, local_result, output_buffer).wait()

# Gather results at rank 0
result = None
if rank == 0:
    result = np.empty(len(data), dtype=np.float32)

comm.Gather(local_result, result, root=0)

# Print results on rank 0
if rank == 0:
    print("Original Data:", data)
    print("Squared Data:", result)

