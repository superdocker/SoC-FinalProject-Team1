/**
* Copyright (C) 2020 Xilinx, Inc
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "xcl2.hpp"
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#define ROW_SIZE 32
#define NUM_FEATURE 40
#define KERNEL_SIZE 3

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    // size_t vector_size_bytes = sizeof(int) * DATA_SIZE;
    cl_int err;
    cl::Context context;
    cl::Kernel krnl_vector_add;
    cl::CommandQueue q;

    // [JHLEE]
    typedef int data_t;
    std::vector<data_t, aligned_allocator<data_t>> csr_row(ROW_SIZE+1);
    std::vector<data_t, aligned_allocator<data_t>> csr_col(NUM_FEATURE);
    std::vector<data_t, aligned_allocator<data_t>> Rule(KERNEL_SIZE*KERNEL_SIZE*2*NUM_FEATURE);
    FILE *frow, *fcol;
    frow = fopen("csr_row.dat","r");
    fcol = fopen("csr_col.dat","r");
    for (int i=0; i<=ROW_SIZE; i++){
        fscanf(frow,"%d",&csr_row[i]);
    }
    for (int i=0; i<NUM_FEATURE; i++){
        fscanf(fcol,"%d",&csr_col[i]);
    }
    fclose(frow);
    fclose(fcol);

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_vector_add = cl::Kernel(program, "vadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // [JHLEE]
    OCL_CHECK(err, cl::Buffer buffer_x(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t)*(ROW_SIZE+1), csr_row.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_y(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t)*NUM_FEATURE, csr_col.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_z(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, sizeof(data_t)*(2*NUM_FEATURE*9), Rule.data(), &err));
    OCL_CHECK(err, err = krnl_vector_add.setArg(0, buffer_x));
    OCL_CHECK(err, err = krnl_vector_add.setArg(1, buffer_y));
    OCL_CHECK(err, err = krnl_vector_add.setArg(2, buffer_z));

    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_x, buffer_y}, 0 /* 0 means from host*/));
    q.finish();

    // [JHLEE]
    auto start = std::chrono::steady_clock::now();
    OCL_CHECK(err, err = q.enqueueTask(krnl_vector_add));
    q.finish();
    auto end = std::chrono::steady_clock::now();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_z}, CL_MIGRATE_MEM_OBJECT_HOST));

    // Copy Result from Device Global Memory to Host Local Memory
    q.finish();
    double exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();

    printf("[JHLEE] FPGA kernel exec time is %f s\n", exec_time*1e-9);
    if (true){
        int RK = NUM_FEATURE*2;
        int RO = NUM_FEATURE;
        printf("\n\n\n================= Summary =================\n\n");

        printf("[Result] Rule generation done\n");
        for (int i=0; i<9; i++){
            printf("   Kernel %2d  ",i);
        }
        printf("\n");
        for (int i=0; i<9; i++){
            printf(" Input Output ");
        }
        for (int l=0; l<NUM_FEATURE; l++){
            printf("\n");
            for (int k=0; k<KERNEL_SIZE*KERNEL_SIZE; k++){
                printf("|  %4d %4d  |",Rule[(8-k)*RK+l],Rule[(8-k)*RK+RO+l]);
            }
        }
        printf("\n");
    }

    // [JHLEE]
    return EXIT_SUCCESS;
}
