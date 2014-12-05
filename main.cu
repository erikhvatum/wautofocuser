// The MIT License (MIT)
// 
// Copyright (c) 2014 WUSTL ZPLAB
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
// 
// Authors: Erik Hvatum <ice.rikh@gmail.com> 

#include <cstdint>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fstream>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <memory>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

void read_float_blob(const char* fn, float* b, std::size_t size)
{
    std::fstream f(fn, std::ios_base::in | std::ios_base::binary);
    if(!f.read(reinterpret_cast<char*>(b), size * sizeof(float)))
    {
        throw std::string("Failed to read all of float blob ") + std::string(fn) + '.';
    }
}

void write_float_blob(const char* fn, const float* b, std::size_t size)
{
    std::fstream f(fn, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    if(!f.write(reinterpret_cast<const char*>(b), size * sizeof(float)))
    {
        throw std::string("Failed to write all of float blob ") + std::string(fn) + '.';
    }
}

void read_complex_float_blob(const char* fn, thrust::host_vector<cuFloatComplex>& b)
{
    std::fstream f(fn, std::ios_base::in | std::ios_base::binary);
    if(!f.read(reinterpret_cast<char*>(b.data()), b.size() * sizeof(float) * 2))
    {
        throw std::string("Failed to read all of complex float blob ") + std::string(fn) + '.';
    }
}

void write_complex_float_blob(const char* fn, const thrust::host_vector<cuFloatComplex>& b)
{
    std::fstream f(fn, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
    if(!f.write(reinterpret_cast<const char*>(b.data()), b.size() * sizeof(float) * 2))
    {
        throw std::string("Failed to write all of complex float blob ") + std::string(fn) + '.';
    }
}
 
int main(int argc, char** argv)
{
    int ret{0};
    findCudaDevice(argc, const_cast<const char **>(argv));
    try
    {
        std::size_t im_size(2560 * 2160);
        thrust::host_vector<cuFloatComplex> im_h_in(im_size);
        read_complex_float_blob("/home/ehvatum/zplrepo/wautofocuser/build/in.complexfloatblob", im_h_in);
        thrust::device_vector<cuFloatComplex> im_d_in(im_h_in);

        cufftHandle plan;
        if(cufftPlan2d(&plan, 2160, 2560, CUFFT_C2C) != CUFFT_SUCCESS)
        {
            throw std::string("cufftPlan2d(..) failed.");
        }
        struct DestroyPlan
        {
            cufftHandle plan;
            DestroyPlan(cufftHandle plan_) : plan{plan_}{}
            ~DestroyPlan()
            {
                cufftDestroy(plan);
            }
        } destroy_plan(plan);

        thrust::device_vector<cuFloatComplex> im_d_out(im_size);
        if(cufftExecC2C(plan,
                        thrust::raw_pointer_cast(im_d_in.data()),
                        thrust::raw_pointer_cast(im_d_out.data()),
                        CUFFT_FORWARD) != CUFFT_SUCCESS)
        {
            throw std::string("cufftExecC2C(..) failed.");
        }

        thrust::host_vector<cuFloatComplex> im_h_out(im_d_out);
        write_complex_float_blob("/home/ehvatum/zplrepo/wautofocuser/build/out.complexfloatblob", im_h_out);
    }
    catch(const std::string& e)
    {
        std::cerr << e << std::endl;
        ret = -1;
    }
    // cudaDeviceReset causes the driver to clean up all state.  While not mandatory in normal operation, it is good
    // practice.  It is also needed to ensure correct operation when the application is being profiled.  Calling 
    // cudaDeviceReset causes all profile data to be flushed before the application exits.
    cudaDeviceReset();
    return ret;
}
