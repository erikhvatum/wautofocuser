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
 
#include "_Highpass.h"
#include "GilStateScopeOperators.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <iostream>
#include <stdexcept>
#include <thrust/adjacent_difference.h>
#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_allocator.h>
#include <thrust/device_delete.h>
#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/device_new.h>
#include <thrust/device_new_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/logical.h>
#include <thrust/memory.h>
#include <thrust/merge.h>
#include <thrust/mismatch.h>
#include <thrust/pair.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/replace.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/swap.h>
#include <thrust/system_error.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/uninitialized_fill.h>
#include <thrust/unique.h>
#include <thrust/version.h>

using namespace thrust::placeholders;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL wautofocuser_ARRAY_API
#include <numpy/arrayobject.h>

static void* do_static_init()
{
    // import_array() is actually a macro that returns NULL if it fails, so it has to be wrapped in order to be called
    // from a constructor which necessarily does not return anything
    import_array();
    findCudaDevice(0, nullptr);
    return reinterpret_cast<void*>(1);
}

static bool static_inited{false};

_Highpass::_Highpass(std::size_t w, std::size_t h)
  : m_w(w),
    m_h(h),
    m_s(w * h),
    m_filter(m_s),
    m_forward_plan(0),
    m_inverse_plan(0)
{
    if(!static_inited)
    {
        do_static_init();
        static_inited = true;
    }
    if(cufftPlan2d(&m_forward_plan, m_h, m_w, CUFFT_C2C) != CUFFT_SUCCESS)
    {
        throw std::string("cufftPlan2d(..) for forward transformation failed.");
    }
    if(cufftPlan2d(&m_inverse_plan, m_h, m_w, CUFFT_C2C) != CUFFT_SUCCESS)
    {
        throw std::string("cufftPlan2d(..) for inverse transformation failed.");
    }
}

_Highpass::~_Highpass()
{
    if(m_forward_plan)
    {
        cufftDestroy(m_forward_plan);
        m_forward_plan = 0;
    }
    if(m_inverse_plan)
    {
        cufftDestroy(m_inverse_plan);
        m_inverse_plan = 0;
    }
}

std::size_t _Highpass::get_w() const
{
    return m_w;
}

std::size_t _Highpass::get_h() const
{
    return m_h;
}

PyObject* _Highpass::get_filter() const
{
    GilLocker gil_locker;
    PyObject* ret;
    npy_intp dims[] = {static_cast<npy_intp>(m_h), static_cast<npy_intp>(m_w)};
    ret = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    thrust::host_vector<float> temp(m_filter);
    memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret)), (const void*)temp.data(), m_w * m_h * sizeof(float));
    return ret;
}

void _Highpass::set_filter(const float* filter)
{
    m_filter.assign(filter, filter + m_w * m_h);
}

struct ComplexToReal
{
    __device__ __host__ float operator () (const cuFloatComplex cf) const
    {
        return cf.x;
    }
};

struct RealToComplex
{
    __device__ __host__ cuFloatComplex operator () (const float f) const
    {
        cuFloatComplex cf;
        cf.x = f;
        cf.y = 0;
        return cf;
    }
};

struct RealComplexMult
{
    __device__ __host__ float2 operator () (const float& r, const cuFloatComplex& c) const
    {
        return {r * c.x, r * c.y};
    }
};

// struct MultiplyRealComponentAndReal
// {
//     __device__ __host__ cuFloatComplex operator () (const cuFloatComplex& c, const float& r) const
//     {
//         return c.x * f;
//     }
// };

// template<typename C, typename R>
// struct RtoC
// {
//     __device__ __host__ C operator () (const R& r) const
//     {
//         C c;
//         c.x = r;
//         c.y = 0;
//         return c;
//     }
// };

// struct Functor
// {
//     __device__ __host__ float operator () (const float& rhs) const
//     {
//         return rhs * 2;
//     }
// };

PyObject* _Highpass::apply(const float* image) const
{
//  thrust::host_vector<float> im_h(image, image + m_w * m_h);
//  thrust::device_vector<float> im_d(im_h);
//  using namespace thrust::placeholders;
//  thrust::transform(thrust::device, im_h.begin(), im_h.end(), thrust::device_pointer_cast(im_d.data()), thrust::negate<float>());
//  thrust::transform(im_h.begin(), im_h.end(), im_d.begin(), make_cuComplex);
//  RtoC<cuFloatComplex, float> rtoc;
//  thrust::transform(thrust::host, im_h.begin(), im_h.end(), im_d.begin(), rtoc);

    PyObject* ret;
    thrust::host_vector<cuFloatComplex> image_h(m_s);
    thrust::transform(image, image + m_s, image_h.begin(), RealToComplex());
    thrust::device_vector<cuFloatComplex> image_d(image_h);

    thrust::device_vector<cuFloatComplex> fft_d(m_s);
    if(cufftExecC2C(m_forward_plan,
                    thrust::raw_pointer_cast(image_d.data()),
                    thrust::raw_pointer_cast(fft_d.data()),
                    CUFFT_FORWARD) != CUFFT_SUCCESS)
    {
        throw std::runtime_error("cufftExecC2C(..) forward transformation failed.");
    }

    thrust::device_vector<cuFloatComplex> fft_t_d(m_s);
    thrust::transform(thrust::device,
                      m_filter.begin(),
                      m_filter.end(),
                      fft_d.begin(),
                      fft_t_d.begin(),
                      RealComplexMult());

    thrust::device_vector<cuFloatComplex> out_d(m_s);
    if(cufftExecC2C(m_inverse_plan,
                    thrust::raw_pointer_cast(fft_t_d.data()),
                    thrust::raw_pointer_cast(out_d.data()),
                    CUFFT_INVERSE) != CUFFT_SUCCESS)
    {
        throw std::runtime_error("cufftExecC2C(..) inverse transformation failed.");
    }
    thrust::host_vector<cuFloatComplex> out_h(out_d);

    {
        GilLocker gil_locker;
        npy_intp dims[] = {static_cast<npy_intp>(m_h), static_cast<npy_intp>(m_w)};
        ret = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
        thrust::transform(thrust::host,
                          out_h.begin(),
                          out_h.end(),
                          reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret))),
                          ComplexToReal());
    }
    return ret;
}
