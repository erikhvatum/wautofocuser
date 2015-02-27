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
 
#pragma once

#include <cstddef>
#include <cufft.h>
#include <Python.h>
#include <thrust/device_vector.h>

class _SpatialFilter
{
public:
    _SpatialFilter(std::size_t w, std::size_t h);
    virtual ~_SpatialFilter();

    std::size_t get_w() const;
    std::size_t get_h() const;
    void set_filter(const float* filter);
    PyObject* get_filter() const;
    PyObject* apply(const float* image) const;

protected:
    std::size_t m_h;
    std::size_t m_w;
    std::size_t m_s;
    thrust::device_vector<float> m_filter;
    cufftHandle m_forward_plan;
    cufftHandle m_inverse_plan;
};
