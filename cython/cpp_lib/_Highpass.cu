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
#include <iostream>
#include <thrust/host_vector.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL wautofocuser_ARRAY_API
#include <numpy/arrayobject.h>

static void* do_import_array()
{
    // import_array() is actually a macro that returns NULL if it fails, so it has to be wrapped in order to be called
    // from a constructor which necessarily does not return anything
    import_array();
    return reinterpret_cast<void*>(1);
}

static bool array_imported{false};

_Highpass::_Highpass(std::size_t w, std::size_t h)
  : m_w(w),
    m_h(h),
    m_filter(w * h)
{
    if(!array_imported)
    {
        do_import_array();
        array_imported = true;
    }
}

_Highpass::~_Highpass()
{
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
    memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ret)), (const void*)temp.data(), m_w * m_h);
    return ret;
}

void _Highpass::set_filter(const float* filter)
{
    m_filter.assign(filter, filter + m_w * m_h);
}
