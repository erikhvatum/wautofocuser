# The MIT License (MIT)
#
# Copyright (c) 2014 WUSTL ZPLAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Authors: Erik Hvatum <ice.rikh@gmail.com>

import cython
cimport numpy
import numpy

cdef extern from "cpp_lib/_Highpass.h":
    cdef cppclass _Highpass:
        _Highpass(size_t w, size_t h)
        size_t get_w() const
        size_t get_h() const
        object get_filter() const
        void set_filter(const float*)

cdef class Highpass:
    cdef _Highpass *thisptr

    def __cinit__(self, h, w):
        self.thisptr = new _Highpass(w, h)

    def __dealloc__(self):
        del self.thisptr

    property w:
        def __get__(self):
            return self.thisptr.get_w()

    property h:
        def __get__(self):
            return self.thisptr.get_h()

    property filter:
        def __get__(self):
            return self.thisptr.get_filter()

        def __set__(self, numpy.ndarray[numpy.float32_t, ndim=2, mode="c"] f):
            if f.shape[0] != self.h or f.shape[1] != self.w:
                raise ValueError("Dimensions of provided filter must match the h and w parameters passed to Highpass's constructor.")
            self.thisptr.set_filter(&f[0,0])
