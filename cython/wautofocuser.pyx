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
# Authors: Erik Hvatum <ice.rikh@gmail.com>, Zach Pincus

import cython
cimport numpy
import numpy

cdef extern from "cpp_lib/_Highpass.h":
    cdef cppclass _Highpass:
        _Highpass(size_t w, size_t h)
        size_t get_w() const
        size_t get_h() const
        object get_filter() const
        void set_filter(const float* filter)
        object apply(const float* image) const

cdef class Highpass:
    cdef _Highpass *thisptr

    def __cinit__(self, cutoff, h, w):
        self.thisptr = new _Highpass(w, h)
        f = self._highpass_butterworth_nd(1.0 / cutoff, (h, w), 1, 2).astype(numpy.float32)
        self.filter = f

    def __dealloc__(self):
        del self.thisptr

    def __call__(self, numpy.ndarray[numpy.float32_t, ndim=2, mode="c"] image):
        return self.thisptr.apply(&image[0,0])

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

    def _highpass_butterworth_nd(self, cutoff, shape, d=1.0, order=2):
        def fftfreq_nd(shape, d=1.0):
              """Return an array containing the frequency bins of an N-dimensional FFT.
              Parameter 'd' specifies the sample spacing."""
              nd = len(shape)
              ds = numpy.resize(d, nd)
              freqs1d = [numpy.linspace(0,1,n, endpoint=False)/d for n,d in zip(shape, ds)]
              freqs1d = [numpy.where(freqs < 0.5, freqs, freqs-1) for freqs in freqs1d]
              s = numpy.ones(shape)
              freqsnd = [numpy.swapaxes(freqs * numpy.swapaxes(s, i, nd-1), i, nd-1) for i,freqs in enumerate(freqs1d)]
              freqsnd = numpy.sqrt(numpy.sum(numpy.square(freqsnd), axis=0))
              return freqsnd
        """Create a high-pass butterworth filter with the given pass-band and
        n-dimensional shape. The 'd' parameter is a scalar or list giving the sample
        spacing in all/each dimension, and the 'order' parameter controls the order
        of the butterworth filter."""
        cutoff = float(cutoff)
        if cutoff > 0.5:
            raise ValueError('Filter cutoff frequency must be <= 0.5.')
        return 1.0 - (1.0 / (1.0 + (fftfreq_nd(shape, d) / cutoff)**(2*order)))

#
##   def filter_nd(array, filter):
##         """Filter an array's fft with a the given filter coefficients."""
##         array, filter = numpy.asarray(array), numpy.asarray(filter)
##         fft = numpy.fft.fftn(array)
##         filtered = fft * filter
##         return numpy.fft.ifftn(filtered)
