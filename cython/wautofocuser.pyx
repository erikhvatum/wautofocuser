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

cdef extern from "cpp_lib/_SpatialFilter.h":
    cdef cppclass _SpatialFilter:
        _SpatialFilter(size_t w, size_t h)
        size_t get_w() const
        size_t get_h() const
        object get_filter() const
        void set_filter(const float* filter)
        object apply(const float* image) const

cdef class SpatialFilter:
    """SpatialFilter(w, h, low_cutoff_period=None, high_cutoff_period=None)"""
    cdef _SpatialFilter *thisptr

    def __cinit__(self, w, h, low_cutoff_period=None, high_cutoff_period=None):
        if low_cutoff_period is None and high_cutoff_period is None:
            raise ValueError('Either or both of low_cutoff_period and high_cutoff_period must be specified.')
        if low_cutoff_period is not None and low_cutoff_period < 2:
            raise ValueError('Value specified for low_cutoff_period must be >= 2 (the Nyquist period).')
        if high_cutoff_period is not None and high_cutoff_period < 2:
            raise ValueError('Value specified for high_cutoff_period must be >= 2 (the Nyquist period).')
        
        if high_cutoff_period is None:
            f = SpatialFilter._lowpass_butterworth_nd(1.0 / low_cutoff_period, (h, w), 1, 2).astype(numpy.float32)
        elif low_cutoff_period is None:
            f = SpatialFilter._highpass_butterworth_nd(1.0 / high_cutoff_period, (h, w), 1, 2).astype(numpy.float32)
        else:
            f = SpatialFilter._bandpass_butterworth_nd(1.0 / high_cutoff_period, 1.0 / low_cutoff_period, (h, w), 1, 2).astype(numpy.float32)
        self.thisptr = new _SpatialFilter(w, h)
        f[0, 0] = 0
        self.filter = f

    def __dealloc__(self):
        del self.thisptr

    def __call__(self, numpy.ndarray[numpy.float32_t, ndim=2] image):
        if   numpy.isfortran(image) and (image.shape[0] != self.w or image.shape[1] != self.h) or \
         not numpy.isfortran(image) and (image.shape[0] != self.h or image.shape[1] != self.w):
            raise ValueError("Dimensions of image to be filtered must match the h and w parameters passed to Highpass's constructor.")
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

        def __set__(self, numpy.ndarray[numpy.float32_t, ndim=2] f):
            if   numpy.isfortran(f) and (f.shape[0] != self.w or f.shape[1] != self.h) or \
             not numpy.isfortran(f) and (f.shape[0] != self.h or f.shape[1] != self.w):
                raise ValueError("Dimensions of provided filter must match the h and w parameters passed to Highpass's constructor.")
            self.thisptr.set_filter(&f[0,0])

    @staticmethod
    def _lowpass_butterworth_nd(cutoff, shape, d=1.0, order=2):
        """Create a low-pass butterworth filter with the given pass-band and 
        n-dimensional shape. The 'd' parameter is a scalar or list giving the sample
        spacing in all/each dimension, and the 'order' parameter controls the order
        of the butterworth filter.  
        """
        cutoff = float(cutoff)
        if cutoff > 0.5:
            raise ValueError('Filter cutoff frequency must be <= 0.5.')
        return 1.0 / (1.0 + (SpatialFilter._fftfreq_nd(shape, d) / cutoff)**(2*order))

    @staticmethod
    def _highpass_butterworth_nd(cutoff, shape, d=1.0, order=2):
        """Create a high-pass butterworth filter with the given pass-band and
        n-dimensional shape. The 'd' parameter is a scalar or list giving the sample
        spacing in all/each dimension, and the 'order' parameter controls the order
        of the butterworth filter."""
        return 1.0 - SpatialFilter._lowpass_butterworth_nd(cutoff, shape, d, order)

    @staticmethod
    def _bandpass_butterworth_nd(low_cutoff, high_cutoff, shape, d=1.0, order=2):
        """Create a band-pass butterworth filter with the given pass-band and 
        n-dimensional shape. The 'd' parameter is a scalar or list giving the sample
        spacing in all/each dimension, and the 'order' parameter controls the order
        of the butterworth filter.  
        """
        return SpatialFilter._lowpass_butterworth_nd(high_cutoff, shape, d, order) * \
               SpatialFilter._highpass_butterworth_nd(low_cutoff, shape, d, order)

    @staticmethod
    def _fftfreq_nd(shape, d=1.0):
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

#
##   def filter_nd(array, filter):
##         """Filter an array's fft with a the given filter coefficients."""
##         array, filter = numpy.asarray(array), numpy.asarray(filter)
##         fft = numpy.fft.fftn(array)
##         filtered = fft * filter
##         return numpy.fft.ifftn(filtered)
