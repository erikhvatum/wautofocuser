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

import datetime
from pathlib import Path
import pyagg
import numpy
from scipy import ndimage
import skimage.io as skio
for function in ('imread', 'imsave', 'imread_collection'):
    skio.use_plugin('freeimage', function)
import sys
from wautofocuser.training_and_testing.image_from_multi_image_statistics import generate_running_percentile_differences
from PyQt5 import Qt

def make_it(write_output=True, output_frame_count_limit=None):
    """If write_output is False, an array of the images rendered is returned."""
    output_dpath = Path('/home/ehvatum/oot')
    experiment_dpath = Path('/mnt/scopearray/life_assay_pilot')
    do_well_idxs = [0,1,2,3,4,6]
    well_dpaths = [experiment_dpath / '{:04}'.format(idx) for idx in do_well_idxs]
    ret = None if write_output else []
    if output_frame_count_limit is not None and output_frame_count_limit <= 0:
        return ret

    crop = Qt.QRectF(Qt.QPointF(0, 175), Qt.QSizeF(2160, 2225))
    image_frame = Qt.QSizeF(16,16)
    frame_outer_gap = Qt.QSizeF(8,8)
    background_c = (0,0,0,255)
    image_background_c = (127,127,127,255)
    title_text_c = (0,200,0,255)
    data_accruing_text_c = (127,0,0,255)
    text_height = 80
    text_gap = 8
    canvas_size = Qt.QSizeF((crop.width() + image_frame.width()*2 + frame_outer_gap.width()*2) * 3,
                            (text_height + text_gap*2)*2 + crop.height() + image_frame.height()*2 + frame_outer_gap.height()*2)

    canvas = pyagg.Agg2D(numpy.empty((canvas_size.height(), canvas_size.width(), 4), dtype=numpy.uint8))
    canvas.flip_text = True
    canvas.set_font('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', text_height)
    canvas.text_h_alignment = pyagg.TextAlignment.AlignCenter
    canvas.text_v_alignment = pyagg.TextAlignment.AlignCenter
    canvas.line_width = 0
    canvas.set_line_color(0,0,0,0)

    output_frame_idx = 0
    for well_dpath, well_idx in zip(well_dpaths, do_well_idxs):
        bf_fpaths = sorted(well_dpath.glob('life_assay_pilot__*_bf.tiff'))
        for bf_fpath in bf_fpaths:
            run_idx = int(bf_fpath.stem.split('_')[-2])
            canvas.buffer[...,:] = background_c
            canvas.set_fill_color(*title_text_c)
            ts_str = datetime.datetime.fromtimestamp(bf_fpath.stat().st_ctime).isoformat()[:19]
            canvas.text(int(canvas_size.width() / 2), text_gap + text_height/2, 'well {} run {:04} ({})'.format(well_idx, run_idx, ts_str))
            out_image = ndimage.zoom(canvas.buffer[...,:3], (0.5, 0.5, 1), order=1)
            if write_output:
                pass
            else:
                ret.append(out_image)
            output_frame_idx += 1
            if output_frame_count_limit is not None and output_frame_idx >= output_frame_count_limit:
                return ret
    return ret
