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
from pyagg import pyagg
import numpy
from scipy import ndimage
import skimage.io as skio
for function in ('imread', 'imsave', 'imread_collection'):
    skio.use_plugin('freeimage', function)
import sys
from wautofocuser.training_and_testing.image_from_multi_image_statistics import generate_running_percentile_differences
from PyQt5 import Qt

def normalize_intensity(im):
    imn = im.astype(numpy.float32)
    imn *= 255/im.max()
    imn = imn.astype(numpy.uint8)
    return numpy.dstack((imn, imn, imn, imn))

def _run_idx_from_image_fpath(im_fpath):
    return int(im_fpath.stem.split('_')[-2])

def make_it(write_output=True, output_frame_count_limit=None):
    """If write_output is False, an array of the images rendered is returned."""

    background_model_frame_count = 4
    background_model_percentile = 90

    output_dpath = Path('/mnt/scopearray/Zhang_William/lipofuscin_fluorescence/video_temp')
    experiment_dpath = Path('/mnt/scopearray/Zhang_William/lipofuscin_fluorescence')
    do_well_idxs = list(range(36))
    well_dpaths = [experiment_dpath / '{:04}'.format(idx) for idx in do_well_idxs]
    ret = None if write_output else []
    if output_frame_count_limit is not None and output_frame_count_limit <= 0:
        return ret

    crop = Qt.QRectF(Qt.QPointF(0, 0), Qt.QSizeF(2560, 2160))
    image_frame = Qt.QSizeF(16,16)
    frame_outer_gap = Qt.QSizeF(8,8)
    background_c = (0,0,0,255)
    image_background_c = (127,127,127,255)
    title_text_c = (200,200,200,255)
    name_text_c = (0,200,0,255)
    no_image_tex_c = (255,0,0,255)
    text_height = 80
    text_gap = 8
    canvas_size = Qt.QSizeF((crop.width() + image_frame.width()*2)*5 + frame_outer_gap.width()*6,
                            text_height * 2 + text_gap * 3 + crop.height() + image_frame.height()*2 + frame_outer_gap.height()*2)

    canvas = pyagg.Agg2D(numpy.empty((canvas_size.height(), canvas_size.width(), 4), dtype=numpy.uint8))
    canvas.flip_text = True
    canvas.set_font('/usr/share/fonts/dejavu/DejaVuSansMono.ttf', text_height)
    canvas.text_h_alignment = pyagg.TextAlignment.AlignCenter
    canvas.text_v_alignment = pyagg.TextAlignment.AlignCenter
    canvas.line_width = 0
    canvas.set_line_color(0,0,0,0)

    output_frame_idx = 0
    for well_dpath, well_idx in zip(well_dpaths, do_well_idxs):
        bf_fpaths = sorted(well_dpath.glob('lipofuscin_fluorescence__*_bf0.png'))
        bg_subtracter = generate_running_percentile_differences(bf_fpaths, background_model_percentile, background_model_frame_count)
        unskipped_run_count = 0
        for run_idx in range(_run_idx_from_image_fpath(bf_fpaths[-1]) + 1):
            canvas.buffer[...,:] = background_c
            canvas.set_fill_color(*title_text_c)
            title_text_pos = numpy.array((int(canvas_size.width() / 2), text_gap + text_height/2), dtype=numpy.float64)
            bf_fpath = well_dpath / 'lipofuscin_fluorescence__{:04}_{:04}_bf0.png'.format(well_idx, run_idx)
            bf_exists = bf_fpath.exists()
            if bf_exists:
                ts_str = datetime.datetime.fromtimestamp(bf_fpath.stat().st_ctime).isoformat()[:19]
                unskipped_run_count += 1
            else:
                ts_str = 'skipped'
            canvas.text(title_text_pos[0], title_text_pos[1], 'well {} run {:04} ({})'.format(well_idx, run_idx, ts_str))

            canvas.set_fill_color(*image_background_c)
            image_frame_rect = Qt.QRectF(Qt.QPointF(0,0), Qt.QSizeF(crop.width()+2*image_frame.width(), crop.height()+2*image_frame.height()))
            image_frame_rect.translate(0, text_gap*3 + text_height*2)
            image_frame_rect.translate(frame_outer_gap.width(), frame_outer_gap.height())
            title_text_pos[0] = int((image_frame_rect.right() + image_frame_rect.left())/2)
            title_text_pos[1] += text_height + text_gap
            canvas.set_fill_color(*name_text_c)
            canvas.text(title_text_pos[0], title_text_pos[1], 'bf0')
            if bf_exists:
                bf = skio.imread(str(bf_fpath))
                image_rect = Qt.QRectF(crop)
                image_rect.translate(image_frame_rect.left() + image_frame.width(), image_frame_rect.top() + image_frame.height())
                image_rect.translate(-crop.left(), -crop.top())
                canvas.rectangle(image_frame_rect.left(), image_frame_rect.top(), image_frame_rect.right(), image_frame_rect.bottom())
                canvas.buffer[image_rect.top():image_rect.bottom(),
                              image_rect.left():image_rect.right(), :] = normalize_intensity(bf[crop.top():crop.bottom(), crop.left():crop.right()])
            else:
                no_image_tex_pos = title_text_pos.copy()
                no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                canvas.set_fill_color(*no_image_tex_c)
                canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'skipped')

            canvas.set_fill_color(*image_background_c)
            image_frame_rect.translate(image_frame.width() * 2 + crop.width() + frame_outer_gap.width(), 0)
            canvas.rectangle(image_frame_rect.left(), image_frame_rect.top(), image_frame_rect.right(), image_frame_rect.bottom())
            title_text_pos[0] = int((image_frame_rect.right() + image_frame_rect.left())/2)
            canvas.set_fill_color(*name_text_c)
            canvas.text(title_text_pos[0], title_text_pos[1], 'abs(bf0 - background_model)')
            if bf_exists:
                image_rect = Qt.QRectF(crop)
                image_rect.translate(image_frame_rect.left() + image_frame.width(), image_frame_rect.top() + image_frame.height())
                image_rect.translate(-crop.left(), -crop.top())
                bg_delta = bg_subtracter.__next__()
                if bg_delta is None:
                    no_image_tex_pos = title_text_pos.copy()
                    no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                    canvas.set_fill_color(*no_image_tex_c)
                    canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'building model... ({}/{})'.format(unskipped_run_count, background_model_frame_count))
                else:
                    canvas.buffer[image_rect.top():image_rect.bottom(),
                                  image_rect.left():image_rect.right(), :] = normalize_intensity(bg_delta[crop.top():crop.bottom(), crop.left():crop.right()])
            else:
                no_image_tex_pos = title_text_pos.copy()
                no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                canvas.set_fill_color(*no_image_tex_c)
                canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'skipped')

            canvas.set_fill_color(*image_background_c)
            image_frame_rect.translate(image_frame.width() * 2 + crop.width() + frame_outer_gap.width(), 0)
            canvas.rectangle(image_frame_rect.left(), image_frame_rect.top(), image_frame_rect.right(), image_frame_rect.bottom())
            title_text_pos[0] = int((image_frame_rect.right() + image_frame_rect.left())/2)
            canvas.set_fill_color(*name_text_c)
            canvas.text(title_text_pos[0], title_text_pos[1], 'green/yellow')
            if bf_exists:
                fluo_fpath = well_dpath / 'lipofuscin_fluorescence__{:04}_{:04}_greenyellow.png'.format(well_idx, run_idx)
                if fluo_fpath.exists():
                    fluo = skio.imread(str(fluo_fpath))
                    image_rect = Qt.QRectF(crop)
                    image_rect.translate(image_frame_rect.left() + image_frame.width(), image_frame_rect.top() + image_frame.height())
                    image_rect.translate(-crop.left(), -crop.top())
                    canvas.buffer[image_rect.top():image_rect.bottom(),
                                  image_rect.left():image_rect.right(), :] = normalize_intensity(fluo[crop.top():crop.bottom(), crop.left():crop.right()])
                else:
                    no_image_tex_pos = title_text_pos.copy()
                    no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                    canvas.set_fill_color(*no_image_tex_c)
                    canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'file not found')
            else:
                no_image_tex_pos = title_text_pos.copy()
                no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                canvas.set_fill_color(*no_image_tex_c)
                canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'skipped')

            canvas.set_fill_color(*image_background_c)
            image_frame_rect.translate(image_frame.width() * 2 + crop.width() + frame_outer_gap.width(), 0)
            canvas.rectangle(image_frame_rect.left(), image_frame_rect.top(), image_frame_rect.right(), image_frame_rect.bottom())
            title_text_pos[0] = int((image_frame_rect.right() + image_frame_rect.left())/2)
            canvas.set_fill_color(*name_text_c)
            canvas.text(title_text_pos[0], title_text_pos[1], 'cyan')
            if bf_exists:
                fluo_fpath = well_dpath / 'lipofuscin_fluorescence__{:04}_{:04}_cyan.png'.format(well_idx, run_idx)
                if fluo_fpath.exists():
                    fluo = skio.imread(str(fluo_fpath))
                    image_rect = Qt.QRectF(crop)
                    image_rect.translate(image_frame_rect.left() + image_frame.width(), image_frame_rect.top() + image_frame.height())
                    image_rect.translate(-crop.left(), -crop.top())
                    canvas.buffer[image_rect.top():image_rect.bottom(),
                                  image_rect.left():image_rect.right(), :] = normalize_intensity(fluo[crop.top():crop.bottom(), crop.left():crop.right()])
                else:
                    no_image_tex_pos = title_text_pos.copy()
                    no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                    canvas.set_fill_color(*no_image_tex_c)
                    canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'file not found')
            else:
                no_image_tex_pos = title_text_pos.copy()
                no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                canvas.set_fill_color(*no_image_tex_c)
                canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'skipped')

            canvas.set_fill_color(*image_background_c)
            image_frame_rect.translate(image_frame.width() * 2 + crop.width() + frame_outer_gap.width(), 0)
            canvas.rectangle(image_frame_rect.left(), image_frame_rect.top(), image_frame_rect.right(), image_frame_rect.bottom())
            title_text_pos[0] = int((image_frame_rect.right() + image_frame_rect.left())/2)
            canvas.set_fill_color(*name_text_c)
            canvas.text(title_text_pos[0], title_text_pos[1], 'uv')
            if bf_exists:
                fluo_fpath = well_dpath / 'lipofuscin_fluorescence__{:04}_{:04}_UV.png'.format(well_idx, run_idx)
                if fluo_fpath.exists():
                    fluo = skio.imread(str(fluo_fpath))
                    image_rect = Qt.QRectF(crop)
                    image_rect.translate(image_frame_rect.left() + image_frame.width(), image_frame_rect.top() + image_frame.height())
                    image_rect.translate(-crop.left(), -crop.top())
                    canvas.buffer[image_rect.top():image_rect.bottom(),
                                  image_rect.left():image_rect.right(), :] = normalize_intensity(fluo[crop.top():crop.bottom(), crop.left():crop.right()])
                else:
                    no_image_tex_pos = title_text_pos.copy()
                    no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                    canvas.set_fill_color(*no_image_tex_c)
                    canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'file not found')
            else:
                no_image_tex_pos = title_text_pos.copy()
                no_image_tex_pos[1] = int((image_rect.top() + image_rect.bottom())/2)
                canvas.set_fill_color(*no_image_tex_c)
                canvas.text(no_image_tex_pos[0], no_image_tex_pos[1], 'skipped')

            out_image = ndimage.zoom(canvas.buffer[...,:3], (0.5, 0.5, 1), order=1)
#           out_image = canvas.buffer[...,:3].copy()
            if write_output:
                out_fpath = output_dpath / '{:06}.bmp'.format(output_frame_idx)
                skio.imsave(str(out_fpath), out_image)
            else:
                ret.append(out_image)
            output_frame_idx += 1
            if output_frame_count_limit is not None and output_frame_idx >= output_frame_count_limit:
                return ret
    return ret
