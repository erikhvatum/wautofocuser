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
# Authors: Erik Hvatum

import json
from pathlib import Path
from .manually_score import ManualImageScorer
import os
from PyQt5 import Qt

class Experiment01aBfFocusQualityManualScorer(ManualImageScorer):
    def __init__(self, ris_widget, parent=None):
        self._experiment01_a_fpath = Path(os.path.expanduser('~')) / 'data' / 'experiment01_a'
        db = self._load_db()
        super().__init__(ris_widget, db, parent)
        save_button = Qt.QPushButton('Save DB')
        self._ui.buttonBarLayout.insertWidget(0, save_button)
        save_button.clicked.connect(self.save_db)

    def _load_db(self):
        # Intended to be called once, and depends on parent class's __init__ for filling the list view.
        db = {}
        self._db_fpath = self._experiment01_a_fpath / 'bf_focus_quality.json'
        if self.db_fpath.exists():
            with open(str(self._db_fpath), 'r') as f:
                db = json.load(f)
        else:
            with open(str(self._experiment01_a_fpath / 'well_developmental_success_db.json'), 'r') as f:
                wdsdb = json.load(f)
            for well, s in wdsdb.items():
                if s != 'LittleOrNone':
                    well_fpath = self._experiment01_a_fpath / well
                    for bf_fpath in well_fpath.glob('experiment01_a__bf_' + well + '_*.png'):
                        db[str(bf_fpath.relative_to(self._experiment01_a_fpath))] = None
        return db

    def save_db(self):
        with open(str(self._image_dict_path), 'w') as f:
            json.dump(self._db, f)

    @property
    def db_fpath(self):
        return Path(self._db_fpath)

    def _getImage(self, image_rfpath):
        return super()._getImage(self._experiment01_a_fpath / image_rfpath)
