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

import concurrent.futures
import enum
from pathlib import Path
import pickle
from PyQt5 import Qt, uic
import numpy
import os
import skimage.io as skio
for function in ('imread', 'imsave', 'imread_collection'):
    skio.use_plugin('freeimage', function)

class ManualScorer(Qt.QDialog):
    class _ScoreRadioId(enum.IntEnum):
        ClearScore = 1
        SetScore0 = 2
        SetScore1 = 3
        SetScore2 = 4

    _radioIdToScore = {_ScoreRadioId.ClearScore : None,
                       _ScoreRadioId.SetScore0 : 0,
                       _ScoreRadioId.SetScore1 : 1,
                       _ScoreRadioId.SetScore2 : 2}

    def __init__(self, risWidget, dict_, parent):
        super().__init__(parent)

        self._rw = risWidget
        self._db = dict_

        self._ui = uic.loadUiType(os.path.join(os.path.dirname(__file__), 'manually_score.ui'))[0]()
        self._ui.setupUi(self)

        self._scoreRadioGroup = Qt.QButtonGroup(self)
        self._scoreRadioGroup.addButton(self._ui.radioNone, self._ScoreRadioId.ClearScore)
        self._scoreRadioGroup.addButton(self._ui.radio0, self._ScoreRadioId.SetScore0)
        self._scoreRadioGroup.addButton(self._ui.radio1, self._ScoreRadioId.SetScore1)
        self._scoreRadioGroup.addButton(self._ui.radio2, self._ScoreRadioId.SetScore2)

        self._ui.actionUp.triggered.connect(self._ui.prevGroupButton.animateClick)
        self._ui.actionDown.triggered.connect(self._ui.nextGroupButton.animateClick)
        self._ui.actionLeft.triggered.connect(self._ui.prevButton.animateClick)
        self._ui.actionRight.triggered.connect(self._ui.nextButton.animateClick)
        self._ui.actionBackspace.triggered.connect(self._ui.radioNone.animateClick)
        self._ui.action0.triggered.connect(self._ui.radio0.animateClick)
        self._ui.action1.triggered.connect(self._ui.radio1.animateClick)
        self._ui.action2.triggered.connect(self._ui.radio2.animateClick)

        self.addActions([self._ui.actionUp,
                         self._ui.actionDown,
                         self._ui.actionLeft,
                         self._ui.actionRight,
                         self._ui.actionBackspace,
                         self._ui.action0,
                         self._ui.action1,
                         self._ui.action2])

class ManualImageScorer(ManualScorer):
    '''imageDict format: {pathlib.Path object referring to image : score (None if no score assigned)}'''
    _ImageFPathRole = 42
    _Forward = 0
    _Backward = 1

    def __init__(self, risWidget, imageDict, parent=None):
        super().__init__(risWidget, imageDict, parent)

        self.removeAction(self._ui.actionUp)
        self.removeAction(self._ui.actionDown)
        self._ui.actionUp.deleteLater()
        self._ui.actionDown.deleteLater()
        del self._ui.actionUp
        del self._ui.actionDown

        self._ui.prevGroupButton.deleteLater()
        self._ui.nextGroupButton.deleteLater()
        del self._ui.prevGroupButton
        del self._ui.nextGroupButton

        self._ui.tableWidget.setColumnCount(2)

        self._db = imageDict
        self._imageFPaths = sorted(list(self._db.keys()))

        self._ui.tableWidget.setRowCount(len(self._db))
        self._ui.tableWidget.setHorizontalHeaderLabels(['Image', 'Rating'])
        for rowIndex, imageFPath in enumerate(self._imageFPaths):
            imageScore = self._db[imageFPath]
            imageItem = Qt.QTableWidgetItem(str(imageFPath));
            imageItem.setData(self._ImageFPathRole, Qt.QVariant(imageFPath))
            self._ui.tableWidget.setItem(rowIndex, 0, imageItem)
            self._ui.tableWidget.setItem(rowIndex, 1, Qt.QTableWidgetItem('None' if imageScore is None else str(imageScore)))

        self._curImageFPath = None
        self._inRefreshScoreButtons = False

        self._readAheadImageFPath = None
        self._readAheadExecutor = concurrent.futures.ThreadPoolExecutor(2)
        self._readAheadFuture = None

        self._ui.tableWidget.currentItemChanged.connect(self._listWidgetSelectionChange)
        self._scoreRadioGroup.buttonClicked[int].connect(self._scoreButtonClicked)
        self._ui.prevButton.clicked.connect(lambda: self._stepImage(self._Backward))
        self._ui.nextButton.clicked.connect(lambda: self._stepImage(self._Forward))

        self._ui.tableWidget.setCurrentItem(self._ui.tableWidget.item(0, 0))

    def _listWidgetSelectionChange(self, curItem, prevItem):
        imageItem = self._ui.tableWidget.item(curItem.row(), 0)
        self._curImageFPath = imageItem.data(self._ImageFPathRole)
        self._refreshScoreButtons()
        if self._readAheadFuture is not None and self._curImageFPath == self._readAheadImageFPath:
            image = self._readAheadFuture.result()
        else:
            image = self._getImage(self._curImageFPath)
        if image is not None:
            if image.dtype == numpy.float32:
                image = (image * 65535).astype(numpy.uint16)
            self._rw.showImage(image)

    def _refreshScoreButtons(self):
        self._inRefreshScoreButtons = True
        score = self._db[self._curImageFPath]
        if score is None:
            self._ui.radioNone.click()
        elif score is 0:
            self._ui.radio0.click()
        elif score is 1:
            self._ui.radio1.click()
        elif score is 2:
            self._ui.radio2.click()
        else:
            self._inRefreshScoreButtons = False
            raise RuntimeError('Bad value for image score.')
        self._inRefreshScoreButtons = False

    def _scoreButtonClicked(self, radioId):
        if not self._inRefreshScoreButtons:
            self._setScore(self._radioIdToScore[radioId])
            self._ui.nextButton.animateClick()

    def _setScore(self, score):
        '''Set current image score.'''
        if score != self._db[self._curImageFPath]:
            self._db[self._curImageFPath] = score
            self._ui.tableWidget.item(self._ui.tableWidget.currentRow(), 1).setText('None' if score is None else str(score))

    def _stepImage(self, direction):
        curRow = self._ui.tableWidget.currentRow()
        newRow = None
        oneAfter = None
        
        if direction == self._Forward:
            if curRow + 1 < self._ui.tableWidget.rowCount():
                newRow = curRow + 1
            if curRow + 2 < self._ui.tableWidget.rowCount():
                oneAfter = curRow + 2
        elif direction == self._Backward:
            if curRow > 0:
                newRow = curRow - 1
            if curRow - 1 > 0:
                oneAfter = curRow - 2
        
        if newRow is not None:
            self._ui.tableWidget.setCurrentItem(self._ui.tableWidget.item(newRow, 0))

        if oneAfter is not None:
            if self._readAheadFuture is not None and self._readAheadFuture.running():
                concurrent.futures.wait((self._readAheadFuture, ))
            self._readAheadImageFPath = self._ui.tableWidget.item(oneAfter, 0).data(self._ImageFPathRole)
            self._readAheadFuture = self._readAheadExecutor.submit(self._getImage, self._readAheadImageFPath)

    def _getImage(self, imageFPath):
        return skio.imread(str(imageFPath))

    @property
    def imageDict(self):
        return self._db
