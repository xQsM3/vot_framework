import json
import glob
import os
from abc import abstractmethod, ABC

from PIL.Image import Image
import numpy as np

from vot import VOTException
from vot.utilities import read_properties
from vot.region import parse

import cv2
from vot.dataset import DatasetException
from vot.dataset import PatternFileListChannel as VOTPatternFileListChannel

class PatternFileListChannel(VOTPatternFileListChannel):
    
    def __init__(self, path, start=1, step=1):
        self.path = os.path.dirname(path)
        super().__init__(path)
    def __scan(self, pattern, start, step):
        extension = os.path.splitext(pattern)[1]
        if not extension in {'.jpg', '.png'}:
            raise DatasetException("Invalid extension in pattern {}".format(pattern))

        i = start
        self._files = []
        
        frameList = sorted (glob.glob(self.path+'/*.jpg') )
        for frame in frameList:
            if not os.path.isfile(frame):
                break
            i = i+ step
            self._files.append(os.path.basename(frame))

        if i <= start:
            raise DatasetException("Empty sequence, no frames found.")
        im = cv2.imread(self.filename(0))
        self._width = im.shape[1]
        self._height = im.shape[0]
        self._depth = im.shape[2]
        