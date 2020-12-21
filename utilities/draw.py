from typing import Tuple, List, Union

from matplotlib import colors
from matplotlib.patches import Polygon
from PIL import Image, ImageDraw
import numpy as np
import cv2 as cv

from vot.region import RegionType
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
    
class GUIDrawHandle():
    def __init__(self,frame=None,region=None):
        self.frame = frame
        self.region = region
    def updateHandle(self,frame,region):
        self.frame = frame
        self.region = region
    def draw_rectangle(self):
        self.region = self.region.convert(RegionType.RECTANGLE)
        p1 = self.region.x,self.region.y
        p2 = round(self.region.x+self.region.width),round(self.region.y+self.region.height)
        print('P1 {}'.format(p1))
        print('P2 {}'.format(p2))
        cv.rectangle(self.frame,p1,p2,(255,0,0),3)
        self.pilframe = Image.fromarray(self.frame)