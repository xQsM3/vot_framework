import glob
import cv2 as cv
import os
from vot.dataset import VOTSequence
from dataset import PatternFileListChannel
import six
from vot.region.shapes import Rectangle

def load_channel(source):
    return PatternFileListChannel(source)

class Sequence(VOTSequence):
    def __init__(self,base):
        self.frame_paths = sorted (glob.glob(base+'/*.jpg') )
        self.start_bbox = {}
        self.length = len(self.frame_paths)
        self.pointer = 0
        super().__init__(base, None)
    
    def assign_start_bbox(self,bbox):
        #create vot region
        region = Rectangle(bbox[0],bbox[1],bbox[2],bbox[3])
        #left,top,width,height x,y,w,h
        self.start_bbox = {self.pointer: region}    
    
    def imread_frame(self,index):
        if index < 0 or index >= self.length:
            return None
        frame = cv.imread(self.frame_paths[index])
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    def length(self):
        return len(frame_paths)

    def move_pointer(self,new_position):
        self.pointer = new_position
    
    # overwrite the _read function of parent Class VOTSequence
    
    def _read(self):
        
        
        channels = {}
        tags = {}
        values = {}
        groundtruth = []

        for c in ["color", "depth", "ir"]:
            channel_path = self.metadata("channels.%s" % c, None)
            if not channel_path is None:
                channels[c] = load_channel(os.path.join(self._base, localize_path(channel_path)))
        # Load default channel if no explicit channel data available
        if len(channels) == 0:
            channels["color"] = load_channel(os.path.join(self._base, "*.jpg"))
        else:
            self._metadata["channel.default"] = next(iter(channels.keys()))
        
        self._metadata["width"], self._metadata["height"] = six.next(six.itervalues(channels)).size

        return channels, groundtruth, tags, values
        