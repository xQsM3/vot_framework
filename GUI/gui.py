# https://github.com/puzzledqs/BBox-Label-Tool
from __future__ import division
from tkinter import *

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,parentparentdir)

import os
import sys
import argparse
import traceback
import logging
import yaml
from datetime import datetime
import colorama

from vot import check_updates, check_debug, __version__
from vot.tracker import Registry, TrackerException
from vot.stack import resolve_stack, list_integrated_stacks
from vot.workspace import Workspace, Cache
from vot.utilities import Progress, normalize_path, ColoredFormatter
from vot.region.shapes import Rectangle

import tkinter.messagebox
from PIL import Image, ImageTk
import os
import glob
import random
import cv2 as cv
import numpy as np
import ntpath
from framework.utilities import cli
from framework.dataset.dataset import Sequence
from framework.utilities.draw import GUIDrawHandle

from threading import Thread
import queue
# colors for the bboxes
COLORS = ['red', 'blue', 'yellow', 'pink', 'cyan', 'green', 'black']
# image sizes for the examples
SIZE = 1710, 1696

class Gui():
    def __init__(self, master):
        
        control_thread = Thread(target=self.run_tracker, daemon=True)
        control_thread.start()
        
        self.imgclass = 0
        self.img = None

        # set up the main frame
        self.parent = master
        self.parent.title("Tracker_custom_test")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)
        
        # set up drawHandle
        self.handle = GUIDrawHandle()
        # initialize global state
        self.imageDir = ''
        self.imageList= []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.hl = None
        self.vl = None
        self.downsample_ratio = 2
        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame, text = "Image Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        self.entry = Entry(self.frame)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
        self.ldBtn = Button(self.frame, text = "Load", command = self.loadDir)
        self.ldBtn.grid(row = 0, column = 2, sticky = W+E)
        
        # run tracker
        self.rtBtn = Button(self.frame,text = "RUN", command = self.run_tracker)
        self.rtBtn.grid(row = 6, column = 2, sticky = W+E)
        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.bind("<Button-1>", self.mouseClickLeft)
        self.mainPanel.bind("<Button-3>",self.mouseClickRight)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("s", self.cancelBBox)
        self.parent.bind("a", self.prevImage) # press 'a' to go backforward
        self.parent.bind("d", self.nextImage) # press 'd' to go forward
        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # showing bbox info & delete bbox
        self.lb1 = Label(self.frame, text = 'Bounding boxes:')
        self.lb1.grid(row = 1, column = 2,  sticky = W+N)
        self.listbox = Listbox(self.frame, width = 22, height = 12)
        self.listbox.grid(row = 2, column = 2, sticky = N)
        self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
        self.btnDel.grid(row = 3, column = 2, sticky = W+E+N)
        self.btnClear = Button(self.frame, text = 'ClearAll', command = self.clearBBox)
        self.btnClear.grid(row = 4, column = 2, sticky = W+E+N)

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 5, column = 1, columnspan = 2, sticky = W+E)
        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.nextBtn = Button(self.ctrPanel, text='Next >>', width = 10, command = self.nextImage)
        self.nextBtn.pack(side = LEFT, padx = 5, pady = 3)
        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
        self.progLabel.pack(side = LEFT, padx = 5)
        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
        self.tmpLabel.pack(side = LEFT, padx = 5)
        self.idxEntry = Entry(self.ctrPanel, width = 5)
        self.idxEntry.pack(side = LEFT)
        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
        self.goBtn.pack(side = LEFT)

        # example pannel for illustration
        self.egPanel = Frame(self.frame, border = 10)
        self.egPanel.grid(row = 1, column = 0, rowspan = 5, sticky = N)
        self.tmpLabel2 = Label(self.egPanel, text = "Examples:")
        self.tmpLabel2.pack(side = TOP, pady = 5)
        self.egLabels = []
        for i in range(3):
            self.egLabels.append(Label(self.egPanel))
            self.egLabels[-1].pack(side = TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)

        # for debugging
##        self.setImage()
##        self.loadDir()
        self.parent.mainloop()
    
    def loadDir(self, dbg = False):

        #self.imageDir = '/media/xqsme/Elements2/Render/YOLO training dataset/images/train/'+str(self.imgclass)
        #os.path.join('forboxing', str(self.imgclass))
        self.imageDir = self.entry.get()
        self.imageList = sorted (glob.glob(self.imageDir+'/*.jpg') )

        
        #self.imageList = glob.glob(os.path.join(self.imageDir, '*.jpg'))
        #self.imageList = sorted(self.imageList)
        if len(self.imageList) == 0:
            print('No .JPG images found in the specified dir!')
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)
        
        '''
         # set up output dir
        self.outDir = os.path.join('/media/xqsme/Elements2/Render/YOLO training dataset/labels/train', str(self.imgclass))
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)
        '''

        filelist = glob.glob(os.path.join(self.imageDir, '*.jpg'))
        self.tmp = []
        self.egList = []
        '''
        #random.shuffle(filelist)
        for (i, f) in enumerate(filelist):
            if i == 3:
                break
            im = Image.open(f)
            r = min(SIZE[0] / im.size[0], SIZE[1] / im.size[1])
            new_size = int(r * im.size[0]), int(r * im.size[1])
            self.tmp.append(im.resize(new_size, Image.ANTIALIAS))
            self.egList.append(ImageTk.PhotoImage(self.tmp[-1]))
            self.egLabels[i].config(image = self.egList[-1], width = SIZE[0], height = SIZE[1])
        '''
        self.loadImage()
        print ('%d video loaded' %(self.total))
    def loadFrame(self,img,region):
        # get frame with region from handle
        self.cur +=1
        self.handle.updateHandle(img,region)
        self.handle.draw_rectangle()
        self.img = self.handle.pilframe
        self.img = self.downSample(self.img)
    def loadImage(self):
        # load image
        
        imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(imagepath)
        self.img = self.downSample(self.img)
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))
        imgw = self.img.size[0]
        imgh = self.img.size[1]        

        # load labels
        self.clearBBox()
        self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        labelname = self.imagename + '.txt'
        day = ntpath.basename( os.path.dirname(imagepath))
        self.labelfilename = os.path.join(self.outDir, day,labelname)
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip().split() #remove \n from string and split integers
                    line = list(map(float,line)) #convert strings to integer
                    #line = [float(w) for w in f.read().split()]
                    if not np.isnan(line).any():
                        x1 = (line[1] - line[3]/2) * imgw
                        y1 = (line[2] - line[4]/2) * imgh
                        x2 = (line[1] + line[3]/2) * imgw
                        y2 = (line[2] + line[4]/2) * imgh
                        tmp = (x1,y1,x2,y2)
                        self.bboxList.append(tuple(tmp))
                        tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], \
                                                                tmp[2], tmp[3], \
                                                                width = 2, \
                                                                outline = COLORS[(len(self.bboxList)-1) % len(COLORS)])
                        self.bboxIdList.append(tmpId)
                        self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(tmp[0], tmp[1], tmp[2], tmp[3]))
                        self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
    def downSample(self,large_img):
        h,w = large_img.size
        ratio = 2
        size = h//self.downsample_ratio,w//self.downsample_ratio
        small_img = large_img.resize(size)
        return small_img
    def saveImage(self):
        pass
    def run_tracker(self):
        #prepare second thread and start tracker in second thread
        self.queue = queue.Queue()
        ThreadedTask(self.queue,self).start()
        self.parent.after(100, self.process_queue())
    def process_queue(self):
        try:
            self.update()
            msg = self.queue.get(0)
        except queue.Empty:
            self.parent.after(100, self.process_queue)
            
    def mouseClickLeft(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)
            self.bboxList.append((x1, y1, x2, y2))
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None
            self.listbox.insert(END, '(%d, %d) -> (%d, %d)' %(x1, y1, x2, y2))
            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        self.STATE['click'] = 1 - self.STATE['click']
    
    def mouseClickRight(self,event):
        ## create a 26*26 bbox if user right clicks
        bbox = event.x-14,event.y-14,event.x+14,event.y+14
        x1,y1,x2,y2 = bbox
        # append bbox to list
        self.bboxList.append(bbox)
        self.bboxIdList.append(self.bboxId)
        self.bboxId = None
        self.listbox.insert(END,'(%d, %d) -> (%d, %d)' %(x1, y1, x2, y2))
        self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
        # draw bbox
        if self.bboxId:
            self.mainPanel.delete(self.bboxId)
        self.bboxId = self.mainPanel.create_rectangle(x1,y1,x2,y2,
                                                      width = 2,outline = 
                                                      COLORS[(len(self.bboxIdList) - 1) % len(COLORS)])
    def mouseMove(self, event):
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2)
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2)
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                            event.x, event.y, \
                                                            width = 2, \
                                                            outline = COLORS[len(self.bboxList) % len(COLORS)])

    def cancelBBox(self, event):
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0

    def delBBox(self):
        sel = self.listbox.curselection()
        if len(sel) != 1 :
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        self.listbox.delete(idx)

    def clearBBox(self):
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []

    def prevImage(self, event = None):
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()

    def nextImage(self, event = None):
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()

    def update(self):
         
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))
        imgw = self.img.size[0]
        imgh = self.img.size[1]

##    def setImage(self, imagepath = r'test2.png'):
##        self.img = Image.open(imagepath)
##        self.tkimg = ImageTk.PhotoImage(self.img)
##        self.mainPanel.config(width = self.tkimg.width())
##        self.mainPanel.config(height = self.tkimg.height())
##        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)

class ThreadedTask(Thread):
    def __init__(self, queue,tool):
        Thread.__init__(self)
        self.queue = queue
        self.tool = tool
    def run(self):
        config = cli.config('AlphaRef')
        
        logger = logging.getLogger("vot")
        stream = logging.StreamHandler()
        stream.setFormatter(ColoredFormatter())
        logger.addHandler(stream)   
        
        logger.setLevel(logging.INFO)
        
        sequence = Sequence(self.tool.imageDir)
        
        sequence.move_pointer(self.tool.cur)
        
        for bbox in self.tool.bboxList:
            # adjust bbox with downsample ration
            bbox = np.asarray(bbox)
            bbox = bbox * self.tool.downsample_ratio
            
            # convert bbox from GUI to VOT definition
            x = bbox[0]
            y = bbox[1]
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox = x,y,w,h
            sequence.assign_start_bbox(bbox)
        
        if config.debug:
            logger.setLevel(logging.DEBUG)

        update, version = check_updates()
        if update:
            logger.warning("A newer version of VOT toolkit is available (%s), please update.", version)
        
        try:    
            cli.do_test_custom(config,logger,sequence,self.tool)
        except:
            pass
if __name__ == '__main__':
    root = Tk()
    tool = Gui(root)
    root.resizable(width =  True, height = True)
    #root.mainloop()
