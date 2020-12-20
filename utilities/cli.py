import os
import sys
import argparse
import traceback
import logging
import yaml
from datetime import datetime
import colorama
from os.path import dirname as up

from argparse import Namespace
from framework.dataset.dataset import Sequence
from vot import check_updates, check_debug, __version__
from vot.tracker import Registry, TrackerException
from vot.stack import resolve_stack, list_integrated_stacks
from vot.workspace import Workspace, Cache
from vot.utilities import Progress, normalize_path, ColoredFormatter

def do_test_custom(config,logger,sequence):
    
    trackers = Registry(config.registry)
    if not config.tracker:
        logger.error("Unable to continue without a tracker")
        logger.error("List of found trackers: ")
        for k in trackers.identifiers():
            logger.error(" * %s", k)
        return

    if not config.tracker in trackers:
        logger.error("Tracker does not exist")
        return

    tracker = trackers[config.tracker]
    
    logger.info("Obtaining runtime for tracker %s", tracker.identifier)
    '''
    if config.visualize:
        import matplotlib.pylab as plt
        from vot.utilities.draw import MatplotlibDrawHandle
        figure = plt.figure()
        figure.canvas.set_window_title('VOT Test')
        axes = figure.add_subplot(1, 1, 1)
        axes.set_aspect("equal")
        handle = MatplotlibDrawHandle(axes, size=sequence.size)
        handle.style(fill=False)
        figure.show()
    '''
    runtime = None

    try:

        runtime = tracker.runtime(log=True)
        
        region, _, _ = runtime.initialize(sequence.frame(sequence.pointer), sequence.start_bbox[sequence.pointer])

        if config.visualize:
            pass
            '''
            axes.clear()
            handle.image(sequence.frame(0).channel())
            handle.style(color="green").region(sequence.frame(0).groundtruth())
            handle.style(color="red").region(region)
            figure.canvas.draw()
            '''
        sequence.move_pointer = sequence.pointer +1
        for i in range(sequence.pointer, sequence.length):
            logger.info("Updating on frame %d/%d", i, sequence.length-1)
            region, _, _ = runtime.update(sequence.frame(i))

            if config.visualize:
                pass
                '''
                axes.clear()
                handle.image(sequence.frame(i).channel())
                handle.style(color="green").region(sequence.frame(i).groundtruth())
                handle.style(color="red").region(region)
                figure.canvas.draw()
                '''
            sequence.move_pointer = sequence.pointer +1
            
        logger.info("Stopping tracker")

        runtime.stop()

        logger.info("Test concluded successfuly")

    except TrackerException as te:
        logger.error("Error during tracker execution: {}".format(te))
        if runtime:
            runtime.stop()
    except KeyboardInterrupt:
        if runtime:
            runtime.stop()
def config(tracker,debug=False,registry=[up(up(up(os.path.realpath(__file__))))],sequence=None,visualize=True):
    config = Namespace(action='test', debug=debug, registry=registry, sequence=sequence, tracker=tracker, visualize=visualize)
    return config