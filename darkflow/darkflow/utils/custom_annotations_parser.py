"""
Parse custom 3d bb files.
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob
import pdb
import cv2


def parse_custom_annotations(ANN, pick, exclusive = False):
    """ We expect to be given a filename referencing our 3dbb.txt containing the annotations. """
    dumps = list()

    with open(ANN, 'r') as f:
        for l in f:
            l = l.rstrip()
            tokens = l.split(',')

            # Make sure we have: Path (1), 2DBB (4), 3DBB (16), Class (1)
            assert len(tokens) == 22
            path, bb_2d, bb_3d, name = parse_tokens(tokens)

            # At some point we might want to differentiate different classes
            # if class not in pick:
            #     continue

            # TODO(kolja): At some point this should be in the annotation file itself
            img = cv2.imread(path)
            height, width, channels = img.shape

            dumps += [[path, [width, height, [name + bb_2d + bb_3d]]]]

    pdb.set_trace()

    # Statistics
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in stat:
                stat[current[0]] += 1
            else:
                stat[current[0]] = 1

    print('\nStatistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))

    return dumps



def parse_tokens(tokens):
    path = tokens[0]
    bb_2d = [int(x) for x in tokens[1:5]]
    bb_3d = [int(x) for x in tokens[5:-1]]
    name = tokens[-1]

    return path, bb_2d, bb_3d, name
