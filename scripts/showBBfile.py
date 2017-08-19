from __future__ import print_function
import numpy as np
import cv2
import csv
import os
import shutil
from optparse import OptionParser
import scripts.helper
import scripts.settings

FRAME_PATHS = [] # these paths are tested, additionally to some random ones, specifed by Random_samples
RANDOM_SAMPLES = 5
DATA_FORMAT = "3d_reg" # in ["3d_reg", "merge_areas"]

parser = OptionParser()

parser.add_option("-p", "--bb_path", dest="bb_path", help="Name of bounding box file that shall be graphically displayed")
(options, args) = parser.parse_args()

if options.bb_path:
    bb_file_path = options.bb_file
else:
    bb_file_path = scripts.helper.chose_from_folder("../annotations/", "--bb_path")
bb_file_path = os.path.normpath(bb_file_path)
bb_file_name = os.path.splitext(os.path.basename(bb_file_path))[0]
bb_file_folder_path = os.path.join(*(bb_file_path.split(os.path.sep)[:-1]))

out_folder = os.path.join(bb_file_folder_path, bb_file_name) + "/"


if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
    os.makedirs(out_folder)

def read_anno_file():
    with open(bb_file_path) as file:
        print("Reading in the whole annotation file", bb_file_path, "as ")
        entries = list(csv.reader(file, delimiter=','))

    sample_indexes = np.random.choice(len(entries), RANDOM_SAMPLES, replace=False)
    FRAME_PATHS.extend([entries[idx][0] for idx in sample_indexes])

    for frame_path in FRAME_PATHS:

        print("analyze", frame_path) #we only have one frame yet, we want all bb annotations that belong to that frame
        frame_entries = []
        for entry in entries:
            # first element is always the path
            if entry[0] == frame_path:
                frame_entries.append(entry)

        full_frame_path = scripts.settings.variable_path_to_abs(frame_path)
        frame_name_base = os.path.basename(full_frame_path)
        img = cv2.imread(full_frame_path)

        for entry in frame_entries:

            img = scripts.helper.draw_annotations(img, entry, DATA_FORMAT)

        out_path = os.path.join(out_folder,frame_name_base)
        print("write to", out_path)
        if not os.path.exists(out_path[:out_path.rfind("/")]):
            print("created folder", out_path)
            os.makedirs(out_path[:out_path.rfind("/")])
        #cv2.imshow(frame_path, img)
        #cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(out_path, img)

read_anno_file()