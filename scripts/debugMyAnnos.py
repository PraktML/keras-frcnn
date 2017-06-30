from __future__ import print_function
import numpy as np
import cv2
import csv
import scripts.settings
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import scripts.settings
import matplotlib.image as mpimg


ANNO = "bb_1-4A"

ANNO_FILE  = "../annotations/"+ANNO+".txt"
OUT_FOLDER = "../annotations/"+ANNO+"/"
FRAME_PATHS = [
]
RANDOM_SAMPLES = 5

if os.path.exists(OUT_FOLDER):
    shutil.rmtree(OUT_FOLDER)
    os.makedirs(OUT_FOLDER)

def read_anno_file():
    with open(ANNO_FILE) as file:
        print("Reading in the whole annotation file", ANNO_FILE, "as ")
        entries = list(csv.reader(file, delimiter=','))

    sample_indexes = np.random.choice(len(entries), RANDOM_SAMPLES, replace=False)
    FRAME_PATHS.extend([entries[idx][0] for idx in sample_indexes])

    for frame_path in FRAME_PATHS:

        print("analyze", frame_path)
        frame_entries = []
        for entry in entries:
            # entry structure (path, x1, y1, x2, y2, classname)
            if entry[0] == frame_path:
                frame_entries.append(entry)

        full_frame_path = scripts.settings.PLATTE_BASEPATH + frame_path
        img = cv2.imread(full_frame_path)

        for entry in frame_entries:
            (_, x1, y1, x2, y2, classname) = entry
            if classname == 'front':
                color = (255, 255, 0)
                c = 'c'
            elif classname == 'back':
                color = (255, 255, 255)
                c= 'w'
            elif classname == 'side':
                color = (0, 255, 0)
                c= 'b'
            elif classname == 'outer':
                color = (255, 0,0)
                c = 'g'
            elif classname == 'top':
                color = (255, 0,225)
                c = 'm'

            img = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color=color, thickness=3)


            #################### MEANING OF THE COLORS/ANNOTATIONS ##########################
            # The cars are always in this angle
            #
            #       (0,0) >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  (x,0)
            #         v
            #         v     O   x  ~~~~~~~~~~~~              O
            #         v        /                ~~~~~~~~F~~~ F
            #         v     S ~~~~~~~~~~~~              S /f |
            #         v     |              ~~~~~~~~~~~~ x  r |
            #         v     |                           |  o |
            #         v     |           side            |  n |
            #         v     |   x                       |  t |
            #         v     | /                         |    x
            #         v     x ~~~~~~~~~~~~              |  /
            #         v     OS             ~~~~~~~~~~~~FS    OF
            #       (0,y)
            #
            #    In my first step it only seems necessary to teach the net to some sides of this cube
            #    FRONTBB: four point marked "F"
            #    SIDEBB:  four points marked "S"
            #    OUTERBB: four points marked "O"


        out_path = OUT_FOLDER + frame_path
        print("write to", out_path)
        if not os.path.exists(out_path[:out_path.rfind("/")]):
            print("created folder")
            os.makedirs(out_path[:out_path.rfind("/")])
        #cv2.imshow(frame_path, img)
        #cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(out_path, img)

read_anno_file()