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

            if DATA_FORMAT == "3d_reg":
                    ################################ STRUCUTRE ########################################################
                    # 0"filepath": frame_path_variable,
                    # 1"x1": min(x_points),        2"y1": min(y_points),
                    # 3"x2": max(x_points),        4"y2": max(y_points),
                    # 5"top_front_right_x": upperPointShort_x if facing_left else upperPointCorner_x,
                    # 6"top_front_right_y": upperPointShort_y - cutter if facing_left else upperPointCorner_y - cutter,
                    # 7"top_front_left_x": upperPointCorner_x if facing_left else upperPointShort_x,
                    # 8"top_front_left_y": upperPointCorner_y - cutter if facing_left else upperPointShort_y - cutter,
                    # 9"top_back_left_x": crossCorner_x if facing_left else upperPointLong_x,
                    # 10"top_back_left_y": crossCorner_y - cutter if facing_left else upperPointLong_y  - cutter,
                    # 11"top_back_right_x": upperPointLong_x if facing_left else crossCorner_x,
                    # 12"top_back_right_y": upperPointLong_y  - cutter if facing_left else crossCorner_y  - cutter,
                    #
                    # 13"bot_front_right_x": shortSide_x if facing_left else corner_x,
                    # 14"bot_front_right_y": shortSide_y - cutter if facing_left else corner_y - cutter,
                    # 15"bot_front_left_x": corner_x if facing_left else shortSide_x,
                    # 16"bot_front_left_y": corner_y - cutter if facing_left else shortSide_y - cutter,
                    # 17"bot_back_left_x": lowerCrossCorner_x if facing_left else longSide_x,
                    # 18"bot_back_left_y": lowerCrossCorner_y - cutter if facing_left else longSide_y  - cutter,
                    # 19"bot_back_right_x": longSide_x if facing_left else lowerCrossCorner_x,
                    # 20"bot_back_right_y": longSide_y - cutter if facing_left else lowerCrossCorner_y  - cutter,
                    # 21"class_name": "3DBB"
                    # })
                    ################################################################################################

                    # use the following colors:
                    # (carId, frame,
                    #  upperPointShort_x, upperPointShort_y,      #red
                    #  upperPointCorner_x, upperPointCorner_y,    #yellow
                    #  upperPointLong_x, upperPointLong_y,        #white
                    #  crossCorner_x, crossCorner_y,              #cyan
                    #  shortSide_x, shortSide_y,                  #blue
                    #  corner_x, corner_y,                        #black
                    #  longSide_x, longSide_y,                    #green
                    #  lowerCrossCorner_x, lowerCrossCorner_y     #purple
                    #  )
                    COLORS = [(0, 0, 255),  # red           0
                              (0, 255, 255),  # yellow      1
                              (255, 255, 255),  # white     2
                              (255, 255, 0),  # cyan        3
                              (255, 0, 0),  # blue          4
                              (0, 0, 0),  # black           5
                              (0, 255, 0),  # green         6
                              (255, 0, 255),  # purple      7
                              ]
                    #################### MEANING OF THE COLORS/ANNOTATIONS ##########################
                    # The cars are always in this angle
                    #
                    #       (0,0) >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  (x,0)
                    #         v
                    #         v        cyan  ~~~~~~~~~~
                    #         v        /                ~~~~~~~~~~ red
                    #         v   white  ~~~~~~~~~~               /  |
                    #         v     |              ~~~~~~~~~~ yellow |
                    #         v     |                           |    |
                    #         v     |                           |    |
                    #         v     |  purple                   |    |
                    #         v     | /                         |   blue
                    #         v   green  ~~~~~~~~~              |   /
                    #         v                    ~~~~~~~~~~~~black
                    #       (0,y)
                    #
                    for i in range(8):
                        img = cv2.circle(img, (int(entry[5+i*2]), int(entry[5+i*2+1])), 5, color=COLORS[i], thickness=2)
                    cv2.rectangle(img, (int(entry[1]), int(entry[2])),
                                  (int(entry[3]), int(entry[4])), (0, 0, 0), 2)

            else:
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

                img = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color=color, thickness=5)


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