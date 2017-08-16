import scripts.settings

import json
import numpy as np
import shutil
import os
import cv2
#SHOTS = ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B"]
OUT_FOLDER = scripts.settings.BOXCARS116K_PATH + "visualize/"
if os.path.exists(OUT_FOLDER):
    shutil.rmtree(OUT_FOLDER)
    os.makedirs(OUT_FOLDER)

RANDOM_SAMPLES = 20

NUMBER_FORMAT = "%06d"
FILE_SUFFIX = ".bmp"

IS_VEHICLE = True

COLORS = [(0, 0, 255),      # red
          (0, 255, 255),    # yellow
          (255, 255, 255),  # white
          (255, 255, 0),    # cyan
          (255, 0, 0),      # blue
          (0, 0, 0),        # black
          (0, 255, 0),      # green
          (255, 0, 255),    # purple
        ]



with open(scripts.settings.BOXCARS116K_JSON_FILE) as jsonfile:
    data = json.load(jsonfile)["samples"]

instances = []

for car in data:
    for instance in car["instances"]:
        # print(car['annotation'], ":", instance)
        instances.append((instance["path"], instance["3DBB"], instance["3DBB_offset"], instance["2DBB"]))

samples_idx = np.random.choice(len(instances), RANDOM_SAMPLES, replace=True)
samples = [instances[idx] for idx in samples_idx]

for sample in samples:

            frame_path = scripts.settings.BOXCARS116K_PATH +  sample[0]

            print("analyze", frame_path)
            img = cv2.imread(frame_path)

            if img is None:
                print("THIS FILE DOESN'T EXIST:", frame_path)
                continue

            for i, point in enumerate(sample[1]):

                img = cv2.circle(img, (int(point[0])-int(sample[2][0]), int(point[1])-int(sample[2][1])), 5, COLORS[i],
                                 thickness=2)


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
                #    In my first step it only seems necessary to teach the net to some sides of this cube
                #

            out_path = OUT_FOLDER + sample[0]
            print("write to", out_path)
            if not os.path.exists(out_path[:out_path.rfind("/")]):
                print("created folder")
                os.makedirs(out_path[:out_path.rfind("/")])

            cv2.imwrite(out_path, img)


