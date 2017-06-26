import numpy as np
import cv2
import csv
import scripts.settings
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import scripts.settings
import matplotlib.image as mpimg


"""
Evaluate the data format of VehicleReID
"""
CHECKED_FRAMES = [
    {"name": "1A", "frames": [1862, 2496, 3016]},
    {"name": "1B", "frames": [1800, 7402, 10300]},
    {"name": "2A", "frames": [1800, 7402, 12278, 12240, 12030]},
    {"name": "2B", "frames": [1862, 4390, 9270, 9476, 9910]},
    {"name": "3A", "frames": [1862, 922, 4896, 9388]},
    {"name": "3B", "frames": [1862, 922, 4896]},
    {"name": "4A", "frames": [806, 4390,8410, 8800,]},
    {"name": "4B", "frames": [806, 4390, 7934, 8166]},
    {"name": "5A", "frames": [804, 4390,17902, 18014]},
    {"name": "5B", "frames": [804, 4390, 14238, 14372]},

]
ANNOTATIONS_FILE = "1B_annotations.txt"
FILE_PREFIX = "1B/1B_"
NUMBER_FORMAT = "%06d"
FILE_SUFFIX = ".bmp"

IS_VEHICLE = True


def read_vehicleReId():
    for shot in CHECKED_FRAMES:
        name = shot['name']
        with open(scripts.settings.SHOTS_FOLDER + name+"_annotations.txt") as file:
            for frame_no in shot['frames']:
                file.seek(0)
                reader = csv.reader(file, delimiter=',')
                cars = []
                for row in reader:
                    if len(row)>2 and int(row[1]) == frame_no:
                        cars.append(row)

                # check frames in a range of -2 to 2 if they fit the boxes
                for test_frame in range(frame_no-2, frame_no+3):
                    frame_path = scripts.settings.SHOTS_FOLDER + name + '/' + name + "_" + NUMBER_FORMAT % test_frame + FILE_SUFFIX
                    print("analyze", name, frame_no,"<->", test_frame, "from:", frame_path)

                    img = cv2.imread(frame_path)
                    if img is None:
                        raise(FileNotFoundError(frame_path))
                    for car in cars:
                        try:
                            (
                                carId, frame,
                                upperPointShort_x, upperPointShort_y,
                                upperPointCorner_x, upperPointCorner_y,
                                upperPointLong_x, upperPointLong_y,
                                crossCorner_x, crossCorner_y,
                                shortSide_x, shortSide_y,
                                corner_x, corner_y,
                                longSide_x, longSide_y,
                                lowerCrossCorner_x, lowerCrossCorner_y
                            ) = car
                        except ValueError:
                            continue

                        img = cv2.circle(img, (int(upperPointShort_x), int(upperPointShort_y)), 5, color=(0,0,255), thickness=2) # red
                        img = cv2.circle(img, (int(upperPointCorner_x), int(upperPointCorner_y)), 5, color=(0,255,255), thickness=2)  # yellow
                        img = cv2.circle(img, (int(upperPointLong_x), int(upperPointLong_y)), 5, color=(255,255,255), thickness=2)  # white
                        img = cv2.circle(img, (int(crossCorner_x), int(crossCorner_y)), 5, color=(255,255,0), thickness=2)  # cyan
                        img = cv2.circle(img, (int(shortSide_x), int(shortSide_y)), 5, color=(255,0,0), thickness=2)  # blue
                        img = cv2.circle(img, (int(corner_x), int(corner_y)), 5, color=(0,0,0), thickness=2)  # black
                        img = cv2.circle(img, (int(longSide_x), int(longSide_y)), 5, color=(0,255,0), thickness=2)  # green
                        img = cv2.circle(img, (int(lowerCrossCorner_x), int(lowerCrossCorner_y)), 5, color=(255,0,255), thickness=2)  # purple

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

                    outfile = "inferred/" + name + "-" +str(frame_no) + "-"+ str(test_frame) + ".bmp"
                    print("write to", outfile)

                    cv2.imwrite(outfile, img)

BB_FILE = "bb_1A.txt"
FRAME_PATHS = ["1B/1B_001804.bmp", "1B/1B_001916.bmp",
               "1B/1B_001926.bmp"]


def read_bbfile():
    with open("../annotations/"+ BB_FILE) as file:
        for frame_path in FRAME_PATHS:

            print("analyze", frame_path)
            file.seek(0)
            reader = csv.reader(file, delimiter=',')

            full_frame_path = scripts.settings.SHOTS_FOLDER + frame_path
            img = cv2.imread(full_frame_path)

            for row in reader:
                try:
                    (
                       path,
                       x1, y1,
                       x2, y2,
                       name
                    ) = row
                except ValueError:
                    print("invalid line", row)
                    continue

                if frame_path != path:
                    continue
                color = (0, 0, 255)
                if name == 'frontBB':
                    color = (255, 255, 0)
                    c = 'r'
                elif name == 'sideBB':
                    color = (0, 255, 0)
                    c= 'b'
                elif name == 'outerBB':
                    color = (255, 0,0)
                    c = 'g'

                img = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), color=color, thickness=5)



                #img = cv2.circle(img, (int(x1), int(y1)), 5, color=color,
                #                 thickness=2)  # red
                #img = cv2.circle(img, (int(x2), int(y2)), 5, color=color,
                #                 thickness=2)  # yellow

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
                #    FRONTBB: yellow_x, red y --> blue_x, black_y
                #    SIDEBB:  ...
                #    OUTERBB:

            #
            plt.show()
            out_path = "inferred/"+frame_path
            print("write to", out_path)
            if not os.path.exists(out_path[:out_path.rfind("/")]):
                print("created folder")
                os.makedirs(out_path[:out_path.rfind("/")])
            #cv2.imshow(frame_path, img)
            #cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(out_path, img)

if IS_VEHICLE:
    read_vehicleReId()
else:
    read_bbfile()