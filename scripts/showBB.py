import numpy as np
import cv2
import csv

"""
Evaluate the data format of VehicleReID
"""

FRAME_NOs = [2056, 1822, 1844, 2166, 2230, 2338]
ANNOTATIONS_FILE = "1A_annotations.txt"



with open(ANNOTATIONS_FILE) as file:
    for frame_no in FRAME_NOs:
        file.seek(0)
        reader = csv.reader(file, delimiter=',')
        #print(reader)
        frame_path = "1A__" + "%05d" % frame_no + ".png"
        img = cv2.imread(frame_path)

        for row in reader:
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
                ) = row
                print(frame)
            except ValueError:
                print("invalid line", row)
                continue

            if int(frame) != frame_no:
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







        #
        cv2.imshow(frame_path, img)
        cv2.waitKey(0)
        #cv2.destroyAllWindows()