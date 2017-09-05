import numpy as np
import cv2
import csv
import scripts.settings
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#from PIL import Image
import scripts.settings
import matplotlib.image as mpimg
import shutil
"""
Evaluate the data format of VehicleReID
"""

#SHOTS = ["1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B"]
OUT_FOLDER = scripts.settings.VRI_SHOTS_PATH + "visualize/"
if os.path.exists(OUT_FOLDER):
    shutil.rmtree(OUT_FOLDER)
    os.makedirs(OUT_FOLDER)

RANDOM_SAMPLES = 20


NUMBER_FORMAT = "%06d"
FILE_SUFFIX = ".bmp"

IS_VEHICLE = True


def read_vehReID_random():
    for shot in scripts.settings.FRAMES_VRI:
        print("analzye shot", shot)
        name = shot['name']
        offset = shot['offset']
        with open(scripts.settings.VRI_SHOTS_PATH  + name + "_annotations.txt") as file:
            entries = list(csv.reader(file, delimiter=','))

        sample_frame_indexes = np.random.choice(range(1,len(entries)), RANDOM_SAMPLES, replace=True)
        sample_frames = shot['frames'] + [int(entries[idx][1]) for idx in sample_frame_indexes]



        for sample in sample_frames:
            frame_path = scripts.settings.VRI_SHOTS_PATH  + name + '/' + name + "_" + NUMBER_FORMAT % (sample + offset) + FILE_SUFFIX
            print("analyze", name, sample, "from:", frame_path)

            entries_for_sample = []
            img = cv2.imread(frame_path)
            if img is None:
                print("THIS FILE DOESN'T EXIST:", frame_path)
                continue

            for entry in entries:
                if len(entry) > 1 and int(entry[1]) == sample:
                    entries_for_sample.append(entry)

            for entry in entries_for_sample:
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
                    ) = entry
                except ValueError:
                    continue

                points = [
                    (upperPointShort_x, upperPointShort_y),
                    (upperPointCorner_x, upperPointCorner_y),
                    (upperPointLong_x, upperPointLong_y),
                    (crossCorner_x, crossCorner_y),
                    (shortSide_x, shortSide_y),
                    (corner_x, corner_y),
                    (longSide_x, longSide_y),
                    (lowerCrossCorner_x, lowerCrossCorner_y)
                ]

                COLORS = [(0, 0, 255),  # red
                          (0, 255, 255),  # yellow
                          (255, 255, 255),  # white
                          (255, 255, 0),  # cyan
                          (255, 0, 0),  # blue
                          (0, 0, 0),  # black
                          (0, 255, 0),  # green
                          (255, 0, 255),  # purple
                          ]

                for i in range(8):
                    img = cv2.circle(img, (int(points[i][0]), int(points[i][1])), 5, color=COLORS[i],
                                 thickness=2)
                    img = cv2.putText(img, str(i),
                                      (int(points[i][0]), int(points[i][1])),
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      1.5, (0, 0, 0), 4)

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
            cv2.line(img, (0, int(shot["sep_y"])), (2000, int(shot["sep_m"] * 2000 + shot["sep_y"])), (0,0,255), 4)

            out_path = OUT_FOLDER + name + "/" + str(sample) + "_" + str(offset) + ".bmp"
            print("write to", out_path)
            if not os.path.exists(out_path[:out_path.rfind("/")]):
                print("created folder")
                os.makedirs(out_path[:out_path.rfind("/")])

            cv2.imwrite(out_path, img)


#
# def read_vehicleReId():
#     for shot in scripts.settings.FRAMES_VRI:
#         name = shot['name']
#         with open(scripts.settings.VRI_SHOTS_PATH  + name+"_annotations.txt") as file:
#             for frame_no in shot['frames']:
#                 file.seek(0)
#                 reader = csv.reader(file, delimiter=',')
#                 cars = []
#                 for row in reader:
#                     if len(row)>2 and int(row[1]) == frame_no:
#                         cars.append(row)
#
#                 # check frames in a range of -2 to 2 if they fit the boxes
#                 for test_frame in range(frame_no-2, frame_no+3):
#                     frame_path = scripts.settings.VRI_SHOTS_PATH  + name + '/' + name + "_" + NUMBER_FORMAT % test_frame + FILE_SUFFIX
#                     print("analyze", name, frame_no,"<->", test_frame, "from:", frame_path)
#
#                     img = cv2.imread(frame_path)
#                     if img is None:
#                         raise(FileNotFoundError(frame_path))
#                     for car in cars:
#                         try:
#                             (
#                                 carId, frame,
#                                 upperPointShort_x, upperPointShort_y,
#                                 upperPointCorner_x, upperPointCorner_y,
#                                 upperPointLong_x, upperPointLong_y,
#                                 crossCorner_x, crossCorner_y,
#                                 shortSide_x, shortSide_y,
#                                 corner_x, corner_y,
#                                 longSide_x, longSide_y,
#                                 lowerCrossCorner_x, lowerCrossCorner_y
#                             ) = car
#                         except ValueError:
#                             continue
#
#                         img = cv2.circle(img, (int(upperPointShort_x), int(upperPointShort_y)), 5, color=(0,0,255), thickness=2) # red
#                         img = cv2.circle(img, (int(upperPointCorner_x), int(upperPointCorner_y)), 5, color=(0,255,255), thickness=2)  # yellow
#                         img = cv2.circle(img, (int(upperPointLong_x), int(upperPointLong_y)), 5, color=(255,255,255), thickness=2)  # white
#                         img = cv2.circle(img, (int(crossCorner_x), int(crossCorner_y)), 5, color=(255,255,0), thickness=2)  # cyan
#                         img = cv2.circle(img, (int(shortSide_x), int(shortSide_y)), 5, color=(255,0,0), thickness=2)  # blue
#                         img = cv2.circle(img, (int(corner_x), int(corner_y)), 5, color=(0,0,0), thickness=2)  # black
#                         img = cv2.circle(img, (int(longSide_x), int(longSide_y)), 5, color=(0,255,0), thickness=2)  # green
#                         img = cv2.circle(img, (int(lowerCrossCorner_x), int(lowerCrossCorner_y)), 5, color=(255,0,255), thickness=2)  # purple
#
#                         #################### MEANING OF THE COLORS/ANNOTATIONS ##########################
#                         # The cars are always in this angle
#                         #
#                         #       (0,0) >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  (x,0)
#                         #         v
#                         #         v        cyan  ~~~~~~~~~~
#                         #         v        /                ~~~~~~~~~~ red
#                         #         v   white  ~~~~~~~~~~               /  |
#                         #         v     |              ~~~~~~~~~~ yellow |
#                         #         v     |                           |    |
#                         #         v     |                           |    |
#                         #         v     |  purple                   |    |
#                         #         v     | /                         |   blue
#                         #         v   green  ~~~~~~~~~              |   /
#                         #         v                    ~~~~~~~~~~~~black
#                         #       (0,y)
#                         #
#                         #    In my first step it only seems necessary to teach the net to some sides of this cube
#                         #
#
#                     outfile = "inferred/" + name + "-" +str(frame_no) + "-"+ str(test_frame) + ".bmp"
#                     print("write to", outfile)
#
#                     cv2.imwrite(outfile, img)

if IS_VEHICLE:
    read_vehReID_random()