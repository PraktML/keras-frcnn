import os
import csv
import scripts.settings as settings
import cv2
import json
import numpy as np

""" 
Convert the notations form the following datasets to be used with Keras-frcnn in the simple data reader mode
the 3D bounding boxes form VehicleReI into a front and a back bounding box. 
The absolute Paths are replaced with relative variables
"""

DATA_FORMAT = "3d_reg" # in ["3d_reg", "merge_areas"]
LIMIT_OUTPUT = 100  # only write the first n entries or "" for no limit


NUMBER_FORMAT_VRI = '%06d'
SUFFIX_VRI = '.png'

OUTPUT_FILE = "../annotations/"
OUTPUT_FILE += "bb_3DregCrop.txt"
OUTPUT_CUT = "cropped/"

# shall the VRI images be cropped?
USE_VRI_CUTTING = True
# output format of the cropped images for VRI
SUFFIX_VRI_CUT = '.png'

VRI_CUT_MIN_Y = 10000000
VRI_CUT_MAX_Y = 0
VRI_CUT_NO_PIXELS = 300 # maximum pixels to be cut
VRI_CUT_BUFFER = 20 # buffer between highest y-coordinate in the picture and the cut


with open(OUTPUT_FILE, 'w+') as outfile:
    fieldnames_area_merging = ["filepath", "x1", "y1", "x2", "y2", "class_name"]
    fieldnames_3d_reg = ["filepath", "x1", "y1", "x2", "y2",
                         "top_front_right_x", "top_front_right_y", "top_front_left_x", "top_front_left_y",
                         "top_back_left_x", "top_back_left_y", "top_back_right_x", "top_back_right_y",
                         "bot_front_right_x", "bot_front_right_y", "bot_front_left_x", "bot_front_left_y",
                         "bot_back_left_x", "bot_back_left_y", "bot_back_right_x", "bot_back_right_y",
                         "class_name"
                         ]
    csvwriter = csv.DictWriter(outfile, delimiter=',', lineterminator='\n',
                               fieldnames=fieldnames_area_merging if DATA_FORMAT == "merge_areas" else fieldnames_3d_reg)
    print("Will create annotation in file:", OUTPUT_FILE)
    counter = 0
    for shot in settings.FRAMES_VRI:
        if counter == 'break': break
        anno_file = settings.VRI_SHOTS_PATH + shot['name'] + "_annotations.txt"
        print("processing shot", shot['name'], "reading in", anno_file)

        if not os.path.isfile(anno_file):
            print("Couldn't find shot annotation file:", anno_file, "skip")
            continue
        with open(anno_file, 'r') as file:
            csvreader = csv.reader(file, delimiter=',')
            for line in csvreader:

                if LIMIT_OUTPUT != "":
                    counter += 1
                    if counter >= LIMIT_OUTPUT:
                        print("Successfully finished after", counter, "entries")
                        counter = 'break'
                        break

                try:
                    (carId, frame,
                     upperPointShort_x, upperPointShort_y,      #red
                     upperPointCorner_x, upperPointCorner_y,    #yellow
                     upperPointLong_x, upperPointLong_y,        #white
                     crossCorner_x, crossCorner_y,              #cyan
                     shortSide_x, shortSide_y,                  #blue
                     corner_x, corner_y,                        #black
                     longSide_x, longSide_y,                    #green
                     lowerCrossCorner_x, lowerCrossCorner_y     #purple
                     ) = [int(entry) for entry in line]
                except ValueError:
                    print("Warning: invalid line in:", line)
                    continue
                x_points = [upperPointShort_x, upperPointCorner_x, upperPointShort_x, crossCorner_x, shortSide_x, corner_x, longSide_x, lowerCrossCorner_x]
                y_points = [upperPointShort_y, upperPointCorner_y, upperPointShort_y, crossCorner_y, shortSide_y, corner_y, longSide_y, lowerCrossCorner_y]

                if frame < shot['from']:
                    continue
                if frame > shot['to']:
                    break
                if frame + shot['offset'] == 2818 and shot['name'] == "2B" or frame + shot['offset'] == 9262 and shot['name'] == "3A":
                    print("manually skipped this pictures!!!!") #TODO: fix this, either get the frame or find a nicer way
                    continue

                ##############################################################
                #### if specified, crop some part of the VRI images  #########
                ##############################################################

                if not USE_VRI_CUTTING:
                    frame_path_variable = "$VRI_SHOTS_PATH$" + shot['name'] + "_" + NUMBER_FORMAT_VRI % (frame + shot['offset']) + SUFFIX_VRI
                    cutter = 0

                else: # Crop Images
                    if not os.path.exists(settings.VRI_SHOTS_PATH + OUTPUT_CUT):
                        os.makedirs(settings.VRI_SHOTS_PATH + OUTPUT_CUT)

                    frame_path_variable = "$VRI_SHOTS_PATH$" + OUTPUT_CUT + shot['name'] + "_" + NUMBER_FORMAT_VRI % (frame + shot['offset']) + SUFFIX_VRI_CUT
                    raw_frame_path = settings.VRI_SHOTS_PATH + shot['name'] + "/" + shot['name'] + "_" + NUMBER_FORMAT_VRI % (frame + shot['offset']) + SUFFIX_VRI
                    print(raw_frame_path)

                    # Cut relatively from the upper y-coordinate
                    #helpmin_y = min([y for y in y_points])
                    #if helpmin_y < VRI_CUT_MIN_Y:
                    #    VRI_CUT_MIN_Y = helpmin_y
                    #if helpmax_y > VRI_CUT_MAX_Y:
                    #    VRI_CUT_MAX_Y = helpmax_y
                    #cutter = min(helpmin_y - VRI_CUT_BUFFER, VRI_CUT_NO_PIXELS)

                    # Cut a fixed amount of pixes in y-direction
                    helpmin_y = min([y for y in y_points])
                    cutter = shot['y_crop']
                    y_points = [y - cutter for y in y_points]

                    if helpmin_y < cutter:
                        continue

                    img = cv2.imread(raw_frame_path)
                    height, width, channels = img.shape

                    crop_img = img[cutter:height, :]  # Crop from x, y, w, h -> 100, 200, 300, 400
                    # NOTE: it's img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

                    cv2.imwrite(settings.variable_path_to_abs(frame_path_variable), crop_img)
                    print(counter, "cropped image: " + str(frame_path_variable))

                #####################################################################
                ################   CREATING DATA FORMAT 3D-REG   ####################
                #####################################################################

                if DATA_FORMAT == "3d_reg":

                    facing_left = upperPointShort_x > upperPointCorner_x
                    csvwriter.writerow({"filepath": frame_path_variable,
                                        "x1": min(x_points),        "y1": min(y_points),
                                        "x2": max(x_points),        "y2": max(y_points),
                                        "top_front_right_x":    upperPointShort_x if facing_left else upperPointCorner_x,
                                        "top_front_right_y":    upperPointShort_y - cutter if facing_left else upperPointCorner_y - cutter,
                                        "top_front_left_x":     upperPointCorner_x if facing_left else upperPointShort_x,
                                        "top_front_left_y":     upperPointCorner_y - cutter if facing_left else upperPointShort_y - cutter,
                                        "top_back_left_x":      crossCorner_x if facing_left else upperPointLong_x,
                                        "top_back_left_y":      crossCorner_y - cutter if facing_left else upperPointLong_y  - cutter,
                                        "top_back_right_x":     upperPointLong_x if facing_left else crossCorner_x,
                                        "top_back_right_y":     upperPointLong_y  - cutter if facing_left else crossCorner_y  - cutter,

                                        "bot_front_right_x":    shortSide_x if facing_left else corner_x,
                                        "bot_front_right_y":    shortSide_y - cutter if facing_left else corner_y - cutter,
                                        "bot_front_left_x":     corner_x if facing_left else shortSide_x,
                                        "bot_front_left_y":     corner_y - cutter if facing_left else shortSide_y - cutter,
                                        "bot_back_left_x":      lowerCrossCorner_x if facing_left else longSide_x,
                                        "bot_back_left_y":      lowerCrossCorner_y - cutter if facing_left else longSide_y  - cutter,
                                        "bot_back_right_x":     longSide_x if facing_left else lowerCrossCorner_x,
                                        "bot_back_right_y":     longSide_y - cutter if facing_left else lowerCrossCorner_y  - cutter,
                                        "class_name": "3DBB"
                                        })

                #####################################################################
                ################   CREATING DATA MERGE AREAS     ####################
                #####################################################################


                else:
                    # outer boundingbox: top left and bottom right corner
                    # (green_x, cyan_y) - (red_x, black_y)
                    csvwriter.writerow({
                        "filepath": frame_path_variable,
                        "x1": longSide_x,        "y1": crossCorner_y,
                        "x2": upperPointShort_x, "y2": corner_y,
                        "class_name": "outer"
                                      })
                    # facing boundingbox: described by red and black, on the left side of the picture they are facing us "frontBB" on the right "backBB"
                    # (black_x, red_y) - (red_x, black_y)
                    csvwriter.writerow({
                        "filepath": frame_path_variable,
                        "x1": corner_x,        "y1": upperPointShort_y,
                        "x2": upperPointShort_x, "y2": corner_y,
                        "class_name": "front" if shot["sep_m"] * corner_x + shot["sep_y"] < corner_y else "back"
                    })
                    # (facing) side boundingbox: described by white and black
                    # (white_x, white_y) - (black_x, black_y)
                    csvwriter.writerow({
                        "filepath": frame_path_variable,
                        "x1": upperPointLong_x, "y1": upperPointLong_y,
                        "x2": corner_x, "y2": corner_y,
                        "class_name": "side"
                    })
                    # (top) side boundingbox: described by white and black
                    # (white_x, cyan_y) - (red_x, yellow_y)
                    csvwriter.writerow({
                        "filepath": frame_path_variable,
                        "x1": upperPointLong_x, "y1": crossCorner_y,
                        "x2": upperPointShort_x, "y2": upperPointCorner_y,
                        "class_name": "top"
                    })

            # end for loop iterating over VRI

        #
        # print(shot['name'])
        # print("min: " + str(min_y))
        # print("max: " + str(max_y))

    with open(settings.BOXCARS116K_JSON_FILE) as jsonfile:
        print("read in", settings.BOXCARS116K_JSON_FILE)
        data = json.load(jsonfile)["samples"]

    instances = []

    for car in data:
        for instance in car["instances"]:
            # #############################################################
            # and instance could look like
            # {"2DBB": [30,30,51,36], "3DBB": [[132.771,244.777],[113.918,246.792], ... ]
            # "3DBB_offset": [54,212],
            # "instance_id": 0,
            # "path": "uvoz/1/000001_000.png"
            # },
            # ############################################################
            # COLORS = [(0, 0, 255),  # red           0
            #           (0, 255, 255),  # yellow      1
            #           (255, 255, 255),  # white     2
            #           (255, 255, 0),  # cyan        3
            #           (255, 0, 0),  # blue          4
            #           (0, 0, 0),  # black           5
            #           (0, 255, 0),  # green         6
            #           (255, 0, 255),  # purple      7
            #           ]
            instance["to_camera"]  = car["to_camera"]
            instances.append(instance)

    counter = 0
    for instance in instances:
        if LIMIT_OUTPUT != "":
            counter += 1
            if counter >= LIMIT_OUTPUT:
                print("--> Successfully finished after", counter, "entries")
                break

        frame_path_variable = "$BOXCARS116K_PATH$" + instance["path"]
        # outer boundingbox: top left and bottom right corner
        # (green_x, cyan_y) - (red_x, black_y)
        points = np.array([(int(xy[0]-instance["3DBB_offset"][0]), int(xy[1]-instance["3DBB_offset"][1]))
                           for xy in instance["3DBB"]])

        # Make sure we switch front and back in case the car is driving away from the camera
        to_cam = instance['to_camera']
        offset = lambda v, towards_camera: (((v + 2) % 4) + ((v // 4) * 4)) if not towards_camera else v

        #####################################################################
        ################   CREATING DATA FORMAT 3D-REG   ####################
        #####################################################################

        if DATA_FORMAT == "3d_reg":
            csvwriter.writerow({"filepath": frame_path_variable,
                                "x1": min([point[0] for point in points]),          "y1": min([point[1] for point in points]),
                                "x2": max([point[0] for point in points]),          "y2": max([point[1] for point in points]),
                                "top_front_right_x": points[offset(0, to_cam)][0],  "top_front_right_y": points[offset(0, to_cam)][1],
                                "top_front_left_x": points[offset(1, to_cam)][0],   "top_front_left_y": points[offset(1, to_cam)][1],
                                "top_back_left_x": points[offset(2, to_cam)][0],    "top_back_left_y": points[offset(2, to_cam)][1],
                                "top_back_right_x": points[offset(3, to_cam)][0],   "top_back_right_y": points[offset(3, to_cam)][1],
                                "bot_front_right_x": points[offset(4, to_cam)][0],  "bot_front_right_y": points[offset(4, to_cam)][1],
                                "bot_front_left_x": points[offset(5, to_cam)][0],   "bot_front_left_y": points[offset(5, to_cam)][1],
                                "bot_back_left_x": points[offset(6, to_cam)][0],    "bot_back_left_y": points[offset(6, to_cam)][1],
                                "bot_back_right_x": points[offset(7, to_cam)][0],   "bot_back_right_y": points[offset(7, to_cam)][1],
                                "class_name": "3DBB"
                                })

            #####################################################################
            ############## CREATING DATA FORMAT MERGE AREAS  ####################
            #####################################################################


        else:
            points_facing = points[[0,1,4,5], :]
            points_side = points[[1,2,5,6], :]
            points_top = points[[0,1,2,3], :]
            csvwriter.writerow({
                "filepath": frame_path_variable,
                "x1": min([point[0] for point in points]),        "y1": min([point[1] for point in points]),
                "x2": max([point[0] for point in points]),        "y2": max([point[1] for point in points]),
                "class_name": "outer"
            })
            # facing boundingbox: described by red and black, on the left side of the picture they are facing us "frontBB" on the right "backBB"
            # (black_x, red_y) - (red_x, black_y)
            csvwriter.writerow({
                "filepath": frame_path_variable,
                "x1": min([point[0] for point in points_facing]),        "y1": min([point[1] for point in points_facing]),
                "x2": max([point[0] for point in points_facing]),        "y2": max([point[1] for point in points_facing]),
                "class_name": "front" if instance["to_camera"] else "back"
            })
            # (facing) side boundingbox: described by white and black
            # (white_x, white_y) - (black_x, black_y)
            csvwriter.writerow({
                "filepath": frame_path_variable,
                "x1": min([point[0] for point in points_side]),        "y1": min([point[1] for point in points_side]),
                "x2": max([point[0] for point in points_side]),        "y2": max([point[1] for point in points_side]),
                "class_name": "side"
            })
            # (top) side boundingbox: described by
            csvwriter.writerow({
                "filepath": frame_path_variable,
                "x1": min([point[0] for point in points_top]),        "y1": min([point[1] for point in points_top]),
                "x2": max([point[0] for point in points_top]),        "y2": max([point[1] for point in points_top]),
                "class_name": "top"
            })
        #end if
    # end loop iterating over all instances of cars.