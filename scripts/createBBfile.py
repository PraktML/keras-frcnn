import os
import csv
import scripts.settings as settings
import cv2
import json
import numpy as np

""" Convert the notations form the following datasets to be used with Keras-frcnn in the simple data reader mode
the 3D bounding boxes form VehicleReI into a front and a back bounding box. """


LIMIT_OUTPUT = 100  # only write the first n entries or "" for no limit

OUTPUT_FILE = "../annotations/"
#OUTPUT_FILE += "bb"+str(LIMIT_OUTPUT)+".txt"
OUTPUT_FILE += "bb_3Dreg.txt"
OUTPUT_CUT = "cropped/"

# this is what is added before entries
TARGET_PATH_VRI = settings.SHOTS_FOLDER #"/home/patrick/MLPrakt/Data/VehicleReId/video_shots/" #"/media/mlprak1/Seagate Backup Plus Drive/VehicleReId/video_shots/" #  "/data/mlprak1/VehicleReId/video_shots/" # ""VehicleReId/video_shots/"  # no spaces possible here!
TARGET_NUMBER_FORMAT_VRI = '%06d'
TARGET_SUFFIX_VRI = '.bmp'
TARGET_CUT = TARGET_PATH_VRI+OUTPUT_CUT
ANNOTATION_FOLDER = settings.SHOTS_FOLDER # settings.PLATTE_BASEPATH + "VehicleReId/video_shots/"

# this is what is added before the entries for BoxCar annotations
TARGET_PATH_BOX = settings.BOXCARS_FOLDER # "/Users/kolja/Downloads/BoxCars116k/images/" #"BoxCars116k/images/"

DATA_FORMAT = "3d_reg" # in ["3d_reg", "merge_areas"]


counter = 0
min_y = 10000000
max_y = 0
cut = 300 # maximum pixels to be cut
puffer = 20 # puffer between highest y-coordinate in the picture and the cut



if not os.path.exists(TARGET_CUT):
    os.makedirs(TARGET_CUT)


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


    print("write to", OUTPUT_FILE)


    for shot in settings.FRAMES_VRI:
        if counter == 'break': break
        print("processing shot", shot['name'])
        anno_file = ANNOTATION_FOLDER + shot['name'] + "_annotations.txt"
        if not os.path.isfile(anno_file):
            print("Annotation File:", anno_file, "doesn't exist, skip")
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



                if frame < shot['from']:
                    continue
                if frame > shot['to']:
                    break

                if frame + shot['offset'] == 2818 and shot['name'] == "2B" or frame + shot['offset'] == 9262 and shot['name'] == "3A":
                    print("manually skipped this pictures!!!!") #TODO: fix this, either get the frame or find a nicer way
                    continue
                raw_frame_path =  TARGET_PATH_VRI + shot['name'] + "/" + shot['name'] + "_" + TARGET_NUMBER_FORMAT_VRI % (frame + shot['offset']) + TARGET_SUFFIX_VRI
                frame_path =  TARGET_CUT + shot['name'] + "_" + TARGET_NUMBER_FORMAT_VRI % (frame + shot['offset']) + TARGET_SUFFIX_VRI
                img_path = TARGET_CUT + shot['name'] + "/"

                if DATA_FORMAT == "3d_reg":
                    x_points = [upperPointShort_x, upperPointCorner_x, upperPointShort_x, crossCorner_x, shortSide_x, corner_x, longSide_x, lowerCrossCorner_x]
                    y_points = [upperPointShort_y, upperPointCorner_y, upperPointShort_y, crossCorner_y, shortSide_y, corner_y, longSide_y, lowerCrossCorner_y]

                    

                    #Crop Images
                    helpmin_y = min([y for y in y_points])
                    helpmax_y = max([y for y in y_points])
                    if helpmin_y < min_y:
                        min_y = helpmin_y
                    if helpmax_y > max_y:
                        max_y = helpmax_y

                    cutter = min(helpmin_y-puffer, cut)
                    y_points = [y - cutter for y in y_points]


                    img = cv2.imread(raw_frame_path)
                    height, width, channels = img.shape

                    crop_img = img[cutter:height, :]  # Crop from x, y, w, h -> 100, 200, 300, 400
                    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

                    cv2.imwrite(frame_path, crop_img)
                    print("cropped image: "+str(frame_path))



                    facing_left = upperPointShort_x > upperPointCorner_x
                    csvwriter.writerow({"filepath": frame_path,
                                        "x1": min(x_points),        "y1": min(y_points),
                                        "x2": max(x_points),        "y2": max(y_points),
                                        "top_front_right_x": upperPointShort_x if facing_left else upperPointCorner_x,
                                        "top_front_right_y": upperPointShort_y - cutter if facing_left else upperPointCorner_y  - cutter,
                                        "top_front_left_x": upperPointCorner_x if facing_left else upperPointShort_x,
                                        "top_front_left_y": upperPointCorner_y - cutter if facing_left else upperPointShort_y  - cutter,
                                        "top_back_left_x": crossCorner_x if facing_left else upperPointLong_x,
                                        "top_back_left_y": crossCorner_y - cutter if facing_left else upperPointLong_y  - cutter,
                                        "top_back_right_x": upperPointLong_x if facing_left else crossCorner_x,
                                        "top_back_right_y": upperPointLong_y  - cutter if facing_left else crossCorner_y  - cutter,

                                        "bot_front_right_x": shortSide_x if facing_left else corner_x,
                                        "bot_front_right_y": shortSide_y - cutter if facing_left else corner_y - cutter,
                                        "bot_front_left_x": corner_x if facing_left else shortSide_x,
                                        "bot_front_left_y": corner_y - cutter if facing_left else shortSide_y - cutter,
                                        "bot_back_left_x": lowerCrossCorner_x if facing_left else longSide_x,
                                        "bot_back_left_y": lowerCrossCorner_y  - cutter if facing_left else longSide_y  - cutter,
                                        "bot_back_right_x": longSide_x if facing_left else lowerCrossCorner_x,
                                        "bot_back_right_y": longSide_y  - cutter if facing_left else lowerCrossCorner_y  - cutter,
                                        "class_name": "3DBB"
                                        })
                else:
                    # outer boundingbox: top left and bottom right corner
                    # (green_x, cyan_y) - (red_x, black_y)
                    csvwriter.writerow({
                        "filepath": frame_path,
                        "x1": longSide_x,        "y1": crossCorner_y,
                        "x2": upperPointShort_x, "y2": corner_y,
                        "class_name": "outer"
                                      })
                    # facing boundingbox: described by red and black, on the left side of the picture they are facing us "frontBB" on the right "backBB"
                    # (black_x, red_y) - (red_x, black_y)
                    csvwriter.writerow({
                        "filepath": frame_path,
                        "x1": corner_x,        "y1": upperPointShort_y,
                        "x2": upperPointShort_x, "y2": corner_y,
                        "class_name": "front" if shot["sep_m"] * corner_x + shot["sep_y"] < corner_y else "back"
                    })
                    # (facing) side boundingbox: described by white and black
                    # (white_x, white_y) - (black_x, black_y)
                    csvwriter.writerow({
                        "filepath": frame_path,
                        "x1": upperPointLong_x, "y1": upperPointLong_y,
                        "x2": corner_x, "y2": corner_y,
                        "class_name": "side"
                    })
                    # (top) side boundingbox: described by white and black
                    # (white_x, cyan_y) - (red_x, yellow_y)
                    csvwriter.writerow({
                        "filepath": frame_path,
                        "x1": upperPointLong_x, "y1": crossCorner_y,
                        "x2": upperPointShort_x, "y2": upperPointCorner_y,
                        "class_name": "top"
                    })

        print(shot['name'])
        print("min: " + str(min_y))
        print("max: " + str(max_y))

    with open(settings.BOXCARS_FOLDER + "json_data/dataset.json") as jsonfile:
        print("read in", settings.BOXCARS_FOLDER + "json_data/dataset.json")
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


        print("go through instances")
        for instance in instances:
            frame_path = TARGET_PATH_BOX + instance["path"]
            # outer boundingbox: top left and bottom right corner
            # (green_x, cyan_y) - (red_x, black_y)
            points = np.array([(int(xy[0]-instance["3DBB_offset"][0]), int(xy[1]-instance["3DBB_offset"][1]))
                               for xy in instance["3DBB"]])

            # Make sure we switch front and back in case the car is driving away from the camera
            to_cam= instance['to_camera']
            offset = lambda v, towards_camera: (((v + 2) % 4) + ((v // 4) * 4)) if not towards_camera else v

            if DATA_FORMAT == "3d_reg":
                csvwriter.writerow({"filepath": frame_path,
                                    "x1": min([point[0] for point in points]),        "y1": min([point[1] for point in points]),
                                    "x2": max([point[0] for point in points]),        "y2": max([point[1] for point in points]),
                                    "top_front_right_x": points[offset(0, to_cam)][0], "top_front_right_y": points[offset(0, to_cam)][1],
                                    "top_front_left_x": points[offset(1, to_cam)][0], "top_front_left_y": points[offset(1, to_cam)][1],
                                    "top_back_left_x": points[offset(2, to_cam)][0], "top_back_left_y": points[offset(2, to_cam)][1],
                                    "top_back_right_x": points[offset(3, to_cam)][0], "top_back_right_y": points[offset(3, to_cam)][1],
                                    "bot_front_right_x": points[offset(4, to_cam)][0], "bot_front_right_y": points[offset(4, to_cam)][1],
                                    "bot_front_left_x": points[offset(5, to_cam)][0], "bot_front_left_y": points[offset(5, to_cam)][1],
                                    "bot_back_left_x": points[offset(6, to_cam)][0], "bot_back_left_y": points[offset(6, to_cam)][1],
                                    "bot_back_right_x": points[offset(7, to_cam)][0], "bot_back_right_y": points[offset(7, to_cam)][1],
                                    "class_name": "3DBB"
                                    })
            else:
                points_facing = points[[0,1,4,5], :]
                points_side = points[[1,2,5,6], :]
                points_top = points[[0,1,2,3], :]
                csvwriter.writerow({
                    "filepath": frame_path,
                    "x1": min([point[0] for point in points]),        "y1": min([point[1] for point in points]),
                    "x2": max([point[0] for point in points]),        "y2": max([point[1] for point in points]),
                    "class_name": "outer"
                })
                # facing boundingbox: described by red and black, on the left side of the picture they are facing us "frontBB" on the right "backBB"
                # (black_x, red_y) - (red_x, black_y)
                csvwriter.writerow({
                    "filepath": frame_path,
                    "x1": min([point[0] for point in points_facing]),        "y1": min([point[1] for point in points_facing]),
                    "x2": max([point[0] for point in points_facing]),        "y2": max([point[1] for point in points_facing]),
                    "class_name": "front" if instance["to_camera"] else "back"
                })
                # (facing) side boundingbox: described by white and black
                # (white_x, white_y) - (black_x, black_y)
                csvwriter.writerow({
                    "filepath": frame_path,
                    "x1": min([point[0] for point in points_side]),        "y1": min([point[1] for point in points_side]),
                    "x2": max([point[0] for point in points_side]),        "y2": max([point[1] for point in points_side]),
                    "class_name": "side"
                })
                # (top) side boundingbox: described by
                csvwriter.writerow({
                    "filepath": frame_path,
                    "x1": min([point[0] for point in points_top]),        "y1": min([point[1] for point in points_top]),
                    "x2": max([point[0] for point in points_top]),        "y2": max([point[1] for point in points_top]),
                    "class_name": "top"
                })


