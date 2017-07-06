import os
import csv
import scripts.settings as settings
import json
import numpy as np

""" Convert the notations form the following datasets to be used with Keras-frcnn in the simple data reader mode
the 3D bounding boxes form VehicleReI dinto a front and a back bounding box. """


LIMIT_OUTPUT = "" # only write the first n entries or "" for no limit

OUTPUT_FILE = "../annotations/"
#OUTPUT_FILE += "bb"+str(LIMIT_OUTPUT)+".txt"
OUTPUT_FILE += "bb_3Dreg.txt"

TARGET_PATH_VRI = "/media/mlprak1/Seagate Backup Plus Drive/VehicleReId/video_shots/" #  "/data/mlprak1/VehicleReId/video_shots/" # ""VehicleReId/video_shots/"  # no spaces possible here!
TARGET_NUMBER_FORMAT_VRI = '%06d'
TARGET_SUFFIX_VRI = '.bmp'
ANNOTATION_FOLDER = settings.PLATTE_BASEPATH + "VehicleReId/video_shots/"

TARGET_PATH_BOX = "/disk/no_backup/mlprak1/BoxCars116k/images/" #"BoxCars116k/images/"

DATA_FORMAT = "3d_reg" # in ["3d_reg", "merge_areas"]


counter = 0
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
                frame_path = TARGET_PATH_VRI + shot['name'] + "/" + shot['name'] + "_" + TARGET_NUMBER_FORMAT_VRI % (frame + shot['offset']) + TARGET_SUFFIX_VRI

                if DATA_FORMAT == "3d_reg":
                    x_points = [upperPointShort_x, upperPointCorner_x, upperPointShort_x, crossCorner_x, shortSide_x, corner_x, longSide_x, lowerCrossCorner_x]
                    y_points = [upperPointShort_y, upperPointCorner_y, upperPointShort_y, crossCorner_y, shortSide_y, corner_y, longSide_y, lowerCrossCorner_y]

                    facing_left = upperPointShort_x >upperPointCorner_x
                    csvwriter.writerow({"filepath": frame_path,
                                        "x1": min(x_points),        "y1": min(y_points),
                                        "x2": max(x_points),        "y2": max(y_points),
                                        "top_front_right_x": upperPointShort_x if facing_left else upperPointCorner_x,
                                        "top_front_right_y": upperPointShort_y if facing_left else upperPointCorner_y,
                                        "top_front_left_x": upperPointCorner_x if facing_left else upperPointShort_x,
                                        "top_front_left_y": upperPointCorner_y if facing_left else upperPointShort_y,
                                        "top_back_left_x": crossCorner_x if facing_left else upperPointLong_x,
                                        "top_back_left_y": crossCorner_y if facing_left else upperPointLong_y,
                                        "top_back_right_x": upperPointLong_x if facing_left else crossCorner_x,
                                        "top_back_right_y": upperPointLong_y if facing_left else crossCorner_y,

                                        "bot_front_right_x": shortSide_x if facing_left else corner_x,
                                        "bot_front_right_y": shortSide_y if facing_left else corner_y,
                                        "bot_front_left_x": corner_x if facing_left else shortSide_x,
                                        "bot_front_left_y": corner_y if facing_left else shortSide_y,
                                        "bot_back_left_x": lowerCrossCorner_x if facing_left else longSide_x,
                                        "bot_back_left_y": lowerCrossCorner_y if facing_left else longSide_y,
                                        "bot_back_right_x": longSide_x if facing_left else lowerCrossCorner_x,
                                        "bot_back_right_y": longSide_y if facing_left else lowerCrossCorner_y,
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

            if DATA_FORMAT == "3d_reg":
                csvwriter.writerow({"filepath": frame_path,
                                    "x1": min([point[0] for point in points]),        "y1": min([point[1] for point in points]),
                                    "x2": max([point[0] for point in points]),        "y2": max([point[1] for point in points]),
                                    "top_front_right_x": points[0][0], "top_front_right_y": points[0][1],
                                    "top_front_left_x": points[1][0], "top_front_left_y": points[1][1],
                                    "top_back_left_x": points[2][0], "top_back_left_y": points[2][1],
                                    "top_back_right_x": points[3][0], "top_back_right_y": points[3][1],
                                    "bot_front_right_x": points[4][0], "bot_front_right_y": points[4][1],
                                    "bot_front_left_x": points[5][0], "bot_front_left_y": points[5][1],
                                    "bot_back_left_x": points[6][0], "bot_back_left_y": points[6][1],
                                    "bot_back_right_x": points[7][0], "bot_back_right_y": points[7][1],
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
