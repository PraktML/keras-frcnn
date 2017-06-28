import os
import csv
import scripts.settings as settings

""" Convert the notations form the following datasets to be used with Keras-frcnn in the simple data reader mode
the 3D bounding boxes form VehicleReI dinto a front and a back bounding box. """


LIMIT_OUTPUT = "" # only write the first n entries or "" for no limit

#OUTPUT_FILE = settings.PROJECTS_BASEPATH + "keras-frcnn/annotations/"
OUTPUT_FILE = "../annotations/"
#OUTPUT_FILE += "bb"+str(LIMIT_OUTPUT)+".txt"
OUTPUT_FILE += "bb_offset.txt"
TARGET_PATH = ""  # no spaces possible here!
TARGET_NUMBER_FORMAT = '%06d'
TARGET_SUFFIX = '.bmp'

shots_meta = [
    {"name": "1A", "from": 0, "to":8000, "offset": 0},
    {"name": "1B", "from": 0, "to":8000, "offset": -2},
    {"name": "2A", "from": 0, "to": 8000, "offset": 0},
    {"name": "2B", "from": 0, "to": 8000, "offset": -2},
    {"name": "3A", "from": 0, "to": 8000, "offset": 1},
    {"name": "3B", "from": 0, "to": 7000, "offset": -2},
    {"name": "4A", "from": 0, "to": 8000, "offset": 0},
 #   {"name": "4B", "from": 0, "to": 8000, "offset": 0},
    {"name": "5A", "from": 0, "to": 8000, "offset": 2},
 #   {"name": "5B", "from": 0, "to": 8000, "offset": 0},
]
counter = 0
with open(OUTPUT_FILE, 'w+') as outfile:
    fieldnames = ["filepath", "x1", "y1", "x2", "y2", "class_name"]
    csvwriter = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',', lineterminator='\n')
    print("write to", OUTPUT_FILE)

    for shot in shots_meta:
        if counter == 'break': break
        print("processing shot", shot['name'])
        anno_file = settings.SHOTS_FOLDER + shot['name'] + "_annotations.txt"
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
                frame_path = TARGET_PATH + shot['name'] + "/" + shot['name'] + "_" + TARGET_NUMBER_FORMAT % (frame + shot['offset'])+ TARGET_SUFFIX

                # outer boundingbox: top left and bottom right corner
                # (green_x, cyan_y) - (red_x, black_y)
                csvwriter.writerow({
                    "filepath": frame_path,
                    "x1": longSide_x,        "y1": crossCorner_y,
                    "x2": upperPointShort_x, "y2": corner_y,
                    "class_name": "outerBB"
                                  })
                # front boundingbox: described by red and black
                # (black_x, red_y) - (red_x, black_y)
                csvwriter.writerow({
                    "filepath": frame_path,
                    "x1": corner_x,        "y1": upperPointShort_y,
                    "x2": upperPointShort_x, "y2": corner_y,
                    "class_name": "frontBB"
                })
                # (facing) side boundingbox: described by white and black
                # (white_x, white_y) - (black_x, black_y)
                csvwriter.writerow({
                    "filepath": frame_path,
                    "x1": upperPointLong_x, "y1": upperPointLong_y,
                    "x2": corner_x, "y2": corner_y,
                    "class_name": "sideBB"
                })