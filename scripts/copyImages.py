import csv
import os
from shutil import copyfile

SOURCE_DIR = "C:/Users/Florian/PycharmProjects/reid/"
TARGET_DIR = "images/"

SOURCE_PREFIX = "1A__"
SOURCE_SUFFIX = ".png"
SOURCE_NUMBER_FORMAT = '%05d'

TARGET_PREFIX = "1A_"
TARGET_SUFFIX = ".png"
TARGET_NUMBER_FORMAT = '%05d'

ANNOTATIONS_FILE = "images/1A_annotations.txt"

with open(ANNOTATIONS_FILE) as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        try:
            frame = int(row[1])
        except ValueError:
            print("invalid line", row)
            continue
        except IndexError:
            print("invalid line", row)
            continue
        source_file = SOURCE_DIR + SOURCE_PREFIX + SOURCE_NUMBER_FORMAT % frame + SOURCE_SUFFIX
        target_file = TARGET_DIR + TARGET_PREFIX + TARGET_NUMBER_FORMAT % frame + TARGET_SUFFIX
        if os.path.isfile(TARGET_DIR + TARGET_PREFIX + TARGET_NUMBER_FORMAT % frame + TARGET_SUFFIX):
            print ("file", target_file, "exists, skip it.")
            continue
        print("copy", source_file, "->", target_file)
        copyfile(source_file, target_file)
