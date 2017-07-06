import csv
import os
from shutil import copyfile
import scripts.settings

SOURCE_DIR = scripts.settings.PLATTE_BASEPATH
SOURCE_PREFIX = "1A_"
SOURCE_SUFFIX = ".bmp"
SOURCE_NUMBER_FORMAT = '%06d'

TARGET_DIR = scripts.settings.PLATTE_BASEPATH + "small/"
TARGET_PREFIX = "1A_"
TARGET_SUFFIX = ".bmp"
TARGET_NUMBER_FORMAT = '%06d'

ANNOTATIONS_FILE = "../annotations/bb_1-4A.txt"

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

with open(ANNOTATIONS_FILE) as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        try:
            path = row[0]
        except ValueError:
            print("invalid line", row)
            continue
        except IndexError:
            print("invalid line", row)
            continue
        source_file = SOURCE_DIR + path #SOURCE_PREFIX + SOURCE_NUMBER_FORMAT % frame + SOURCE_SUFFIX
        target_file = TARGET_DIR + path # TARGET_PREFIX + TARGET_NUMBER_FORMAT % frame + TARGET_SUFFIX

        if not os.path.exists(target_file[:target_file.rfind("/")]):
            print("created folder for", target_file)
            os.makedirs(target_file[:target_file.rfind("/")])

        if os.path.isfile(target_file):
            print ("file", target_file, "exists, skip it.")
            continue

        print("copy", source_file, "->", target_file)
        copyfile(source_file, target_file)
