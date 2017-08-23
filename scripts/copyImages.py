import csv
import os
from shutil import copyfile
import scripts.settings

SOURCE_BOXCARS116K_PATH = scripts.settings.BOXCARS116K_PATH
SOURCE_VRI_SHOTS_PATH = scripts.settings.VRI_SHOTS_PATH

TARGET_BOXCARS116K_PATH = "/media/florian/Windows8_OS/Users/Florian/PycharmProjects/BoxCars116k/"
TARGET_VRI_SHOTS_PATH = "/media/florian/Windows8_OS/Users/Florian/PycharmProjects/vri_shots/"

BB_FILE = "../annotations/bb_3DregCrop.txt"

# SOURCE_DIR = scripts.settings.VRI_SHOTS_PATH
# SOURCE_PREFIX = "1A_"
# SOURCE_SUFFIX = ".bmp"
# SOURCE_NUMBER_FORMAT = '%06d'
#
# TARGET_DIR = scripts.settings.VRI_SHOTS_PATH + "small/"
# TARGET_PREFIX = "1A_"
# TARGET_SUFFIX = ".bmp"
# TARGET_NUMBER_FORMAT = '%06d'

with open(BB_FILE) as file:
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

#       source_file = SOURCE_DIR + path  # SOURCE_PREFIX + SOURCE_NUMBER_FORMAT % frame + SOURCE_SUFFIX
#       target_file = TARGET_DIR + path  # TARGET_PREFIX + TARGET_NUMBER_FORMAT % frame + TARGET_SUFFIX
        source_file = scripts.settings.variable_path_to_abs(
            path,
            boxcars116k_path=SOURCE_BOXCARS116K_PATH,
            vri_shots_path=SOURCE_VRI_SHOTS_PATH
        )
        target_file = scripts.settings.variable_path_to_abs(
            path,
            boxcars116k_path=TARGET_BOXCARS116K_PATH,
            vri_shots_path=TARGET_VRI_SHOTS_PATH
        )

        if not os.path.exists(target_file[:target_file.rfind(os.path.sep)]):
            print("created folder for", target_file)
            os.makedirs(target_file[:target_file.rfind(os.path.sep)])

        if os.path.isfile(target_file):
            print("file", target_file, "exists, skip it.")
            continue

        print("copy", source_file, "->", target_file)
        copyfile(source_file, target_file)
