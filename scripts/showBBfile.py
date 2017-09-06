from __future__ import print_function
import numpy as np
import cv2
import csv
import os
import shutil
from optparse import OptionParser
import scripts.helper
import scripts.settings
import matplotlib.pyplot as plt

FRAME_PATHS = []  # these paths are tested, additionally to some random ones, specifed by Random_samples
RANDOM_SAMPLES = -1  # set it to -1 to take all
DATA_FORMAT = "3d_reg"  # in ["3d_reg", "merge_areas"]


parser = OptionParser()

parser.add_option("-p", "--bb_path", dest="bb_path",
                  help="Name of bounding box file that shall be graphically displayed")
parser.add_option("-o", "--out_file", dest="out_file", default="../annotations/bb_sel.txt")

(options, args) = parser.parse_args()

if options.bb_path:
    bb_file_path = options.bb_file
else:
    bb_file_path = scripts.helper.chose_from_folder("../annotations/", "*.txt", "--bb_path", )

out_file = options.out_file

bb_file_path = os.path.normpath(bb_file_path)
bb_file_name = os.path.splitext(os.path.basename(bb_file_path))[0]
bb_file_folder_path = os.path.join(*(bb_file_path.split(os.path.sep)[:-1]))

out_folder = os.path.join(bb_file_folder_path, bb_file_name) + "/"


if os.path.exists(out_folder):
    shutil.rmtree(out_folder)
    os.makedirs(out_folder)


def remove_duplicates_sorted(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def read_anno_file():
    with open(bb_file_path) as file:
        print("Reading in the whole annotation file", bb_file_path, "as ")
        entries = list(csv.reader(file, delimiter=','))
        entries = sorted(entries, key=lambda e: e[0])
        sorted_file = bb_file_path+"_sort.txt"
    with open(sorted_file, "w+") as file:
        print("writing sorted file to", sorted_file)
        csvwriter = csv.writer(file, delimiter=",")
        for entry in entries:
            csvwriter.writerow(entry)

    with open(out_file, "w+") as outf:
        csvwriter = csv.writer(outf, delimiter=',')
        if RANDOM_SAMPLES > -1:
            sample_indexes = np.random.choice(len(entries), RANDOM_SAMPLES, replace=False)
        else:
            sample_indexes = list(range(len(entries)))
        FRAME_PATHS.extend([entries[idx][0] for idx in sample_indexes])

        sorted_frame_paths = remove_duplicates_sorted(FRAME_PATHS)
        for frame_path in sorted_frame_paths:

            print("analyze", frame_path)
            # we only have one frame yet, we want all bb annotations that belong to that frame
            frame_entries = []
            for entry in entries:
                # first element is always the path
                if entry[0] == frame_path:
                    frame_entries.append(entry)

            full_frame_path = scripts.settings.variable_path_to_abs(frame_path)
            frame_name_base = os.path.basename(full_frame_path)
            img = cv2.imread(full_frame_path)
            assert img is not None

            for entry in frame_entries:
                coords = np.array(
                    [entry[i] for i in list(range(1, 5)) + list(range(5, 20, 2))+list(range(6, 21, 2))]
                )
                class_name = entry[21]
                img = scripts.helper.draw_annotations(img, coords, fac=class_name)
            result = {'keep': True}

            def press(event):
                if event.key == 'd':
                    result['keep'] = False

            fig, ax = plt.subplots(1)
            ax.imshow(img)
            fig.canvas.mpl_connect('key_press_event', press)
            plt.waitforbuttonpress(0)
            plt.close()
            if result['keep']:
                counter = 0
                for entry in frame_entries:
                    csvwriter.writerow(entry)
                    counter += 1
                print("will keep this file in", out_file, "with", counter, "entries")

            out_path = os.path.join(out_folder, frame_name_base)
            print("write to", out_path)
            if not os.path.exists(out_path[:out_path.rfind("/")]):
                print("created folder", out_path)
                os.makedirs(out_path[:out_path.rfind("/")])
            # cv2.imshow(frame_path, img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(out_path, img)

read_anno_file()
