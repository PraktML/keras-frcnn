from __future__ import print_function
import cv2
import scripts.settings
import numpy as np


def get_data(input_path, train_test_split=None, test_only=False):



    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    num_lines = sum(1 for _ in open(input_path, 'r'))
    if train_test_split is not None:
        with open(input_path, 'r') as f:
            anno_bbs = set([
                line.strip().split(',')[0] for line in f
            ])
            count_anno_bb = len(anno_bbs)
        if count_anno_bb != len(train_test_split):
            input("Length of splits file {} and # of bounding box entries {} don't match, "
                  "if you press enter the splits file will be ignored".format(len(train_test_split), count_anno_bb))
            train_test_split = None

    with open(input_path, 'r') as f:
        idx = 0

        for line in f:
            line_split = line.strip().split(',')

            try:
                #(filename, x1, y1, x2, y2, class_name) = line_split
            #except:
                (filename, x1, y1, x2, y2,
                 bb_x1, bb_y1, bb_x2, bb_y2, bb_x3, bb_y3, bb_x4, bb_y4,
                 bb_x5, bb_y5, bb_x6, bb_y6, bb_x7, bb_y7, bb_x8, bb_y8,
                 class_name
                 ) = line_split
            except:
                print(line_split, "from file", input_path, "not splittable")
                raise
            # add prefix to filename
            filename = scripts.settings.variable_path_to_abs(filename)

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                # each image file is only read in once, but there will be several BB in it.
                all_imgs[filename] = {}

                if idx % (num_lines//10) == 0:
                    print("Simple Parser:", idx, "read", filename)

                img = cv2.imread(filename)
                if img is None:
                    raise (FileNotFoundError(filename))
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []

                # determine if the file will be test or training data
                if train_test_split is None:
                    if np.random.randint(0, 6) >= 1 and not test_only:
                        all_imgs[filename]['imageset'] = 'trainval'
                    else:
                        all_imgs[filename]['imageset'] = 'test'
                else:
                    splitfilename = filename.replace(
                        "/media/florian/PLATTE/programmieren/VehicleReId/video_shots/",
                        "/data/mlprak1/VehicleReId/video_shots/"
                    )

                    splitfilename = splitfilename.replace(
                        "/media/florian/PLATTE/programmieren/BoxCars116k/",
                        "/disk/ml/datasets/BoxCars116k/"
                    )
                    all_imgs[filename]['imageset'] = train_test_split[splitfilename]

            idx += 1
            try:
                all_imgs[filename]['bboxes'].append(
                    {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2),
                     'bb_x1': int(bb_x1), 'bb_y1': int(bb_y1), 'bb_x2': int(bb_x2), 'bb_y2': int(bb_y2),
                     'bb_x3': int(bb_x3), 'bb_y3': int(bb_y3), 'bb_x4': int(bb_x4), 'bb_y4': int(bb_y4),
                     'bb_x5': int(bb_x5), 'bb_y5': int(bb_y5), 'bb_x6': int(bb_x6), 'bb_y6': int(bb_y6),
                     'bb_x7': int(bb_x7), 'bb_y7': int(bb_y7), 'bb_x8': int(bb_x8), 'bb_y8': int(bb_y8),
                     },
                )
            except:
                all_imgs[filename]['bboxes'].append(
                    {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

    # make sure the bg class is last in the list
    if found_bg:
        if class_mapping['bg'] != len(class_mapping) - 1:
            key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
            val_to_switch = class_mapping['bg']
            class_mapping['bg'] = len(class_mapping) - 1
            class_mapping[key_to_switch] = val_to_switch

    return all_imgs, classes_count, class_mapping
