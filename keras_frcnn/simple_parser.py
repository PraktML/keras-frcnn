from __future__ import print_function
import cv2
import scripts.settings
import numpy as np


def get_data(input_path, image_folder='', train_test_split=None):
    assert image_folder=='' or image_folder[-1] == '/'
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True



    with open(input_path, 'r') as f:

        # # check if the right splits file is provided
        # if train_test_split is not None:
        #     count_bb = sum(1 for _ in f)
        #     if len(train_test_split)!=count_bb:
        #         input("Length of splits file {} and # of bounding box entries {} don't match, "
        #               "if you press enter the splits file will be ignored".format(len(train_test_split),count_bb))
        #         train_test_split = None
        # print('Parsing annotation files')
        # f.seek(0)
        idx = 0
        for line in f:
            line_split = line.strip().split(',')
            (filename, x1, y1, x2, y2, class_name) = line_split

            # add prefix to filename
            filename = image_folder + filename

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
                #each image file is only read in once, but there will be several BB in it.
                all_imgs[filename] = {}

                print("Simple Parser: read", scripts.settings.PROJECTS_BASEPATH + filename)
                img = cv2.imread(filename)
                if img is None:
                    raise(FileNotFoundError(filename))
                (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []

                # determine if the file will be test or training data
                if train_test_split is None:
                    if np.random.randint(0, 6) >= 0:
                        all_imgs[filename]['imageset'] = 'trainval'
                    else:
                        all_imgs[filename]['imageset'] = 'test'
                else:
                    all_imgs[filename]['imageset'] = train_test_split[filename]

            print(idx, "append to", filename, ".")
            idx+=1
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
