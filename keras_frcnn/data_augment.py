import cv2
import numpy as np
import copy
#import scripts.helper as helper

def augment(img_data, config, augment=True):
    """
    :return: img_data_aug (the x,y values of the bounding boxes adjusted to the augmentation)
             img          (the augmented image itself)
    """
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

    img = cv2.imread(img_data_aug['filepath'])

    if augment:

        if config.use_crop and np.random.randint(0, 1) == 0 and len(img_data_aug['bboxes']) > 0:
            rows, cols = img.shape[:2]
            bbox_c = img_data_aug['bboxes'][0]
            crop_x1 = max(0, bbox_c['x1'] - np.random.randint(0, 400))
            crop_y1 = max(0, bbox_c['y1'] - np.random.randint(0, 400))
            crop_x2 = min(cols, bbox_c['x2'] + np.random.randint(0, 400))
            crop_y2 = min(rows, bbox_c['y2'] + np.random.randint(0, 400))

            if crop_x1 > 0 or crop_y1 > 0 or crop_x2 < cols or crop_y2 < rows:
                #helper.show_img_data(img, img_data_aug, 600, prefix="before_", scale=False)
                # there happened some actual cropping
                img_data_aug['crop'] = [crop_x1, crop_y1, cols - crop_x2, rows - crop_y2]
                img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
                new_bboxes = []
                for bbox in img_data_aug['bboxes']:
                    new_x1 = bbox['x1'] - crop_x1
                    new_y1 = bbox['y1'] - crop_y1
                    new_x2 = bbox['x2'] - crop_x1
                    new_y2 = bbox['y2'] - crop_y1

                    if new_x1 >= 0 and new_y1 >= 0 and new_x2 < crop_x2 - crop_x1 and new_y2 < crop_y2 - crop_y1:
                        # this bounding box is fully contained in the cropped area
                        new_bbox = {'class': bbox['class']}
                        for key in ['x1', 'x2', 'bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8']:
                            new_bbox[key] = bbox[key] - crop_x1
                        for key in ['y1', 'y2', 'bb_y1', 'bb_y2', 'bb_y3', 'bb_y4', 'bb_y5', 'bb_y6', 'bb_y7', 'bb_y8']:
                            new_bbox[key] = bbox[key] - crop_y1

                        new_bboxes.append(new_bbox)
                assert len(new_bboxes) > 0, "Somehow no bounding box was added"
                img_data_aug['bboxes'] = new_bboxes
                #helper.show_img_data(img, img_data_aug, 600, prefix="after_", scale=False)
                print("now:", img_data_aug)

        rows, cols = img.shape[:2]  # might have changed due to cropping

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img_data_aug['hf'] = True
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = cols - x1
                bbox['x1'] = cols - x2

                if 'bb_x1' in bbox:  # data_format '3d_reg'
                    for key in ['bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8']:
                        # the inner/outer points stay the same, thus no swapping as with the outer bbox necessary
                        bbox[key] = cols - bbox[key]

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            assert False, "doesn't make sense to flip the images like this"
            img_data_aug['vf'] = True
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = rows - y1
                bbox['y1'] = rows - y2

        if config.rot_90:
            assert False, "doesn't make sense to rotate the images like this"
            img_data_aug['rot'] = True
            angle = np.random.choice([0, 90, 180, 270], 1)[0]
            if angle == 270:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = cols - x2
                    bbox['y2'] = cols - x1
                elif angle == 180:
                    bbox['x2'] = cols - x1
                    bbox['x1'] = cols - x2
                    bbox['y2'] = rows - y1
                    bbox['y1'] = rows - y2
                elif angle == 90:
                    bbox['x1'] = rows - y2
                    bbox['x2'] = rows - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

    img_data_aug['width'] = img.shape[1]
    img_data_aug['height'] = img.shape[0]
    return img_data_aug, img
