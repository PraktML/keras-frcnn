import numpy as np
import pdb
import math
from . import data_generators
import copy


def calc_iou(R, img_data, C, class_mapping):
    """
    :param R: NMS ROIs shape [(num_anchors*img_width*img_height=num_rois, 4), (num_anchors * img_width * img_height,)]
    :param img_data: {'bboxes': [{'x1': #, .. 'y2': #, 'bb_x1': #, .. 'bb_y8': #}, {'x1': #, .. 'bb_y8': #} .. ]
                      'filepath': ..., 'height': ... 'width': ... }
    :param C: config file
    :param class_mapping: {'class1': 0, 'class2': 1, ... 'bg': #, ...}
    :return: [X, Y1, Y2]
        X = x_roi + 1Dim, shape: (1, num_rois, 4) foreach roi: [x1, y1, w, h]
        Y1 = y_class_num + 1Dim, shape: (1, num_rois, num_classes) foreach roi: one-hot vector with target class
        Y2 = concat<y_class_regr_label, y_class_regr_coords> shape: (1, num_rois, num_output==20*(2==num_classes-1) * 2)
            for each ROI: the first 20*(2-1) entries are a 20-hot encoding of the class 
                          the last  20*(2-1) entries are the regression values for each of the 20 output values. 
    """ # TODO: figure what Y1, Y2 really are, they are fed in form:

    bboxes = img_data['bboxes']
    (width, height) = (img_data['width'], img_data['height'])
    # get image dimensions for resizing
    (resized_width, resized_height) = data_generators.get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 20))

    for bbox_num, bbox in enumerate(bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / C.rpn_stride))

        # TODO: this doesn't work for augmenting the images, as the values are rotated/flipped?
        # Select all 8 points from 3DBB
        gta[bbox_num, 4] = int(round(bbox['bb_x1'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 5] = int(round(bbox['bb_x2'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 6] = int(round(bbox['bb_x3'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 7] = int(round(bbox['bb_x4'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 8] = int(round(bbox['bb_x5'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 9] = int(round(bbox['bb_x6'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 10] = int(round(bbox['bb_x7'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 11] = int(round(bbox['bb_x8'] * (resized_width / float(width)) / C.rpn_stride))

        gta[bbox_num, 12] = int(round(bbox['bb_y1'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 13] = int(round(bbox['bb_y2'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 14] = int(round(bbox['bb_y3'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 15] = int(round(bbox['bb_y4'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 16] = int(round(bbox['bb_y5'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 17] = int(round(bbox['bb_y6'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 18] = int(round(bbox['bb_y7'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 19] = int(round(bbox['bb_y8'] * (resized_height / float(height)) / C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []

    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        for bbox_num in range(len(bboxes)):
            curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                           [x1, y1, x2, y2])
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < C.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])

            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                # hard negative example
                cls_name = 'bg'
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))

                # Make sure we don't log on 0
                eps = 10e-10

                # Calculating ground truth for regression parameters of 3d bounding box
                acc_3d = ([safe_log((gta[best_bbox, 1] - gta[best_bbox, i + 4]) / float(w)) for i in range(8)] +
                          [safe_log((gta[best_bbox, 3] - gta[best_bbox, i + 12]) / float(h)) for i in range(8)])

            else:
                print('roi = {} should not appear in calc_iou, only good (>.7) or bad (<.3) examples'.format(best_iou))
                raise RuntimeError

        # create one-hot encoding.
        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1
        y_class_num.append(copy.deepcopy(class_label))
        coords = [0] * 20 * (len(class_mapping) - 1)
        labels = [0] * 20 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 20 * class_num
            sx, sy, sw, sh = C.classifier_regr_std
            coords[label_pos:20 + label_pos] = (
                [sx * tx, sy * ty, sw * tw, sh * th] +  # outer BB regr values
                [sw * v for v in acc_3d[:8]] +  # all 8 x values
                [sh * v for v in acc_3d[8:]]  # all 8 y values
            )
            labels[label_pos:20 + label_pos] = [1] * 20
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None

    X = np.array(x_roi)
    Y1 = np.array(y_class_num)
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0)


def safe_log(val):
    eps = 10e-6

    if val <= 0:
        val = eps

    return np.log(val)


def apply_regr(x, y, w, h, tx, ty, tw, th, bb3d):
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h

        bb3d_x = [int(round(math.exp(v) * w)) for v in bb3d[:8]]
        bb3d_y = [int(round(math.exp(v) * h)) for v in bb3d[8:]]

        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1, bb3d_x + bb3d_y

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw) * w
        h1 = np.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, probs, overlap_thresh=0.9, max_boxes=300):
    """

    :param boxes: shape (num_anchors * width * height, 4)
    :param probs: shape (num_anchors * width * height)
    :param overlap_thresh:
    :param max_boxes:
    :return: [boxes, probabilities] that pass
             shapes [(num_anchors * img_width * img_height - not, 4), (num_anchors * img_width * img_height - not)]

    """
    # code used from here: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    np.testing.assert_array_less(x1, x2)
    np.testing.assert_array_less(y1, y2)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # calculate the areas
    area = (x2 - x1) * (y2 - y1)

    # sort the bounding boxes
    idxs = np.argsort(probs)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the intersection

        xx1_int = np.maximum(x1[i], x1[idxs[:last]])
        yy1_int = np.maximum(y1[i], y1[idxs[:last]])
        xx2_int = np.minimum(x2[i], x2[idxs[:last]])
        yy2_int = np.minimum(y2[i], y2[idxs[:last]])

        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)

        area_int = ww_int * hh_int

        # find the union
        area_union = area[i] + area[idxs[:last]] - area_int

        # compute the ratio of overlap
        overlap = area_int / (area_union + 1e-6)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    # return only the bounding boxes that were picked using the integer data type
    boxes = boxes[pick].astype("int")
    probs = probs[pick]
    return boxes, probs


import time


def rpn_to_roi(rpn_layer, regr_layer, C, dim_ordering, use_regr=True, max_boxes=300, overlap_thresh=0.9):
    """
    :param rpn_layer: predictions from the RPN network in the format
                        concat<y_is_box_valid, y_rpn_overlap>, shape (1, 2*num_anchors, img_width, img_height) if 'tf'
    :param regr_layer: predictions from the RPN network in format
                        concat<4*y_rpn_overlap, y_rpn_regr>, shape (1, 2*4*num_anchors, img_width, img_height) if 'tf'
    :param C: config file
    :param dim_ordering: 'tf' tensorflow or 'th' theano
    :param use_regr: use regr_layer to adjust the positions of the fixed grid of the RPN regions
    :param max_boxes: NMS parameter
    :param overlap_thresh: NMS parameter
    :return: all_boxes after NMS shape (num_anchors * img_width * img_height - not passed, 4)
    """
    regr_layer = regr_layer / C.std_scaling

    anchor_sizes = C.anchor_box_scales
    anchor_ratios = C.anchor_box_ratios

    assert rpn_layer.shape[0] == 1

    if dim_ordering == 'th':
        (rows, cols) = rpn_layer.shape[2:]

    elif dim_ordering == 'tf':
        (rows, cols) = rpn_layer.shape[1:3]

    curr_layer = 0
    if dim_ordering == 'tf':
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
    elif dim_ordering == 'th':
        A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))
    # A has four times stacked a prediction for each output prediction from the RPN
    # A shape: (4, width, height, 2*num_anchors), these four values are x1, y1, x2, y2 (not width/height)

    for anchor_size in anchor_sizes:  # e.g. [128, 256, 512]
        for anchor_ratio in anchor_ratios:  # e.g. [[1, 1], [1, 2], [2, 1]]

            # anchor width and height
            anchor_x = (anchor_size * anchor_ratio[0]) / C.rpn_stride  # e.g. (128 * 1) / 16 = 8
            anchor_y = (anchor_size * anchor_ratio[1]) / C.rpn_stride
            if dim_ordering == 'th':
                regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            else: # 'tf'
                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
                regr = np.transpose(regr, (2, 0, 1))

            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))  # for e.g. cols=4, rows=3 this creates:
            # X = array([[0, 1, 2, 3],        Y = array([[0, 0, 0, 0],    X filled with values from arange(cols),
            #            [0, 1, 2, 3],                   [1, 1, 1, 1],    Y filled with values from arange(rows)
            #            [0, 1, 2, 3]])                  [2, 2, 2, 2]])   both have shape: (rows, cols)

            A[0, :, :, curr_layer] = X - anchor_x / 2
            A[1, :, :, curr_layer] = Y - anchor_y / 2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # ROI width are at least 1 pixel long and the position of x2 is anchor_width + anchor position
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]

            # assure that the ROIs are within the image
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

            curr_layer += 1

    # transposing A to shape (4, num_anchors, width, height)
    # then reshaping & transposing A to shape (num_anchors * width * height, 4)
    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    # reshaping the predicted valid & overlap to shape (1, num_anchors, width, height) and then flatten it
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]

    # ROIs with length|height 0 or less will be deleted.
    idxs = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, idxs, 0)
    all_probs = np.delete(all_probs, idxs, 0)

    result = non_max_suppression_fast(all_boxes, all_probs, overlap_thresh=overlap_thresh, max_boxes=max_boxes)[0]

    return result
