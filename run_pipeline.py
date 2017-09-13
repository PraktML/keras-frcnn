from __future__ import division
import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import scripts.helper
from darkflow.net.build import TFNet
import time

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--path", "--path_testdata", dest="test_path", help="Path to test data.",
                  default='images_test/')
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.")  # , default=32)
parser.add_option("--run", "--run_folder", dest="run_folder", help=
"Location to read the metadata related to the training (generated when training).")
parser.add_option("--model", dest="model", help="select which model to take (maybe there are ones from several epochs")
parser.add_option("--print_classes", dest="print_classes", action="store_true", default=False)
parser.add_option("--margin", dest="margin", default="10")
(options, args) = parser.parse_args()

if not options.run_folder:  # if filename is not given
    run_folder = scripts.helper.chose_from_folder("runs/", "*", "--run_folder") + "/"
else:
    run_folder = options.run_folder + "" if options.run_folder[-1] == '/' else '/'
config_output_filename = run_folder + "config.pickle"

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if options.model:
    model_path = options.model
else:
    model_path = scripts.helper.chose_from_folder(run_folder, "*.hdf5", "--model")

margin = int(options.margin)

config_output_filename = run_folder + "config.pickle"
print("Specified Model for Testing:", model_path)

model_name = model_path[model_path.rfind("/") + 1:]
results_folder = run_folder + "results_" + model_name[:model_name.rfind(".")] + "/" + time.strftime("%Y%m%d-%H%M%S/")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
print("write to", results_folder)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path
assert img_path[-1] == '/'


def format_img_size(img, C):
    """ formats the image size based on config """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape
    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # print("ratio=", ratio)

    return img, ratio


def format_img_channels(img, C):
    """ formats the image channels based on config """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2, bb3d):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    real_bb3d = [int(round(v // ratio)) for v in bb3d]

    return (real_x1, real_y1, real_x2, real_y2, real_bb3d)


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)
class_mapping = {v: k for k, v in class_mapping.items()}

if options.num_rois:
    C.num_rois = int(options.num_rois)

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (1024, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.5
rpn_overlap_trashhold = 0.5

visualise = True

options = {"model": "./darkflow/cfg/yolo.cfg", "load": "./darkflow/bin/yolo.weights", "threshold": 0.1}
tfnet = TFNet(options)


for idx, img_name in enumerate(sorted(os.listdir(img_path))):
    if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
        continue
    filepath = os.path.join(img_path, img_name)

    full_img = cv2.imread(filepath)
    full_img_cpy = full_img.copy()
    predictions = tfnet.return_predict(full_img)

    frcnn_counter_not_found = 0
    for pred in predictions:
        if pred['label'] != 'car' or pred['confidence'] < 0.5:
            continue

        tl = pred['topleft']
        br = pred['bottomright']
        height, width = full_img.shape[:2]
        
        x_min = max(tl['x'] - margin, 0)
        x_max = min(br['x'] + margin, width)
        y_min = max(tl['y'] - margin, 0)
        y_max = min(br['y'] + margin, height)

        img = full_img[y_min:y_max, x_min:x_max]
        cv2.rectangle(full_img_cpy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image',img)
        # cv2.waitKey(0)

        st = time.time()

        X, ratio = format_img(img, C)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    regr_result = P_regr[0, ii, 20 * cls_num:20 * (cls_num + 1)]

                    tx = regr_result[0]
                    ty = regr_result[1]
                    tw = regr_result[2]
                    th = regr_result[3]
                    bb3d = regr_result[4:]

                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    bb3d_x = [v / C.classifier_regr_std[2] for v in bb3d[:8]]
                    bb3d_y = [v / C.classifier_regr_std[3] for v in bb3d[8:]]

                    x, y, w, h, bb3d_regr = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th, bb3d_x + bb3d_y)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)] + [
                        C.rpn_stride * (x + w - v) for v in bb3d_regr[:8]] + [C.rpn_stride * (y + h - v) for v in
                                                                              bb3d_regr[8:]])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []
        bbs_real = []

        top_bbox = None
        top_cls = None
        top_prob = 0
        for key in bboxes:
            bboxes_cls = np.array(bboxes[key])
            # print('Found {} BBs before NMS'.format(len(bbox)))

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bboxes_cls, np.array(probs[key]),
                                                                        overlap_thresh=rpn_overlap_trashhold)
            # print('Found {} BBs after NMS'.format(len(new_boxes)))

            if len(new_boxes) > 0 and new_probs[0] > top_prob:
                top_bbox = new_boxes[0, :]
                top_cls = key
                top_prob = new_probs[0]
        if top_bbox is None:
            frcnn_counter_not_found += 1
            continue
        key = top_cls
        res = top_bbox

        img = cv2.putText(full_img_cpy, "{0:0.2f}".format(top_prob), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                          (0, 255, 0), 2)

        x1 = res[0]
        y1 = res[1]
        x2 = res[2]
        y2 = res[3]
        bb3d = res[4:]

        (real_x1, real_y1, real_x2, real_y2, real_bb3d) = get_real_coordinates(ratio, x1, y1, x2, y2, bb3d)

        real_x1 += x_min
        real_x2 += x_min
        real_y1 += y_min
        real_y2 += y_min
        real_bb3d = [x + x_min for x in real_bb3d[:8]] + [y + y_min for y in real_bb3d[8:]]
        # print(real_x1, real_y1, real_x2, real_y2, real_bb3d)
        # exit()

        ############## FIND these POINTS by WEIGHING the PROPOSALS ##########################
        #
        #
        #       (0,0) >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  >  (x,0)
        #         v
        #         v          3 ~~~~~~~~~~~
        #         v        /        top    ~~~~~~~~~~~~~ 4
        #         v     2 ~~~~~~~~~~~~               / f |
        #         v     |              ~~~~~~~~~~~~ 1  r |
        #         v     |                           |  o |
        #         v     |           side            |  n |
        #         v     |     7                     |  t |
        #         v     |                           |    8
        #         v     6  ~~~~~~~~~~~              |   /
        #         v                    ~~~~~~~~~~~~ 5
        #       (0,y)
        #

        # P1 - P2
        cv2.line(full_img_cpy, (real_bb3d[0], real_bb3d[8]), (real_bb3d[1], real_bb3d[9]), (0, 0, 0), 1, cv2.LINE_AA)
        # P1 - P4
        cv2.line(full_img_cpy, (real_bb3d[0], real_bb3d[8]), (real_bb3d[3], real_bb3d[11]), (0, 0, 0), 1, cv2.LINE_AA)
        # P1 - P5
        cv2.line(full_img_cpy, (real_bb3d[0], real_bb3d[8]), (real_bb3d[4], real_bb3d[12]), (0, 0, 0), 1, cv2.LINE_AA)
        # P2 - P3
        cv2.line(full_img_cpy, (real_bb3d[1], real_bb3d[9]), (real_bb3d[2], real_bb3d[10]), (0, 0, 0), 1, cv2.LINE_AA)
        # P2 - P6
        cv2.line(full_img_cpy, (real_bb3d[1], real_bb3d[9]), (real_bb3d[5], real_bb3d[13]), (0, 0, 0), 1, cv2.LINE_AA)
        # P3 - P4
        cv2.line(full_img_cpy, (real_bb3d[2], real_bb3d[10]), (real_bb3d[3], real_bb3d[11]), (0, 0, 0), 1, cv2.LINE_AA)
        # P3 - P7
        cv2.line(full_img_cpy, (real_bb3d[2], real_bb3d[10]), (real_bb3d[6], real_bb3d[14]), (0, 0, 0), 1, cv2.LINE_AA)
        # P4 - P8
        cv2.line(full_img_cpy, (real_bb3d[3], real_bb3d[11]), (real_bb3d[7], real_bb3d[15]), (0, 0, 0), 1, cv2.LINE_AA)
        # P5 - P6
        cv2.line(full_img_cpy, (real_bb3d[4], real_bb3d[12]), (real_bb3d[5], real_bb3d[13]), (0, 0, 0), 1, cv2.LINE_AA)
        # P5 - P8
        cv2.line(full_img_cpy, (real_bb3d[4], real_bb3d[12]), (real_bb3d[7], real_bb3d[15]), (0, 0, 0), 1, cv2.LINE_AA)
        # P6 - P7
        cv2.line(full_img_cpy, (real_bb3d[5], real_bb3d[13]), (real_bb3d[6], real_bb3d[14]), (0, 0, 0), 1, cv2.LINE_AA)
        # P7 - P8
        cv2.line(full_img_cpy, (real_bb3d[6], real_bb3d[14]), (real_bb3d[7], real_bb3d[15]), (0, 0, 0), 1, cv2.LINE_AA)

        # Draw points belonging to front in red, draw points belonging to back in blue
        front = [0, 1, 4, 5] if key == "Front" else [2, 3, 6, 7]
        back = [2, 3, 6, 7] if key == "Front" else [0, 1, 4, 5]
        for p in front:
            cv2.circle(full_img_cpy, (real_bb3d[p], real_bb3d[p + 8]), 1, (0, 0, 255), 3)

        for p in back:
            cv2.circle(full_img_cpy, (real_bb3d[p], real_bb3d[p + 8]), 1, (255, 0, 0), 3)

        bbs_real.append({"class": key, "prob": top_prob,
                         "x1": real_x1, "y1": real_y1, "x2": real_x2, "y2": real_y2})
        textLabel = '{}: {}'.format(key, int(100 * top_prob))
        all_dets.append((key, 100 * top_prob))

         # if options.print_classes:
            # (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            # textOrg = (real_x1, real_y1 - 0)

            # cv2.rectangle(full_img_cpy, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          # (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0), 2)
            # cv2.rectangle(full_img_cpy, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                          # (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (255, 255, 255), -1)
            # cv2.putText(full_img_cpy, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        print(all_dets, )
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        cv2.imwrite(results_folder + '{}.png'.format(img_name), full_img_cpy)
    print("finished", img_name, len(predictions), "Yolo Predictions ->", frcnn_counter_not_found, "not used")