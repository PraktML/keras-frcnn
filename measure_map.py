import os
import cv2
import numpy as np
import sys
import copy
import pickle
from optparse import OptionParser
import time
import scripts.helper as helper
from keras_frcnn import config
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from keras_frcnn import data_generators
from sklearn.metrics import average_precision_score


def get_map(pred, gt, f):
    T = {}
    P = {}
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        pred_box = pred[box_idx]
        pred_class = pred_box['class']
        # pred_x1 = pred_box['x1']
        # pred_x2 = pred_box['x2']
        # pred_y1 = pred_box['y1']
        # pred_y2 = pred_box['y2']
        pred_prob = pred_box['prob']
        if pred_class not in P:
            P[pred_class] = []
            T[pred_class] = []
        P[pred_class].append(pred_prob)
        found_match = False

        for gt_box in gt:
            gt_class = gt_box['class']
            gt_3dd = copy.deepcopy(gt_box)
            keys_x = ['x1', 'x2', 'bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8']
            keys_y = ['y1', 'y2', 'bb_y1', 'bb_y2', 'bb_y3', 'bb_y4', 'bb_y5', 'bb_y6', 'bb_y7', 'bb_y8']
            # gt_x1 = gt_box['x1'] / fx
            # gt_x2 = gt_box['x2'] / fx
            # gt_y1 = gt_box['y1'] / fy
            # gt_y2 = gt_box['y2'] / fy
            for key_x in keys_x:
                gt_3dd[keys_x] /= fx
            for key_y in keys_y:
                gt_3dd[keys_y] /= fy

            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue  # TODO: why can this occur?
            # iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            iou = data_generators.iou((pred_box['x1'], pred_box['y1'], pred_box['x2'], pred_box['y2']),
                                      (gt_3dd['x1'], gt_3dd['y1'], gt_3dd['x2'], gt_3dd['y2']))
            iou3d = data_generators.iou3d(pred_box, gt_3dd)
            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break  # TODO: delete break here, if we want to check for all.
            else:
                continue

        T[pred_class].append(int(found_match))

    for gt_box in gt:
        if not gt_box['bbox_matched']:  #TODO there was "and not gt_box['difficult']:" before, why?
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0)

    # import pdb
    # pdb.set_trace()
    return T, P


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("-m", "--model", dest="model", default="model_frcnn.hdf5",
                  help="Name of the model that shall be loaded")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="simple"),

(options, args) = parser.parse_args()

# if not options.test_path:  # if filename is not given
#     parser.error('Error: path to test data must be specified. Pass --path to command line')
if not options.test_path:  # if filename is not given
    run_path = helper.chose_from_folder("runs/", "*", "--path")
else:
    run_path = options.test_path

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

config_output_filename = os.path.join(run_path, options.config_filename)

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False




def format_img(img, C):
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    fx = width / float(new_width)
    fy = height / float(new_height)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, fx, fy


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
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

model_path = os.path.join(run_path, options.model)
model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')


splits = None
# if the splits already exist, we will load them in
# delete this file if you are a different amount of pictures now
if os.path.exists(C.output_folder + "splits.pickle"):
    with open(C.output_folder + "splits.pickle", 'rb') as splits_f:
        splits = pickle.load(splits_f)
# C.train_path contains the path to the annotation file (a saved version is also stored in the run folder)
all_imgs, _, _ = get_data(C.train_path, splits)
test_imgs = [v for s,v in all_imgs.items() if v['imageset'] == 'test']

T = {}
P = {}
for idx, img_data in enumerate(test_imgs):
    print('test image {}/{}:'.format(idx, len(test_imgs)))
    st = time.time()
    filepath = img_data['filepath']

    img = cv2.imread(filepath)

    X, fx, fy = format_img(img, C)  # fx, fy are the scale factors for width and height for image X

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

            if np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
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
            # bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
            bboxes[cls_name].append(
                [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)] + [
                    C.rpn_stride * (x + w - v) for v in bb3d_regr[:8]] + [C.rpn_stride * (y + h - v) for v in
                                                                          bb3d_regr[8:]])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = []

    for key in bboxes:
        bbox = np.array(bboxes[key])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            res = new_boxes[jk, :]
            points = ['x1', 'y1', 'x2', 'y2',
                    'bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8',
                    'bb_y1', 'bb_y2', 'bb_y3', 'bb_y4', 'bb_y5', 'bb_y6', 'bb_y7', 'bb_y8'
                    ]
            # det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
            det = {points: res[idx] for idx, points in enumerate(points)}
            det['class'] = key
            det['prob'] = new_probs[jk]
            all_dets.append(det)

    print('Elapsed time = {}'.format(time.time() - st))
    t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))
    for key in t.keys():
        if key not in T:
            T[key] = []
            P[key] = []
        T[key].extend(t[key])
        P[key].extend(p[key])
    all_aps = []
    for key in T.keys():
        ap = average_precision_score(T[key], P[key])
        print('{} AP: {}'.format(key, ap))
        all_aps.append(ap)
    print('mAP = {}'.format(np.mean(np.array(all_aps))))
    # print(T)
    # print(P)
