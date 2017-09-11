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
    """
    :param pred:
    :param gt:
    :param f:
    :return: T, P dict with each class name as key
            P pred probabilities list
    """
    T = {}
    P = {}
    # metric_mse = []
    # metric_dist = []
    fx, fy = f

    for bbox in gt:
        bbox['bbox_matched'] = False

    pred_probs = np.array([s['prob'] for s in pred])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

    for box_idx in box_idx_sorted_by_prob:
        # iterate over all bounding boxes sorted by probability, the highest with an overlap of
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

        best_dist3d = float('inf')
        best_dist3d_gt_id = -1
        best_mse3d = float('inf')
        best_mse3d_gt_id = -1

        # trying to find a matching GT for this found bounding box
        for gt_id, gt_box in enumerate(gt):
            gt_class = gt_box['class']
            gt_3dd = copy.deepcopy(gt_box)
            keys_x = ['x1', 'x2', 'bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8']
            keys_y = ['y1', 'y2', 'bb_y1', 'bb_y2', 'bb_y3', 'bb_y4', 'bb_y5', 'bb_y6', 'bb_y7', 'bb_y8']
            # gt_x1 = gt_box['x1'] / fx
            # gt_x2 = gt_box['x2'] / fx
            # gt_y1 = gt_box['y1'] / fy
            # gt_y2 = gt_box['y2'] / fy
            for key_x in keys_x:
                gt_3dd[key_x] /= fx
            for key_y in keys_y:
                gt_3dd[key_y] /= fy

            gt_seen = gt_box['bbox_matched']
            if gt_class != pred_class:
                continue
            if gt_seen:
                continue  # TODO: why can this occur?
            # iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
            iou = data_generators.iou((pred_box['x1'], pred_box['y1'], pred_box['x2'], pred_box['y2']),
                                      (gt_3dd['x1'], gt_3dd['y1'], gt_3dd['x2'], gt_3dd['y2']))
            # dist3d = data_generators.dist3d(pred_box, gt_3dd, gt_3dd['x2'] - gt_3dd['x1'], gt_3dd['y2']-gt_3dd['y1'])
            # mse3d = data_generators.mse3d(pred_box, gt_3dd)
            #
            # if dist3d < best_dist3d:
            #     best_dist3d = dist3d
            #     best_dist3d_gt_id = gt_id
            # if mse3d < best_mse3d:
            #     best_mse3d = mse3d
            #     best_mse3d_gt_id = gt_id

            if iou >= 0.5:
                found_match = True
                gt_box['bbox_matched'] = True
                break
            else:
                continue
        T[pred_class].append(int(found_match))
        # metric_dist.append((best_dist3d_gt_id, best_dist3d))
        # metric_mse.append((best_mse3d_gt_id, best_mse3d))

    for gt_box in gt:
        if not gt_box['bbox_matched']:  #TODO there was "and not gt_box['difficult']:" before, why?
            if gt_box['class'] not in P:
                P[gt_box['class']] = []
                T[gt_box['class']] = []

            T[gt_box['class']].append(1)
            P[gt_box['class']].append(0.0)  # this was recognized with a probability of 0.0

    # import pdb
    # pdb.set_trace()
    return T, P  # , metric_dist, metric_mse


sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="run_folder", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
                  help="Number of ROIs per iteration. Higher means more memory use.", default=32)
parser.add_option("--config_filename", dest="config_filename", help=
"Location to read the metadata related to the training (generated when training).",
                  default="config.pickle")
parser.add_option("-m", "--model", dest="model",
                  help="Name of the model that shall be loaded")
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="simple"),

(options, args) = parser.parse_args()

# if not options.test_path:  # if filename is not given
#     parser.error('Error: path to test data must be specified. Pass --path to command line')
if not options.run_folder:  # if filename is not given
    run_folder = helper.chose_from_folder("runs/", "*", "--path")
else:
    run_folder = options.run_folder

if options.model:
    model_path = options.model
else:
    model_path = helper.chose_from_folder(run_folder, "*.hdf5", "--model")
config_output_filename = run_folder + "config.pickle"
print("Specified Model for Testing:", model_path)

model_name = model_path[model_path.rfind("/") + 1:]
results_folder = os.path.join(run_folder, "results_" + model_name[:model_name.rfind(".")] + "/")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
print("write to", results_folder)


if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

config_output_filename = os.path.join(run_folder, options.config_filename)

with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False


def format_img(img_c, C):
    img = np.copy(img_c)
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
# TODO: remove this
splits = None
C.train_path = "annotations/testfile_small.txt"
all_imgs, _, _ = get_data(C.train_path, splits, test_only=True)
test_imgs = [v for s,v in all_imgs.items() if v['imageset'] == 'test']

T = {}
P = {}
MSE = {}
DIST = {}
for idx, img_data in enumerate(test_imgs):
    print('test image {}/{}:'.format(idx, len(test_imgs)))
    st = time.time()
    filepath = img_data['filepath']
    img_name = filepath[filepath.rfind("/")+1:]

    img = cv2.imread(filepath)

    X, fx, fy = format_img(img, C)  # fx, fy are the scale factors for width and height for image X

    for bbox in img_data['bboxes']:
        points = [
            'x1', 'y1', 'x2', 'y2',
            'bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8',
            'bb_y1', 'bb_y2', 'bb_y3', 'bb_y4', 'bb_y5', 'bb_y6', 'bb_y7', 'bb_y8'
        ]
        img = helper.draw_annotations(img, [bbox[point] for point in points], data_format="3d_reg",
                                      fac=bbox['class'], numbers=False)
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

    for cls in bboxes:
        bbox = np.array(bboxes[cls])

        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[cls]), overlap_thresh=0.5)
        for jk in range(new_boxes.shape[0]):
            res = new_boxes[jk, :]
            points = [
                'x1', 'y1', 'x2', 'y2',
                'bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8',
                'bb_y1', 'bb_y2', 'bb_y3', 'bb_y4', 'bb_y5', 'bb_y6', 'bb_y7', 'bb_y8'
            ]
            # det = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': key, 'prob': new_probs[jk]}
            det = {points: res[idx] for idx, points in enumerate(points)}
            det['class'] = cls
            det['prob'] = new_probs[jk]
            all_dets.append(det)

    print('Elapsed time = {}'.format(time.time() - st))
    t, p = get_map(all_dets, img_data['bboxes'], (fx, fy))

    # sort indexes by predicted probabilites (is also done like that in get_map)
    pred_probs = np.array([s['prob'] for s in all_dets])
    box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]
    _, ratio = helper.format_img(img, C)

    for box_idx in box_idx_sorted_by_prob:
        points3dxy = ['bb_x1', 'bb_x2', 'bb_x3', 'bb_x4', 'bb_x5', 'bb_x6', 'bb_x7', 'bb_x8',
                      'bb_y1', 'bb_y2', 'bb_y3', 'bb_y4', 'bb_y5', 'bb_y6', 'bb_y7', 'bb_y8']
        bbox = all_dets[box_idx]
        (real_x1, real_y1, real_x2, real_y2, real_bb3d) = helper.get_real_coordinates(
            ratio, bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2'],
            [bbox[k] for k in points3dxy]

        )
        bb_real = {'class': bbox['class'], 'x1': real_x1, 'y1': real_y1, 'x2': real_x2, 'y2': real_y2}
        for i, k in enumerate(points3dxy):
            bb_real[k] = real_bb3d[i]
        colors = [(0, 0, 255),  # red           0
                  (0, 255, 255),  # yellow      1
                  (255, 255, 255),  # white     2
                  (255, 255, 0),  # cyan        3
                  (255, 0, 0),  # blue          4
                  (0, 0, 0),  # black           5
                  (0, 255, 0),  # green         6
                  (255, 0, 255),  # magenta     7
                  ]
        for point in range(8):
            cv2.circle(img, (real_bb3d[point], real_bb3d[point + 8]), 1, colors[point], 3)

        # P1 - P2
        cv2.line(img, (real_bb3d[0], real_bb3d[8]), (real_bb3d[1], real_bb3d[9]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P1 - P4
        cv2.line(img, (real_bb3d[0], real_bb3d[8]), (real_bb3d[3], real_bb3d[11]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P1 - P5
        cv2.line(img, (real_bb3d[0], real_bb3d[8]), (real_bb3d[4], real_bb3d[12]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P2 - P3
        cv2.line(img, (real_bb3d[1], real_bb3d[9]), (real_bb3d[2], real_bb3d[10]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P2 - P6
        cv2.line(img, (real_bb3d[1], real_bb3d[9]), (real_bb3d[5], real_bb3d[13]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P3 - P4
        cv2.line(img, (real_bb3d[2], real_bb3d[10]), (real_bb3d[3], real_bb3d[11]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P3 - P7
        cv2.line(img, (real_bb3d[2], real_bb3d[10]), (real_bb3d[6], real_bb3d[14]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P4 - P8
        cv2.line(img, (real_bb3d[3], real_bb3d[11]), (real_bb3d[7], real_bb3d[15]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P5 - P6
        cv2.line(img, (real_bb3d[4], real_bb3d[12]), (real_bb3d[5], real_bb3d[13]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P5 - P8
        cv2.line(img, (real_bb3d[4], real_bb3d[12]), (real_bb3d[7], real_bb3d[15]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P6 - P7
        cv2.line(img, (real_bb3d[5], real_bb3d[13]), (real_bb3d[6], real_bb3d[14]), colors[box_idx%8], 1, cv2.LINE_AA)
        # P7 - P8
        cv2.line(img, (real_bb3d[6], real_bb3d[14]), (real_bb3d[7], real_bb3d[15]), colors[box_idx%8], 1, cv2.LINE_AA)

        # we have current bounding box bb_real, want metrics:
        # - dist to best gt (normalize by gt)
        # - mse  to best gt (normalize by gt)
        # - was ground truth detected at all? (use their metrics)

        best_dist = float("inf")
        best_dist_idx = None
        for gt_idx, bb_gt in enumerate(img_data['bboxes']):
            dist = data_generators.dist3d(bb_gt, bb_real, bb_gt['x2']-bb_gt['x1'], bb_gt['y2']-bb_gt['y1'])
            if dist < best_dist:
                best_dist = dist
                best_dist_idx = gt_idx

        print("Best dist for bb#", box_idx, "to gt#", best_dist_idx, "with:", best_dist)

        if best_dist_idx is not None:
            mean_gtx, mean_gty = data_generators.mean3d(img_data['bboxes'][best_dist_idx])
            mean_rx, mean_ry = data_generators.mean3d(bb_real)
            cv2.line(img, (int(mean_gtx), int(mean_gty)), (int(mean_rx), int(mean_ry)),
                     colors[box_idx%8], 1, cv2.LINE_AA)
            img = cv2.putText(img, "{0:0.2f}".format(best_dist), (int(mean_rx), int(mean_ry)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                              colors[box_idx % 8], 2)

        best_mse = float('inf')
        best_mse_idx = None
        for gt_idx, bb_gt in enumerate(img_data['bboxes']):
            mse = data_generators.mse3d(bb_gt, bb_real)
            if mse < best_mse:
                best_mse = mse
                best_mse_idx = gt_idx

        print("Best MSE for bb#", box_idx, "to gt#", best_mse_idx, "with:", best_mse)

        if bb_real['class'] not in DIST:
            DIST[bb_real['class']] = []
        if bb_real['class'] not in MSE:
            MSE[bb_real['class']] = []

        DIST[bb_real['class']].append(best_dist)
        MSE[bb_real['class']].append(best_mse)

    # print("Distances:", metric_dist)
    # print("MSE:", metric_mse)

    for cls in t.keys():
        if cls not in T:
            T[cls] = []
            P[cls] = []
        # add new entries too all classes that had to be recognized.
        T[cls].extend(t[cls])
        P[cls].extend(p[cls])
    all_aps = []
    s = ""
    for cls in T.keys():
        ap = average_precision_score(T[cls], P[cls])
        s += ', {} AP: {}'.format(cls, ap)
        all_aps.append(ap)
    print('mAP = {}'.format(np.mean(np.array(all_aps))), s)

    s = ""
    all_dist = []
    for cls in MSE.keys():
        s += ', {} DIST: {}'.format(cls, np.mean(DIST[cls]))
        all_dist.extend(DIST[cls])
    print('DIST: {}'.format(np.mean(all_dist)), s)

    s = ""
    all_mse = []
    for cls in MSE.keys():
        s += ', {} MSE: {}'.format(cls, np.mean(MSE[cls]))
        all_mse.extend(MSE[cls])
    print('MSE: {}'.format(np.mean(all_mse)), s)

    # print(T)
    # print(P)
    img_path = results_folder + '{}.png'.format(img_name)
    print("write img to", img_path)
    cv2.imwrite(img_path, img)


