from __future__ import division
import random
import pprint
import sys
import traceback
import time
import numpy as np
from optparse import OptionParser
import os
import pickle
import json


from keras import backend as K
from keras.optimizers import Adam  # , SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet as nn
import keras_frcnn.roi_helpers as roi_helpers
import scripts.helper as helper
from keras.utils import generic_utils
# from keras.callbacks import TensorBoard


sys.setrecursionlimit(40000)
parser = OptionParser()

C = config.create_config_read_parser(parser)

if C.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif C.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")
print("train_frcnn.py", "--path", C.train_path, "--frcnn_weights", C.load_model,
      "--num_epochs", C.num_epochs, "--epoch_length", C.epoch_length, "--save_every", C.save_every,
      "--num_rois", C.num_rois)
splits = None
# if the splits already exist, we will load them in
# delete this file if you are a different amount of pictures now
if os.path.exists(C.output_folder + "splits.pickle"):
    with open(C.output_folder + "splits.pickle", 'rb') as splits_f:
        splits = pickle.load(splits_f)

all_imgs_dict, classes_count, class_mapping = get_data(
    C.train_path, train_test_split=splits)
all_imgs = []
for key in all_imgs_dict:
    all_imgs.append(all_imgs_dict[key])

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping
C.classes_count = classes_count
# inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)
# returns: rpn <- [x_class, x_regr, base_layers], base_layers is not used below

# will be filled
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)
# returns: classifier <- [out_class, out_regr]

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

logger = helper.Logger(C.output_folder, "log.txt")
sample_logger = helper.Logger(C.output_folder, "samples.csv")
res_logger = helper.Logger(C.output_folder, "results.csv")
res_logger.log(
    "Epoch #,Classification Accuracy,Mean # of BB from RPN overlapping with ground truthboxes,Former best Loss,"
    "Total Loss,Loss RPN Classifier,Loss RPN Regression,Loss Classifier-Net Classification,"
    "Loss Classifier-Net Regression,Epoch Time\n"
)
# TC = TensorBoard(log_dir=log_path)
# TC.set_model(model_classifier)

# this is a model that holds both the RPN and the classifier, used to load/save weights for the models
# is not trained itself!
model_all = Model([img_input, roi_input], rpn[:2] + classifier)
if C.load_model is not None:

    print("Reload already trained model", C.load_model)  # C.load_model already includes run path
    model_rpn.load_weights(C.load_model, by_name=True)
    model_classifier.load_weights(C.load_model, by_name=True)
else:
    try:
        print('loading base net weights from {}'.format(C.base_net_weights))
        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights from {}. Weights can be found at {} and {}'.format(
            C.base_net_weights,
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/'
            'resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/'
            'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))

optimizer = Adam(lr=1e-4)
optimizer_classifier = Adam(lr=1e-4)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                         metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

with open(C.output_folder + "splits.pickle", 'wb') as splits_f:
    splits = {filename: all_imgs_dict[filename]['imageset'] for filename in all_imgs_dict}
    if C.verbose:
        print("saving splits file with", len(splits), "entries")
    pickle.dump(splits, splits_f, protocol=2)
del all_imgs_dict

iter_num = 0
losses = np.zeros((C.epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
log_collector = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training', "verbose" if C.verbose else "")

for epoch_num in range(C.current_epoch, C.num_epochs):

    progbar = generic_utils.Progbar(C.epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, C.num_epochs))

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == C.epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, C.epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes.'
                          'Check RPN settings or keep training.')

            X, Y, img_data = next(data_gen_train)
            # X: x_img, zero centered
            # Y: [y_rpn_cls=[y_is_box_valid, y_rpn_overlap],
            #     y_rpn_regr=[4*y_rpn_overlap, y_rpn_regr]],
            # an anchor position is valid if IoU <0.3 | >0.7 and overlapping if IoU >0.7 with a GT box (values 0 | 1)
            # the regression values are the the closest object.
            # img_data: img_data_aug
            # shape (1,38,67,18)
            loss_rpn = model_rpn.train_on_batch(X, Y)

            [Y1_rpn_pred, Y2_rpn_pred] = model_rpn.predict_on_batch(X)
            # predictions for all anchor positions&shapes, (if stride>1, ignore the "bottom right" values)
            #       Y1_rpn_pred: predicted prob this is and is not an object: (1, 2*num_anchors, img_width, img_height)
            #       Y2_rpn_pred: predicted regression to box edges to closest ground truth bounding box

            R = roi_helpers.rpn_to_roi(Y1_rpn_pred, Y2_rpn_pred, C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,
                                       max_boxes=300)
            # R = [[x,y,w,h], [x,y,h,h],...] anchor boxes that were within the image (some fall out if stride>1)
            #                                and that NMS allowed to be so close to each other (np array)

            X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)
            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            # X2: ROIs with coordinates resized(x,y,w,h) * C.classifier_regr_std
            # Y1: ROIs with target class
            # Y2: ROIs with 20 regression values for each point resized(center point) * C.classifier_regr_std)

            if X2 is None:
                # best IoU of each RoI is below threshold C.classifier_min_overlap
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue  # if no regions are found train again.

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            # TODO: figure out, what is a good number of ROIs to be looked at at the same time,
            # TODO: standard value was 32, but could also be 4?
            # We have to feed in num_rois samples (of ROIs), ideally half of them should be positive and half negative.
            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois // 2:
                    # if there are not even num_rois/2 positive samples we will take at least all the ones we have
                    selected_pos_samples = pos_samples.tolist()

                else:
                    # if there are more than num_rois/2 positive samples, we can pick at random
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()


                try:
                    # we will try to fill up the selected samples with negative ones to have num_rois in total
                    # first try to draw them from neg_samples where each can only be selected once ("ohne zurücklegen")
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    try:
                        # if that fails draw from the negative samples where each element
                        # can be selected more than once ("mit zurücklegen")
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()
                    except:
                        # TODO: `neg_samples` was empty
                        # TODO: File "mtrand.pyx", line 1121, in mtrand.RandomState.choice
                        # TODO: (numpy/random/mtrand/mtrand.c:17200) ValueError: a must be non-empty
                        # this means there are no negative samples at all in this picture, if this case passes through
                        # the net will throw an error as the input won't have num_rois samples to work with.
                        selected_neg_samples = []
                        # assert False

                sel_samples = selected_pos_samples + selected_neg_samples

            else:
                # in the extreme case if num_rois is set to 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2) or len(selected_pos_samples) == 0:
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            log_entry = [
                ("Imagepath", img_data['filepath']),
                ("Img_W", img_data['width']),
                ("Img_H", img_data['height']),
                ("#BBoxes", len(img_data['bboxes'])),
                ("#ROI anchors", R.shape[0]),
                ("Sel. Samples", len(sel_samples)),
                ("Sel. Pos. Samples", len(selected_pos_samples)),
                ("from all Pos Samples", len(pos_samples)),
                ("Sel. Neg. Samples Shape", len(selected_neg_samples)),
                ("from all Neg Samples", len(neg_samples))
            ]
            log_collector.append(log_entry)

            class_color = {(1, 0): (50, 50, 50), (0, 1): (200, 200, 200)}
            img = np.copy(X)
            # for sel in sel_samples:
            #     helper.draw_annotations(img[0, :, :, :], Y2[0, sel, 20:], X2[0, sel, :],
            #                             class_color[tuple(Y1[0, sel, :].tolist())])
            # cv2.imwrite("train.png", img)

            loss_class = model_classifier.train_on_batch(
                x=[X, X2[:, sel_samples, :]],
                y=[Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
            )
            # TODO: got error here on 'y=' input:
            # TODO: alueError: Error when checking model input: expected input_2 to have shape (None, 32, 4)
            # TODO: but got array with shape (1, 2, 4)
            # --> not enough samples selected???

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            #
            # x_roi         y_cls, y_reg
            #
            # X->    RPN -->  Y             X  ->   CLASS  ->   Y1     y_class_num
            # |               |             X2 ->    NET   ->   Y2     [y_class_reg_label, y_class_reg_coords]
            # |               |
            # -------------------helper--->X2,Y1,Y2

            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3]))])

            # end of epoch, will break the while true loop in this if statement.
            if iter_num == C.epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                # for X, Y, img_data in data_gen_val:
                #
                #     outs = test_helper.test_picture(X, C, model_rpn=model_rpn, model_classifier_only=model_classifier,
                #                              visual_output=False, class_mapping=class_mapping)
                #     print ("got:", outs)
                #     print ("target:", Y)
                #     break

                elapsed_time = time.time() - start_time
                # if C.verbose:
                #     print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                #         mean_overlapping_bboxes))
                #     print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                #     print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                #     print('Loss RPN regression: {}'.format(loss_rpn_regr))
                #     print('Loss Detector classifier: {}'.format(loss_class_cls))
                #     print('Loss Detector regression: {}'.format(loss_class_regr))
                #     print('Elapsed time: {}'.format(elapsed_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    model_path = C.output_folder + C.model_name
                    if C.verbose:
                        print('Total loss decreased from {} to {}, \nsaving weights to {}'.format(
                            best_loss, curr_loss, model_path))
                    best_loss = curr_loss
                    model_all.save_weights(model_path)
                try:
                    if epoch_num % C.save_every == 0 and epoch_num != 0:
                        checkpoint_path = C.output_folder + "model_frcnn" + str(epoch_num) + ".hdf5"
                        print("saving weights of epoch:", epoch_num, "to", checkpoint_path)
                        model_all.save_weights(checkpoint_path)
                except:
                    e = "couldn't save checkpoint, did you specify 'save_every' in config?"
                    print(e)
                    logger.log(e)
                    logger.log("\n---\n")
                    pass
                log = [
                    ("Epoch #", epoch_num),
                    ("Classification Accuracy", class_acc),
                    ("Mean # of BB from RPN overlapping with ground truthboxes", mean_overlapping_bboxes),
                    ("Former best Loss", best_loss),
                    ("Total Loss", curr_loss),
                    ('Loss RPN Classifier', loss_rpn_cls),
                    ("Loss RPN Regression", loss_rpn_regr),
                    ("Loss Classifier-Net Classification", loss_class_cls),
                    ("Loss Classifier-Net Regression", loss_class_regr),
                    ("Epoch Time", elapsed_time)
                ]
                if C.verbose:
                    print(log)
                    for log_entry in log_collector:
                        s = ""
                        for name, val in log_entry:
                            s += str(val) + ","
                        s += "\n"
                        sample_logger.log(s)
                    s = ""
                    for name, val in log:
                        s += str(val) + ","
                    s+= "\n"
                    res_logger.log(s)

                logger.log(log)
                log_collector = []
                C.stats.append((epoch_num, log))
                # TC.on_epoch_end(epoch_num, {'acc': class_acc, 'loss': loss_class_cls})
                #                TC.on_epoch_end(epoch_num, log)
                break

        except Exception as e:
            tr = traceback.format_exc()

            try:
                for name, val in log_entry:
                    tr += name + ": " + str(val) + ", "
                tr += "\n----\n"
            except:
                # selected samples might not be defined at the call of the exception
                pass

            logger.log(tr)
            print("WARNING: An EXCEPTION occured in the main loop of training")
            print(tr)

            continue

    # with open(C.output_folder+"epoch.txt", 'w') as epoch_f:
    #    epoch_f.write(epoch_num)
    C.current_epoch = epoch_num + 1
    config_output_filename = C.output_folder + C.config_filename
    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C, config_f, protocol=2)  # pickle version 2 so can be read by Python 2
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            config_output_filename))
    with open(C.output_folder + "config.json", 'w') as configjson:
        json.dump(vars(C), configjson)

# TC.on_train_end()
input('Training complete, press enter to exit')
