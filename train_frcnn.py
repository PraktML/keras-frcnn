from __future__ import division
import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import os
import pickle
import json
import scripts.settings

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet as nn
import keras_frcnn.roi_helpers as roi_helpers
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

log_path = C.output_folder + "logs/"
if not os.path.exists(log_path):
    os.makedirs(log_path)
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
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
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
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, img_data = next(data_gen_train)  # x_img, [y_rpn_cls, y_rpn_regr], img_data_aug
            # shape            (1,38,67,18)
            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)

            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,
                                       max_boxes=300)

            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
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

            if C.num_rois > 1:
                if len(pos_samples) < C.num_rois // 2:  # usually half are positive and half are negative examples
                    selected_pos_samples = pos_samples.tolist()
                else:

                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                try:  # TODO: `neg_samples` was empty   File "mtrand.pyx", line 1121, in mtrand.RandomState.choice (numpy/random/mtrand/mtrand.c:17200) ValueError: a must be non-empty
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                                replace=True).tolist()
                    except:
                        selected_neg_samples = []
                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                         [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            #
            #           x_roi         y_cls, y_reg
            #
            #             X->    RPN -->  Y             X  ->   CLASS  ->   Y1     y_class_num
            #             |               |             X2 ->    NET   ->   Y2     [y_class_reg_label, y_class_reg_coords]
            #             |               |
            #             -------------------helper--->X2,Y1,Y2

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
                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(elapsed_time))

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
                    if epoch_num % C.save_every == 0:
                        checkpoint_path = C.output_folder + "model_frcnn" + str(epoch_num) + ".hdf5"
                        print("saving weights of epoch:", epoch_num, "to", checkpoint_path)
                        model_all.save_weights(checkpoint_path)
                except:
                    print("couldn't save checkpoint, did you specify 'save_every' in config?")
                    pass
                log = {
                    "Classification Accuracy": class_acc,
                    "Mean # of BB from RPN overlapping with ground truthboxes": mean_overlapping_bboxes,
                    "Total Loss": curr_loss,
                    'Loss RPN Classifier': loss_rpn_cls,
                    "Loss RPN Regression": loss_rpn_regr,
                    "Loss Classifier-Net Classification": loss_class_cls,
                    "Loss Classifier-Net Regression": loss_class_regr,
                    "Epoch #": epoch_num,
                    "Epoch Time": elapsed_time,
                }
                print(log)
                C.stats.append((epoch_num, log))
                # TC.on_epoch_end(epoch_num, {'acc': class_acc, 'loss': loss_class_cls})
                #                TC.on_epoch_end(epoch_num, log)
                break

        except Exception as e:
            with open(C.output_folder + "error_log.txt", "a") as log:
                log.write(str(e))
                log.write("---\n")
            print('Exception:: {}'.format(e))

            raise
            #continue

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
print('Training complete, exiting.')
