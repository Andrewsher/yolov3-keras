'''
pre-train on the origin dataset
'''
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam, SGD
import os

from Model.yolo import create_model, create_tiny_model
from Model.loss import yolo_loss
from Model.callbacks import SnapshotCallbackBuilder
from anchors.get_anchors import get_anchors, get_classes
from utils.pascal_voc_parser import get_pascal_detection_data
from data_generator import data_generator_wrapper


anchors_path = 'anchors/anchors.txt'
classes_path = 'anchors/classes.txt'
dataset_path = '/home/qkh/data/gang_jin/VOC_gangjin/'  # set your data path here
input_shape = (640, 640)
log_dir = 'model_data_stage_2/'
batch_size = 4
pretrained_weights = 'model_data_stage_1/trained_weights_final.h5'


def train():
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    # split for train/val
    val_split = 0.2
    lines, _, _ = get_pascal_detection_data(input_path=dataset_path)
    np.random.seed(2019)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    # create model, load the pre-trained weights
    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape=input_shape, yolo_loss=yolo_loss, anchors=anchors, freeze_body=0,
                                  num_classes=num_classes, load_pretrained=True, weights_path=pretrained_weights)
    else:
        model = create_model(input_shape=input_shape, yolo_loss=yolo_loss, anchors=anchors, freeze_body=0,
                             num_classes=num_classes, load_pretrained=True, weights_path=pretrained_weights)  # make sure you know what you freeze
    model.summary()
    # set callback functions
    callback_builder = SnapshotCallbackBuilder(nb_epochs=1000, nb_snapshots=20, init_lr=1e-4)
    callbacks = callback_builder.get_callbacks(log_dir=log_dir)
    callbacks.append(CSVLogger(log_dir + 'record.csv'))
    callbacks.append(TensorBoard(log_dir=log_dir))

    # train
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    model.compile(optimizer=SGD(lr=1e-4), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // 1),
                        validation_data=data_generator_wrapper(lines[num_train:], 1, input_shape, anchors,
                                                               num_classes),
                        validation_steps=max(1, num_val // 1),
                        epochs=1000,
                        initial_epoch=0,
                        callbacks=callbacks)
    model.save_weights(log_dir + 'trained_weights_final.h5')


    # Further training if needed.

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train()
