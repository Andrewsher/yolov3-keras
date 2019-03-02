'''
pre-train on the origin dataset
'''
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam
import os
import h5py

from Model.yolo import create_model, create_tiny_model
from Model.loss import yolo_loss
from anchors.get_anchors import get_anchors, get_classes
# from utils.pascal_voc_parser import get_pascal_detection_data
from data_generator import data_generator_wrapper

anchors_path = 'anchors/anchors.txt'
classes_path = 'anchors/classes.txt'
# dataset_path = '/home/qkh/hdd1/data/gang_jin/VOC_gangjin/'   # set your data path here
data_file_path = '/home/qkh/hdd1/data/gang_jin/data_544.h5'      # set your h5 file path here
input_shape = (544, 544)
log_dir = 'model_data_stage_1/'
batch_size = 8
pretrained_weights = 'pretrained_weights/yolo.h5'


def train():
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    # split for train/val
    val_split = 0.1
    h5_file = h5py.File(data_file_path, 'r')
    data_indexes = [i for i in range(h5_file['image_data'].shape[0])]
    print('{} images gotten'.format(len(data_indexes)))
    np.random.seed(2019)
    np.random.shuffle(data_indexes)
    num_val = int(len(data_indexes) * val_split)
    num_train = len(data_indexes) - num_val

    # lines, _, _ = get_pascal_detection_data(input_path=dataset_path)
    # np.random.seed(2019)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines) * val_split)
    # num_train = len(lines) - num_val

    # create model, load the pre-trained weights
    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape=input_shape, yolo_loss=yolo_loss, anchors=anchors, freeze_body=2,
                                  num_classes=num_classes, load_pretrained=True)
    else:
        model = create_model(input_shape=input_shape, yolo_loss=yolo_loss, anchors=anchors, freeze_body=2,
                             num_classes=num_classes, load_pretrained=True)  # make sure you know what you freeze
    # model = multi_gpu_model(model, gpus=2)
    model.summary()
    # set callback functions
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', verbose=1,
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    csv_logger = CSVLogger(log_dir + 'record.csv')
    tensorboard = TensorBoard(log_dir=log_dir)

    # train the last layers
    # # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    model.compile(optimizer=Adam(lr=1e-3), loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size*4))
    model.fit_generator(data_generator_wrapper(h5_file, batch_size*4, data_indexes[:num_train], anchors, num_classes, input_shape),
                        steps_per_epoch=max(1, num_train // (batch_size*4)),
                        validation_data=data_generator_wrapper(h5_file, batch_size, data_indexes[num_train:], anchors, num_classes, input_shape),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=20,
                        initial_epoch=0,
                        callbacks=[checkpoint, reduce_lr, early_stopping, tensorboard])
    # model.save_weights(log_dir + 'trained_weights_final.h5')


    # train all the layers
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=1e-3),
                  loss={'yolo_loss': lambda y_true, y_pred: y_pred})  # recompile to apply the change
    print('Unfreeze all of the layers.')
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(data_generator_wrapper(h5_file, batch_size, data_indexes[:num_train], anchors, num_classes, input_shape),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator_wrapper(h5_file, batch_size, data_indexes[num_train:], anchors, num_classes, input_shape),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=1000,
                        initial_epoch=20,
                        callbacks=[checkpoint, reduce_lr, early_stopping, csv_logger, tensorboard])
    model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    train()
    # import cv2
    # lines, _, _ = get_pascal_detection_data(input_path=dataset_path)
    # for line in lines:
    #     if line['filepath'].endswith('D8146090.jpg'):
    #         img = cv2.imread(line['filepath'])
    #         for bbox in line['bboxes']:
    #             cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 0, 255))
    #             cv2.imwrite('img.jpg', img)
    # cv2.imwrite('img.jpg', img)
    # img = cv2.imread(lines[0]['filepath'])
    # img = cv2.resize(img, (640, 640))
    # anchors = [[20,20], [22,22], [25,25], [33,33],[35,35],[38,38],[45,45],[50,50],[55,55]]
    # print(anchors)
    # for bbox in lines[0]['bboxes']:
    #     cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
    #                                                       'x2'], bbox['y2']), (0, 0, 255))
    #     cv2.imwrite('img.jpg', img)
    # bbox = lines[0]['bboxes'][0]
    # x1, y1 = 300, 300
    #
    # for anchor in anchors:
    #     cv2.rectangle(img, (x1, y1), (x1+int(anchor[0]), y1+int(anchor[1])), (0, 0, 255), 1)
    #
    # cv2.imwrite('img_current_anchors.jpg', img)

