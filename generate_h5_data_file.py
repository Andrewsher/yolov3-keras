import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam
import os
from skimage import exposure
import h5py

from Model.yolo import create_model, create_tiny_model
from Model.loss import yolo_loss
from anchors.get_anchors import get_anchors, get_classes
from utils.pascal_voc_parser import get_pascal_detection_data
from data_generator import data_generator_wrapper
from utils.utils import get_random_data, preprocess_true_boxes
from utils.DataAugmentForObejctDetection import DataAugmentForObjectDetection
from tqdm import tqdm


anchors_path = 'anchors/anchors.txt'
classes_path = 'anchors/classes.txt'
dataset_path = '/home/qkh/hdd1/data/gang_jin/VOC_gangjin/'   # set your data path here
generated_path = '/home/qkh/hdd1/data/gang_jin/'             # set the path for saving generated h5 file
input_shape = (544, 544)
aug_scale = 64

class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)
all_imgs, _, _ = get_pascal_detection_data(input_path=dataset_path)

dataAug = DataAugmentForObjectDetection()
image_data = []
box_data = []
for i in tqdm(range(len(all_imgs))):
    # 获取原始图像
    origin_img, origin_box = get_random_data(all_imgs[i], input_shape, random=False)
    origin_img = exposure.rescale_intensity(origin_img, out_range=(0, 255))
    origin_img = np.uint8(origin_img)
    image_data.append(origin_img)
    box_data.append(origin_box)

    # 获取增强数据
    # data augmentation
    for b in range(aug_scale - 1):
        auged_img, auged_box = dataAug.dataAugment(origin_img, origin_box[:, :4])
        auged_img = np.uint8(auged_img)
        tmp_box = np.zeros(origin_box.shape)
        tmp_box[:, :4] = auged_box[:, :4]
        tmp_box[:, 4] = origin_box[:, 4]
        image_data.append(auged_img)
        box_data.append(tmp_box)

image_data = np.array(image_data)
box_data = np.array(box_data)

file = h5py.File(os.path.join(generated_path, 'data_544.h5'), 'w')
file.create_dataset('image_data', data=image_data)
file.create_dataset('box_data', data=box_data)
file.close()
