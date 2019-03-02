import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw
import pandas as pd
from tqdm import tqdm

from utils.preprocess import yolo_eval
from utils.utils import letterbox_image
from Model.yolo import yolo_body, tiny_yolo_body
import os
from keras.utils import multi_gpu_model


class YOLO(object):
    def __init__(self,
                 model_weights_path='model_data_stage_1/trained_weights_final.h5',
                 anchors_path='anchors/anchors.txt',
                 classes_path='anchors/classes.txt',
                 score=0.1,
                 iou=0.0,
                 input_shape=(416, 416),
                 gpu_num=1):
        # update default values
        self.model_weights_path = model_weights_path
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.score = score
        self.iou = iou
        self.input_shape = input_shape
        self.gpu_num = gpu_num

        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_weights_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_weights_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.input_shape != (None, None):
            assert self.input_shape[0]%32 == 0, 'Multiples of 32 required'
            assert self.input_shape[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.input_shape)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        return out_boxes

    def close_session(self):
        self.sess.close()


anchors_path = 'anchors/anchors.txt'
classes_path = 'anchors/classes.txt'
test_path = '/home/qkh/hdd1/data/gang_jin/test_dataset'
weights_file_path = 'model_data_stage_1_9/trained_weights_final.h5'
input_shape = (544, 544)
score = 0.5
output_file = 'predict.csv'

def detec_dir(dir_path=test_path):
    detect_result = []
    yolo = YOLO(model_weights_path=weights_file_path,
                anchors_path=anchors_path,
                classes_path=classes_path,
                score=score,
                input_shape=input_shape)
    for root, dirs, files in tqdm(os.walk(dir_path)):
        for file_name in files:
            if file_name.endswith('.jpg'):
                try:
                    image = Image.open(os.path.join(test_path, file_name))
                except:
                    print('Open ' + file_name + ' Error!')
                else:
                    boxes = yolo.detect_image(image)
                    boxes = np.int32(boxes)
                    for box in boxes:
                        str_box = str(box[1]) + ' ' + str(box[0]) + ' ' + str(box[3]) + ' ' + str(box[2])
                        detect_result.append([file_name, str_box])

    return detect_result


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    detect_result = detec_dir(test_path)
    data = pd.DataFrame(detect_result)
    data.to_csv(output_file, header=False, index=False)
