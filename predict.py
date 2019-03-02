
from Model.yolo import create_model, create_tiny_model
from Model.loss import yolo_loss
from anchors.get_anchors import get_anchors, get_classes
from Model.object_detector import YOLO

from PIL import Image
import os

anchors_path = 'anchors/anchors.txt'
classes_path = 'anchors/classes.txt'
test_path = '/home/qkh/hdd1/data/gang_jin/test_dataset'
weights_file_path = 'model_data_stage_1_9/trained_weights_final.h5'
input_shape = (544, 544)
score = 0.5

img = os.path.join(test_path, '08B8C2A8.jpg')


def detect_img(yolo):

    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
    else:
        r_image = yolo.detect_image(image)
        r_image.save('result.jpg')
    yolo.close_session()


def main():

    # predict
    detect_img(YOLO(model_weights_path=weights_file_path,
                    anchors_path=anchors_path,
                    classes_path=classes_path,
                    score=score,
                    input_shape=input_shape))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
