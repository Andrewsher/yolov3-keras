import numpy as np
from skimage import exposure
from utils.utils import get_random_data, preprocess_true_boxes
from utils.DataAugmentForObejctDetection import DataAugmentForObjectDetection


def data_generator(h5_file, batch_size, data_indexes, anchors, num_classes, input_shape):
    '''data generator for fit_generator'''
    n = len(data_indexes)
    i = 0   # 2维图片在data_indexes中的index
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(data_indexes)
            image_data.append(h5_file['image_data'][data_indexes[i]])
            box_data.append(h5_file['box_data'][data_indexes[i]])
            i = (i + 1) % n

        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data / 255., *y_true], np.zeros(batch_size)


def data_generator_wrapper(h5_file, batch_size, data_indexes, anchors, num_classes, input_shape):
    n = len(data_indexes)
    if n==0 or batch_size<=0: return None
    return data_generator(h5_file, batch_size, data_indexes, anchors, num_classes, input_shape)


if __name__ == '__main__':
    from anchors.get_anchors import get_anchors, get_classes
    from utils.pascal_voc_parser import get_pascal_detection_data

    dataset_path = '/home/qkh/data/gang_jin/VOC_gangjin/'
    anchors_path = 'anchors/anchors.txt'
    classes_path = 'anchors/classes.txt'

    lines, _, _ = get_pascal_detection_data(input_path=dataset_path)
    # new_lines = []
    # for line in lines:
    #     if line['filepath'].endswith('D8146090.jpg'):
    #         new_lines.append(line)
    batch_size = 2
    input_shape = (416, 416)
    anchors = get_anchors(anchors_path)
    num_classes = len(get_classes(classes_path))

    x = data_generator_wrapper(lines, batch_size, input_shape, anchors, num_classes)
    data, _ = x.__next__()
    data, _ = x.__next__()
    data, _ = x.__next__()
    data, _ = x.__next__()
    data, _ = x.__next__()


