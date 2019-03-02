import os

root_path = '/home/test/data/gang_jin'		# set your data path here
train_file = open(os.path.join(root_path, 'VOC_gangjin/ImageSets/Main/train.txt'), 'w')
test_file = open(os.path.join(root_path, 'VOC_gangjin/ImageSets/Main/test.txt'), 'w')
for _, _, train_files in os.walk(os.path.join(root_path, 'train_dataset')):
    for file in train_files:
        train_file.writelines(file.split('.')[0]+'\n')
for _, _, test_files in os.walk(os.path.join(root_path, 'test_dataset')):
    for file in test_files:
        test_file.write(file.split('.')[0]+'\n')
train_file.close()
test_file.close()
