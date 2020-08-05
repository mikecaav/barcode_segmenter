from shutil import move
import os
from random import shuffle

MASK_PATH = '1d_barcode_extended_plain/Detection'
IMAGE_PATH = '1d_barcode_extended_plain/Original'
TRAIN_PATH = 'dataset/train'
TEST_PATH = 'dataset/test'
VALIDATION_PATH = 'dataset/validation'

TEST_SIZE = .2
VALIDATION_SIZE = .2


class Path:
    @staticmethod
    def create_directory(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def move_list_of_files_to_dir(list_of_paths, src, dst):
        for path in list_of_paths:
            src_path = f'{src}/{path}'
            dst_path = f'{dst}/{path}'
            move(src_path, dst_path)


if __name__ == '__main__':
    mask_path_list_dir = os.listdir(MASK_PATH)
    Path.create_directory(TRAIN_PATH)
    Path.create_directory(TEST_PATH)
    Path.create_directory(VALIDATION_PATH)
    shuffle(mask_path_list_dir)
    mask_path_length = len(mask_path_list_dir)
    train_length = round(mask_path_length * (1 - TEST_SIZE))
    test_length = round(mask_path_length * TEST_SIZE)
    validation_length = round(train_length * VALIDATION_SIZE)
    train_length = round(train_length * (1 - VALIDATION_SIZE))

    paths_list = [TRAIN_PATH, TEST_PATH, VALIDATION_PATH]
    test_index = [0, test_length]
    train_index = [test_length, test_length + train_length]
    validation_index = [test_length + train_length, test_length + validation_length + train_length]
    index_list = [train_index, test_index, validation_index]
    for index, datatype_path in zip(index_list, paths_list):
        for i in range(index[0], index[1]):
            filename = mask_path_list_dir[i]
            mask_src_file_path = os.path.join(MASK_PATH, filename)
            mask_dst_path = os.path.join(datatype_path, 'mask')
            Path.create_directory(mask_dst_path)
            mask_dst_path = os.path.join(mask_dst_path, filename)
            move(mask_src_file_path, mask_dst_path)

            image_src_file_path = os.path.join(IMAGE_PATH, filename)
            image_dst_path = os.path.join(datatype_path, 'image')
            Path.create_directory(image_dst_path)
            image_dst_path = os.path.join(image_dst_path, filename)
            move(image_src_file_path, image_dst_path)
