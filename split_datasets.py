from shutil import move
import os
from random import shuffle

TARGET_PATH = '1d_barcode_extended_plain/1d_barcode_extended_plain/Detection'
PREDICTOR_PATH = '1d_barcode_extended_plain/1d_barcode_extended_plain/Original'
TRAIN_PATH = 'dataset/train'
TEST_PATH = 'dataset/test'

target_path_list_dir = os.listdir(TARGET_PATH)
TRAIN_SIZE = .7


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
    Path.create_directory(TRAIN_PATH)
    Path.create_directory(TEST_PATH)
    shuffle(target_path_list_dir)
    target_path_length = len(target_path_list_dir)
    train_length = round(target_path_length * TRAIN_SIZE)
    test_length = round(target_path_length * (1 - TRAIN_SIZE))
    paths_list = [TRAIN_PATH, TEST_PATH]
    train_index = [0, train_length]
    test_index = [train_length, train_length + test_length]
    index_list = [train_index, test_index]
    for index, datatype_path in zip(index_list, paths_list):
        for i in range(index[0], index[1]):
            filename = target_path_list_dir[i]
            target_src_path = os.path.join(TARGET_PATH, filename)
            target_dst_path = os.path.join(datatype_path, filename)
            predictor_src_path = os.path.join(PREDICTOR_PATH, filename)
            predictor_dst_path = os.path.join(datatype_path, filename)
            move(target_src_path, target_dst_path)
            move(predictor_src_path, predictor_dst_path)

