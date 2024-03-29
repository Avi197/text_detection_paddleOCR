import itertools
import json
import glob
import os

from text_detection_ctpn.utils.bbox_utils import natural_sort


def get_img_label_pair(path):
    list_file = natural_sort(os.listdir(path))
    group_list = list(itertools.zip_longest(*[iter(list_file)] * 2))
    if group_list[-1][0].split('.')[0] != group_list[-1][1].split('.')[0]:
        print('something wrong')
        return None
    return list(itertools.zip_longest(*[iter(list_file)] * 2))


if __name__ == '__main__':
    data_folder = '/home/phamson/data/text_detection/hdmtk'
    paddle_label_path = '/home/phamson/data/text_detection/hdmtk_paddle_label.txt'

    list_file_label_pair = get_img_label_pair(data_folder)

    label_dict = {}
    with open(paddle_label_path, 'w') as file:
        for img, label in list_file_label_pair:

            img_label_dict = []

            img_path = os.path.join(data_folder, img)
            label_path = os.path.join(data_folder, label)
            with open(label_path, 'r') as label_file:
                img_label = {}
                for line in label_file.readlines():
                    value = line.split(',')
                    img_label["transcription"] = str(value[-1].strip())
                    img_label["points"] = [[value[0], value[1]], [value[2], value[3]],
                                           [value[4], value[5]], [value[6], value[7]]]
                    img_label_dict.append(img_label)
            file.write(f'{img_path}\t{img_label_dict}\n')