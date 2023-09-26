import numpy as np
import json
import glob

# label_path = '/opt/data/pti_ocr/dk_train_9_5'
# for file in glob.glob(label_path):


# anno_file = '/opt/data/pti_ocr/dk_train_9_5/text_detection_crop_3/result.json'
# paddle_text = '/opt/github/text_detection_paddleOCR/training_data/crop_3.txt'

# label studio image path
# label_path = '/label-studio/data/license_plate_train/lp_detection'
# anno_img_path = ''

result_list = []


def coco_parser(anno_file, paddle_train_text, paddle_val_text, label_path, val_percentage=0.2):
    anno_img_path = ''
    with open(anno_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']

    with open(paddle_train_text, 'w', encoding='utf-8') as f_train, \
            open(paddle_val_text, 'w', encoding='utf-8') as f_val:
        for idx, img in enumerate(images):
            paddle_anno = []
            for anno in annotations:
                if img['id'] == anno['image_id']:
                    round_anno = [int(np.rint(i)) for i in anno['segmentation'][0]]
                    round_anno = [[round_anno[0], round_anno[1]], [round_anno[2], round_anno[3]],
                                  [round_anno[4], round_anno[5]], [round_anno[6], round_anno[7]]]
                    tmp = {
                        "transcription": "AAAA",
                        "points": round_anno,
                    }
                    paddle_anno.append(tmp)
            file_name = img['file_name'].replace(label_path, anno_img_path)
            if idx < len(images) * val_percentage:
                f_val.write(f'{file_name}\t{json.dumps(paddle_anno)}\n')
            else:
                f_train.write(f'{file_name}\t{json.dumps(paddle_anno)}\n')


if __name__ == '__main__':
    anno_file = '/opt/github/text_detection_paddleOCR/training_data/lp_crop.json'
    paddle_train = '/opt/github/text_detection_paddleOCR/training_data/crop_lp_ver_2.txt'
    paddle_val = '/opt/github/text_detection_paddleOCR/training_data/crop_lp_val_ver_2.txt'

    # label studio image path
    label_path = '/data/local-files/?d=/TRAIN_1/'

    # local image path
    anno_img_path = ''

    result_list = []

    coco_parser(anno_file, paddle_train, paddle_val, label_path)
