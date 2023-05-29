import numpy as np
import json
import glob

# label_path = '/opt/data/pti_ocr/dk_train_9_5'
# for file in glob.glob(label_path):


anno_file = '/opt/data/pti_ocr/dk_train_9_5/text_detection_crop_3/result.json'
paddle_text = '/opt/github/text_detection_paddleOCR/training_data/crop_3.txt'

label_path = '/label-studio/data/DK_train'
anno_img_path = ''
with open(anno_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

images = data['images']
annotations = data['annotations']

result_list = []

with open(paddle_text, 'w', encoding='utf-8') as t:
    for img in images:
        paddle_anno = []
        for anno in annotations:
            if img['id'] == anno['image_id']:
                round_anno = [int(np.rint(i)) for i in anno['segmentation'][0]]
                round_anno = [[round_anno[0], round_anno[1]], [round_anno[2], round_anno[3]],
                              [round_anno[4], round_anno[5]], [round_anno[6], round_anno[7]]]
                # img_result['annotations'].append(round_anno)
                tmp = {
                    "transcription": "AAAA",
                    "points": round_anno,
                }
                paddle_anno.append(tmp)
        file_name = img['file_name'].replace(label_path, anno_img_path)
        t.write(f'{file_name}\t{json.dumps(paddle_anno)}\n')
