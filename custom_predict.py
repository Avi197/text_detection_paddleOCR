import cv2
import time

import numpy as np

from text_detection_paddle.text_detector_paddle import TextDetectorPaddle
from text_detection_paddle.utils import utility

# python3 tools/infer/text_detector_paddle.py --det_algorithm="DB"
# --det_model_dir="./output/det_db_aug/best_accuracy"
# --image_dir='/opt/data/13_pdfsam_2021-12-14-14-09-08-01-1.jpg'
# --use_gpu=True


if __name__ == '__main__':
    args = utility.parse_args()
    args.det_algorithm = "DB"
    args.det_model_dir = os.path.join(model_dir, 'paddle')
    args.use_gpu = False
 
    image_file_list = get_image_file_list(args.image_dir)

    paddle_text_detector = TextDetectorPaddle(args)

    img = cv2.imread(image_file_list[0])

    paddle_text_detector.predict_ratio(self.np_img)
    
