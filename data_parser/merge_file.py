import glob
import shutil

with open('/opt/github/text_detection_paddleOCR/label.txt', 'wb') as outf:
    for file in glob.glob('/opt/github/text_detection_paddleOCR/training_data/*.txt'):
        with open(file, 'rb') as fd:
            shutil.copyfileobj(fd, outf)
            outf.write(b'\n')

