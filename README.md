# Text detection using paddleOCR

##### PaddleOCR installation

clone PaddleOCR https://github.com/PaddlePaddle/PaddleOCR

Install paddle lib on conda
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html

```
conda install paddlepaddle-gpu==2.2.2 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge
```

##### Image label format

preprocess folder contain script that convert ICDAR and CTPN format to paddleOCR format
<br>
ICDAR format

```
"polygon_points,lang,text"
78,55,419,55,419,109,78,109,Latin,###
111,283,1521,283,1521,323,111,323,Latin,###
```

text-detection-CTPN format

```
"(xmin, ymin, xmax, ymax)"
28,20,31,40
32,20,47,40
48,20,63,40
64,20,79,40
80,20,95,40
```

PaddleOCR format

```
" Image file name             Image annotation information encoded by json.dumps"
img_file_name.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]

if you plan on training text detection only, change transcription to random text like AAAA
```

Training data in yml

```    
data_dir: path_to_folder_img
label_file_list: path_to_label_file.txt
```

final image path wil be joined with data_dir variable in yml file

```
data_dir + img_file_name.jpg
```

### Training

##### Detail for customizing PaddleOCR

config [PaddleOCR config](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_en/config_en.md)
<br>
configs folder contain 1 sample config file

##### Training scripts with config

```
modify the yml file with training data to quickly train the model
```

```
python3 tools/train.py -c configs/det/det_mv3_db_dk.yml

-o to modify the yml variable without edit it
```

It is recommended to run the defaults model on new data and modify those data, then train with the modified data
500 data samples, lr=0.001, epoch=500 is recommended for quick training with acceptable result

##### Convert trained model to inference model

```
python3 tools/export_model.py -c configs/det/det_mv3_db_dk.yml -o Global.pretrained_model="./output/det_db/best_accuracy"

change Global.pretrained_model variable to the just trained models path
```

### Prediction

Use the inference model to get prediction result

command line

```
python3 tools/infer/predict_det.py --det_algorithm="DB" --det_model_dir="./output/det_db_inference/" --image_dir="./doc/imgs/" --use_gpu=True
```

code

```
custom_predict.py
```

text_detection Class

```
CustomPaddleOCR.py inherit from main PaddleClass
text_detector_paddle.py add custom pre/post_process text detection
```

### Note

If encoutner this error
```
RuntimeError: (PreconditionNotMet) Cannot load cudnn shared library. Cannot invoke method cudnnGetVersion.
```

run this
```
export LD_LIBRARY_PATH={conda_env/lib}:$LD_LIBRARY_PATH
```



