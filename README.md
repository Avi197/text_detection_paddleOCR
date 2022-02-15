text detection using paddleOCR

#### PaddleOCR installation
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/conda/linux-conda.html
#### Image label format

preprocess folder contain script that convert ICDAR and CTPN format to paddleOCR format
<br>
ICDAR format
```
78,55,419,55,419,109,78,109,Latin,###
111,283,1521,283,1521,323,111,323,Latin,###
```
text-detection-CTPN format
```
(xmin, ymin, xmax, ymax)
28,20,31,40
32,20,47,40
48,20,63,40
64,20,79,40
80,20,95,40
```
PaddleOCR format

```
" Image file name             Image annotation information encoded by json.dumps"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```

#### Training

Detail for customizing PaddleOCR config [PaddleOCR config](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/doc/doc_en/config_en.md)
<br>
configs folder contain 1 sample config file


Training scripts with config
```
python3 tools/train.py -c configs/det/det_mv3_custom.yml
```
