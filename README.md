# CNN-easy-captcha-predict
CNN 实现训练了一个验证码的识别器 30MB大小 实现了对业务上验证码的精准识别 准确率 99.95%

# 快速开始

## 本地部署

```shell
python server.py
```

## 容器部署
```shell
docker run -d -p 15556:15556 cnn-easy-captcha
```

## 如何预测

```python
import requests
import json

headers = {}
img_url = "你的图片URL"
base_ocr_url = "http://localhost:15556/predict"

img_resp = requests.get(img_url, headers=headers, stream=True)
files = {
    "file": img_resp.content
}
# form-data 请求方式 将流传输给服务
ocr_resp = requests.post(base_ocr_url, files=files)
ocr_json = json.loads(ocr_resp.text)
print(ocr_json['predict'])

```


# 文件目录
```shell
.
├── Dockerfile
├── LICENSE
├── README.md
├── captcha_cnn.py
├── captcha_dataset.py
├── model
│   └── captcha_model_1000_ry.pth 训练了1000轮的模型
├── requirements.txt 依赖项
└── server.py 启动主程序
```
