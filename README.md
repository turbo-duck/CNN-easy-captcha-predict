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
└── cnn_train.py 训练脚本
```

# 如何训练
## CNN网络

```python
class CaptchaCNN(nn.Module):
    def __init__(self):
        super(CaptchaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 14 * 4, 1024)
        # 添加Dropout层
        self.dropout = nn.Dropout(0.5)  
        self.fc2 = nn.Linear(1024, 4 * 10)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.conv3(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = x.view(-1, 128 * 14 * 4)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        x = self.fc2(x)
        x = x.view(-1, 4, 10)
        return x
```

conv1 卷积层1：
- 输入通道 3（RGB彩色图像是3，黑白灰度图是1）
- 输出通道 32 （产生32个特征图）
- 卷积核大小 3*3
- 步长 1
- 填充 1

conv2 卷积层2：
- 输入通道 32 （第一层输出32）
- 输出通道 64（产生64个特征图）
- 其余同上

conv2 卷积层3：
- 输入通道 64（第二层输出64）
- 输出通道 128 
- 其余同上

fc1 全连接层：
- 输入节点数：128 * 14 * 4（第三层输出128，验证码大小112*38，经过三次2x2的池化，每次池化尺寸缩小一半，最终为：14*4）
- 输出节点数：1024 超参数，输出神经元数量

dropout：
- 随机丢弃一半的神经元，防止过拟合

fc2 全连接层：
- 输入节点数：1024 （fc1输出节点数）
- 输出节点数：4 * 10 （验证码是4位数字，每一位都可能是0-9情况）


## 数据定义
```python
class CaptchaDataset(Dataset):
    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label_str = self.labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # 转换为整数列表
        label = [int(char) for char in label_str]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

```


