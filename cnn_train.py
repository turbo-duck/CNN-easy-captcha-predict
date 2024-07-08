import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from captcha_cnn import CaptchaCNN
from captcha_dataset import CaptchaDataset


transform = transforms.Compose([
    transforms.Resize((38, 112)),
    # 随机旋转
    transforms.RandomRotation(10),
    # 随机平移
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    # 随机颜色扰动
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])


# ========= 训练集 ==========
train_image_dir = 'data/train'
train_image_paths = os.listdir(train_image_dir)
# train_labels = [("image1.png", "1234"), ("image2.png", "5678")]
train_labels = []

for each_image in train_image_paths:
    each_label = str(each_image).split(".")[0]
    train_labels.append((each_image, each_label))

train_dataset = CaptchaDataset(train_image_dir, labels=train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =========== 测试集 ==========
test_image_dir = 'data/test'
test_image_paths = os.listdir(test_image_dir)
test_labels = []
for each_image in test_image_paths:
    each_label = str(each_image).split(".")[0]
    test_labels.append((each_image, each_label))

test_dataset = CaptchaDataset(test_image_dir, labels=test_labels, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


model = CaptchaCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# 检查 MPS 支持
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA backend")
else:
    device = torch.device("cpu")
    print("MPS and CUDA not available, using CPU")

model.to(device)

num_epochs = 4001
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # 将数据移到GPU
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = 0
        for i in range(4):
            loss += criterion(outputs[:, i, :], labels[:, i])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f"model/captcha_model_{epoch + 1}.pth")

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = 0
            for i in range(4):
                loss += criterion(outputs[:, i, :], labels[:, i])
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 2)
            total += labels.size(0) * 4  # 总字符数
            correct += (predicted == labels).sum().item()

    val_loss /= len(test_loader)
    print(f"Validation Loss: {val_loss}, Accuracy: {100 * correct / total}%")
