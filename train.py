import csv
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 176)

#D:/python code/deep_learn/data/classify-leaves

labels_dataframe = pd.read_csv('D:/python code/deep_learn/data/classify-leaves/train.csv')
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
class_to_num = dict(zip(leaves_labels, range(n_classes)))
num_to_class = {v: k for k, v in class_to_num.items()}


# 继承pytorch的dataset，创建自己的
class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, k=0, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.k = k
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * valid_ratio) - 1

        if mode == 'train':
            # 第一列包含图像文件的名称,self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            self.train_image = np.asarray(pd.concat([self.data_info.iloc[1:self.k * self.train_len, 0],
                                                     self.data_info.iloc[1 + (self.k + 1) * self.train_len:, 0]]))
            # 第二列是图像的 label
            self.train_label = np.asarray(pd.concat([self.data_info.iloc[1:self.k * self.train_len, 1],
                                                     self.data_info.iloc[1 + (self.k + 1) * self.train_len:, 1]]))
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(
                self.data_info.iloc[1 + self.k * self.train_len:1 + (self.k + 1) * self.train_len, 0])
            self.valid_label = np.asarray(
                self.data_info.iloc[1 + self.k * self.train_len:1 + (self.k + 1) * self.train_len, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        img_as_img = Image.open(self.file_path + single_image_name)

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        #         if img_as_img.mode != 'L':
        #             img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


num_epochs = 2
batch_size = 16
lr = 0.05

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if torch.cuda.is_available():
    torch.cuda.init()
    print('GPU available')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # 将模型移动到设备上（如GPU）

train_path = 'D:/python code/deep_learn/data/classify-leaves/train.csv'
test_path = 'D:/python code/deep_learn/data/classify-leaves/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = 'D:/python code/deep_learn/data/classify-leaves/'


def train_ch_def(net, num_epochs, lr, device):
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net = net.to(device)

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []


    for i in range(5):
        train_dataset = LeavesData(train_path, img_path, k=i, mode='train')
        val_dataset = LeavesData(train_path, img_path, k=i, mode='valid')

        train_iter = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # 表示随机洗牌读数
        val_iter = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            net.train()
            train_loss, train_acc, n = 0.0, 0.0, 0

            train_bar = tqdm(train_iter,file=sys.stdout)
            for X, y in train_bar:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                train_loss += l.item() * y.shape[0]
                train_acc += (y_hat.argmax(dim=1) == y).sum().item()
                n += y.shape[0]
                train_bar.desc = 'train epoch[{}/{}] loss:{:.3f}' \
                    .format(epoch + 1, num_epochs, l)


            net.eval()
            val_loss, val_acc, m = 0.0, 0.0, 0
            val_bar = tqdm(val_iter, file=sys.stdout)
            for X, y in val_bar:
                X = X.to(device)
                y = y.to(device)
                with torch.no_grad():
                    y_hat = net(X)
                    l = loss(y_hat, y)
                    val_loss += l.item() * y.shape[0]
                    val_acc += (y_hat.argmax(dim=1) == y).sum().item()
                    m += y.shape[0]
                    val_bar.desc = 'valid epoch[{}/{}]'.format(epoch + 1, num_epochs)

            train_loss /= n
            train_acc /= n
            val_loss /= m
            val_acc /= m
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            valid_accuracies.append(val_acc)
            valid_losses.append(val_loss)


            print(f"Epoch {i + 1}.{epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f},"
                  f"Val Loss = {val_loss: .4f}, Val Acc = {val_acc: .4f}")

            with open('output.txt', 'w') as f:
                print(train_losses, file=f)
                print(valid_losses, file=f)
                print(train_accuracies, file=f)
                print(valid_accuracies, file=f)


train_ch_def(model, num_epochs, lr, device)
torch.save(model.state_dict(), './working/resnet18.pth')



# 定义源文件路径和目标路径
source_file = 'D:/python code/deep_learn/data/classify-leaves/sample_submission.csv'
destination_folder = './working/sample_submission.csv'

# 使用shutil.copy()函数复制文件
shutil.copy(source_file, destination_folder)

# 加载模型进行测试，并且把文件输出到csv提交文件里

model.load_state_dict(torch.load('./working/resnet18.pth'))
model.eval()  # 设置模型为评估模式

test_path = 'D:/python code/deep_learn/data/classify-leaves/test.csv'
# csv文件中已经images的路径了，因此这里只到上一级目录
img_path = 'D:/python code/deep_learn/data/classify-leaves/'

test_dataset = LeavesData(test_path, img_path, k=0, mode='test')
test_iter = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 将模型移动到设备
model.to(device)

csv_file = './working/sample_submission.csv'
num = 0

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image', 'label'])  # 写入表头

for images in test_iter:
    # 对图像进行预处理
    images = images.to(device)
    outputs = model(images)
    # 获取预测结果
    _, predicted = torch.max(outputs, 1)

    # 打印预测结果
    # print(num_to_class[predicted])
    class_predictions = [num_to_class[i.item()] for i in predicted]

    # 将列表元素按行写入CSV文件
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # 检查文件是否为空
            writer.writerow(['image', 'label'])  # 写入表头
        for index, element in enumerate(class_predictions):
            writer.writerow([f'images/{18353 + index + num}.jpg', element])
    num += batch_size

print("列表元素已写入CSV文件：", csv_file)

# 查看预测结果
predict=pd.read_csv('./working/sample_submission.csv')

print(predict)