## PPG-Based-Identity-Authentication
---
## 项目简介
本项目实现了一种基于光体积描记（PPG）信号的孪生神经网络身份认证系统。
通过可穿戴设备采集到的 PPG 信号，模型能够区分不同用户，实现非接触式、生理特征级的身份验证。

项目主要功能包括：

1. 使用 **孪生神经网络（Siamese Network）** 进行特征提取与相似度度量。
2. 支持 **新用户认证**，无需重新训练模型即可进行识别。
3. 提供在多个数据集（BIDMC、CapnoBase）的数据处理代码。

## 推荐环境
项目运行操作系统为linux，torch == 2.5.1+cu124
推荐根据实际情况安装推荐包，避免版本冲突，environment.yml为作者的环境配置

## 文件下载
训练所需的预训练参数文件vgg16-397923af.pth可在百度网盘中下载。
链接: https://pan.baidu.com/s/15eqpKG6p3cpBu5tzfibq2A 提取码: qp4c
将文件存放至"model_data"目录

BIDMC数据集下载地址：https://physionet.org/content/bidmc/1.0.0/

CapnoBase数据集下载地址：https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/NLB8IT

基于BIDMC数据集训练的模型参数文件：BIDMC_best_epoch_weights.pth
下载地址: https://pan.baidu.com/s/1cHXjUMQ0SZ9JrHHj5vIQOQ 提取码: j4w8

基于CapnoBase数据集训练的模型参数文件：Cap_best_epoch_weights.pth
链接: https://pan.baidu.com/s/1Tu3ixFNNxcM7aeL4r2rWXg 提取码: 48u6

## 数据处理
对于每个数据集的训练和预测需要分开进行，因为数据处理把图片保存在相同文件夹中

### BIDMC
将下载的BIDMC数据集保存在"datasets"目录下，运行bidmcdatapro.py，将PPG数据转化成图片，保存在"datasets/images_background/"文件夹下，注意根据文件位置修改folder_path参数值
### CapnoBase
将下载的CapnoBase数据集保存在"datasets"目录下，运行capdatapro.py，将PPG数据转化成图片，保存在"datasets/images_background/"文件夹下，注意根据文件位置修改folder_path参数值

## 训练模型
"datasets/images_background/"文件夹的格式如下：
```python
- image_background
	- character01
		- 01.png
		- 02.png
		- ……
	- character02
	- character03
	- ……
```
训练步骤为：  
1. 数据处理部分会自动按上述格式放置数据集，放在根目录下的dataset文件夹下。     
2. 之后将train.py当中的train_own_data设置成True。  
3. 运行train.py开始训练。 

## 模型预测
首先预测前需要修改siamese.py中的"model_path"参数，修改为自己的模型参数位置。此处修改不影响训练。
提供两个数据集训练好的模型参数，详见"文件下载"部分
运行test.py，模型会自动预测，输出混淆矩阵、准确率、精确率、召回率和F1分数。
data_dir参数需要修改为自己的预测文件夹，图片存放格式和"datasets/images_background/"相同。


## Reference
本项目中的孪生神经网络部分基于 Bubbliiiing 的 Siamese-pytorch 实现（https://github.com/bubbliiiing/Siamese-pytorch.git）