import os
import numpy as np
from PIL import Image
from siamese import Siamese
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import random
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 定义阈值
THRESHOLD = 0.5

# 加载 Siamese 模型
model = Siamese()

def load_image_pairs(data_dir, max_pairs=10000):
    """
    加载数据集中所有的图像对和标签，限制生成的图像对数量。

    Parameters:
        data_dir (str): 数据集的路径。
        max_pairs (int): 最大生成的图像对数量。

    Returns:
        list: 图像对的列表。
        list: 每对图像的标签列表（1 表示同类，0 表示不同类）。
    """
    image_pairs = []
    labels = []

    folders = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]
    print(f"Found {len(folders)} folders in the dataset.")
    
    for folder in folders:
        print(f"Processing folder: {folder}")
        images = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('png', 'jpg', 'jpeg'))]
        print(f"  Found {len(images)} images in folder {folder}.")
        # 生成同类的图像对
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                image_pairs.append((images[i], images[j]))
                labels.append(1)

    # 生成跨文件夹的不同类图像对
    for folder in folders:
        images = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('png', 'jpg', 'jpeg'))]
        other_folders = [f for f in folders if f != folder]
        for other_folder in other_folders:
            other_images = [os.path.join(other_folder, img) for img in os.listdir(other_folder) if img.endswith(('png', 'jpg', 'jpeg'))]
            for img1 in images:
                for img2 in other_images:
                    image_pairs.append((img1, img2))
                    labels.append(0)

        # 随机打乱并限制正负样本数量相等
    combined = list(zip(image_pairs, labels))
    positive_samples = [pair for pair in combined if pair[1] == 1]
    negative_samples = [pair for pair in combined if pair[1] == 0]
    
    # 确保正负样本数量相等
    min_samples = 1000
    positive_samples = random.sample(positive_samples, min_samples)
    negative_samples = random.sample(negative_samples, min_samples)
    
    # 合并并打乱
    combined = positive_samples + negative_samples
    random.shuffle(combined)
    image_pairs, labels = zip(*combined)


    print(f"Generated {len(image_pairs)} image pairs (limited to {max_pairs}).")
    return image_pairs, labels

def evaluate_model(image_pairs, labels):
    """
    评估模型性能。

    Parameters:
        image_pairs (list): 图像对的列表。
        labels (list): 每对图像的标签列表（1 表示同类，0 表示不同类）。

    Returns:
        dict: 包含准确率、召回率、精确率和 F1-Score 的字典。
        list: 预测结果列表。
    """
    predictions = []

    for index, (img1_path, img2_path) in enumerate(image_pairs):
        try:
            if index % 500 == 0:
                print(f"Processing pair {index + 1}/{len(image_pairs)}: {img1_path}, {img2_path}")
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            prob = model.detect_image(img1, img2).item()
            predictions.append(1 if prob >= THRESHOLD else 0)
        except Exception as e:
            print(f"Error processing images {img1_path} and {img2_path}: {e}")
            continue

    print("Calculating metrics...")
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    precision = precision_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1_score': f1
    }, predictions

if __name__ == "__main__":
    data_dir = "" # 选择自己的测试集文件夹
    max_pairs = 2000
    print("Loading image pairs and labels...")
    image_pairs, labels = load_image_pairs(data_dir, max_pairs=max_pairs)

    print("Evaluating model...")
    metrics, predictions = evaluate_model(image_pairs, labels)

    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    # 计算并打印混淆矩阵
    conf_matrix = confusion_matrix(labels, predictions)
    print("\nConfusion Matrix:")
    print(conf_matrix)
