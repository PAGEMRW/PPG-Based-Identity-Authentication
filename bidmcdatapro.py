import wfdb
import numpy as np
import os
import vitaldb
from PyEMD import EMD, EEMD
import matplotlib
from pathlib import Path
matplotlib.use('Agg')  # 使用无 GUI 的后端
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
def getdata(datapath):
    record = wfdb.rdrecord(datapath)
    signal = record.p_signal[:, 1]
    return signal

# 定义保存顶视图函数
def save_top_view(imfs, filename):
    """
    将 IMF 的矩阵保存为顶视图图像文件
    :param imfs: IMF 分量矩阵 (2D NumPy array)
    :param filename: 保存图像的文件名
    """
    imf_matrix = np.array(imfs)[:6]  # 提取前 6 个 IMF
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(imf_matrix, aspect='auto', cmap='viridis', extent=[0, imf_matrix.shape[1], 1, 6])
    plt.axis('off')  # 去掉坐标轴和边框
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # 关闭图形以释放内存
    print(f"顶部视图保存为 '{filename}'")

# 超时包装函数
def calculate_imfs(method, signal, queue):
    """
    计算 IMF 的函数，结果存入队列。
    :param method: 分解方法 (EMD/EEMD)
    :param signal: 输入信号
    :param queue: 用于传递结果的队列
    """
    try:
        imfs = method(signal)
        queue.put(imfs)
    except Exception as e:
        queue.put(None)
        print(f"分解失败：{e}")

# 带超时机制的 IMF 计算函数
def get_imfs_with_timeout_immediate(method, signal, timeout=120):
    """
    带即时结果返回的 IMF 计算
    :param method: 分解方法 (EMD/EEMD)
    :param signal: 输入信号
    :param timeout: 超时时间（秒）
    :return: 计算得到的 IMFs 或 None（超时）
    """
    from multiprocessing import Process, Queue
    import time

    def calculate_imfs(method, signal, queue):
        """子进程中计算 IMFs，并将结果放入队列"""
        try:
            imfs = method(signal)
            queue.put(imfs)
        except Exception as e:
            queue.put(None)
            print(f"计算失败：{e}")

    queue = Queue()
    process = Process(target=calculate_imfs, args=(method, signal, queue))
    process.start()

    start_time = time.time()
    while time.time() - start_time < timeout:
        if not queue.empty():
            # 如果队列中有结果，立即返回
            result = queue.get()
            process.terminate()  # 确保子进程资源释放
            process.join()
            return result
        time.sleep(0.1)  # 避免频繁占用 CPU 资源

    # 超时处理
    if process.is_alive():
        process.terminate()
        process.join()
        print("IMF 计算超时，跳过此片段")
    return None

folder_path = Path("datasets/bidmc-ppg-and-respiration-dataset-1.0.0")

start_index = 1
end_index = 53
# 遍历从 bidmc01 到 bidmc53 的文件
for i in range(start_index, end_index + 1):
    # 构造文件名（两位编号，前缀为0）
    file_name = f"bidmc{i:02d}"  # :02d 确保编号为两位
    file_path = f"{folder_path}/{file_name}"
    signal = getdata(file_path)
    lsseg = [signal[i:i + 1250] for i in range(0, len(signal), 1250)]
    i = 0
    output_path = "datasets/images_background/"+file_name+"/"
    os.makedirs(output_path, exist_ok=True)
    for seg in lsseg:
        i+=1
        # emd_method = EMD()
        # imfs = get_imfs_with_timeout_immediate(emd_method, seg, timeout=120)
        # if imfs is not None:
        #     image_name = f"{file_name}_EMD_{i:04d}.png"
        #     image_path = os.path.join(output_path, image_name)
        #     save_top_view(imfs, image_path)

        # 使用 EEMD 方法生成顶视图
        eemd_method = EEMD()
        imfs = get_imfs_with_timeout_immediate(eemd_method, seg, timeout=120)
        if imfs is not None:
            image_name = f"{file_name}_EEMD_{i:04d}.png"
            image_path = os.path.join(output_path, image_name)
            save_top_view(imfs, image_path)
