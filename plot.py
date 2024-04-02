# Generated by GPT-4
import matplotlib.pyplot as plt
import sys
import os

def plot_loss(file_path):
    # 检查文件是否存在
    if not os.path.isfile(file_path):
        print(f"文件 {file_path} 不存在。")
        return
    
    # 读取Loss值
    with open(file_path, 'r') as f:
        lines = f.readlines()
        losses = [float(line.strip()) for line in lines if line.strip()]

    # 获取轮数
    epochs = list(range(1, len(losses) + 1))

    # 绘制Loss图
    plt.figure()
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('layer: {784,256,10}, lr=1e-1, epoch=100, BGD')
    plt.legend()

    # 保存图片
    save_path = os.path.splitext(file_path)[0] + '.png'
    plt.savefig(save_path)
    print(f"图像已保存至 {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("请提供一个文本文件路径作为参数。")
    else:
        file_path = sys.argv[1]
        plot_loss(file_path)