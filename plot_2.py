# generated from GPT-4
import matplotlib.pyplot as plt
import sys
import os



def read_loss(file_path):
    """读取文件中的Loss值"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            losses = [float(line.strip()) for line in lines if line.strip()]
        return losses
    except Exception as e:
        print(f"读取文件 {file_path} 时发生错误: {e}")
        return []

def plot_losses(file_paths):
    """绘制多个Loss曲线"""
    plt.figure()
    labels = {file_paths[0]: "L1", file_paths[1]:"L2"}
    for file_path in file_paths:
        losses = read_loss(file_path)
        epochs = list(range(1, len(losses) + 1))
        plt.plot(epochs, losses, label=labels[file_path])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title("layer: {784,256,10}, lr=1e-3, epoch=2000, lambda=0.1")

    # 保存图片到第一个文件的文件夹下
    save_path = os.path.join(os.path.dirname(file_paths[0]), 'loss_comparison.png')
    plt.savefig(save_path)
    print(f"图像已保存至 {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("请提供两个文本文件路径作为参数。")
    else:
        file_paths = sys.argv[1:3]
        plot_losses(file_paths)
