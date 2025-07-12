import torch
import numpy as np
import nibabel as nib
import os

from train_models.networks.generator import Generator

# 1. 配置参数
MODEL_WEIGHTS_PATH = './DeepCA/outputs_results/checkpoints/Epoch_200.tar' # 模型权重路径
INPUT_FILE_PATH = './datasets/CCTA_BP/recon_1.npy'                      # 输入文件路径 (.npy)
OUTPUT_FILE_PATH = './inference_result.nii.gz'                         # 输出文件路径 (.nii.gz)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # 运行设备

def preprocess_input(file_path):
    """加载并预处理单个输入文件。"""
    data_np = np.load(file_path)
    
    # 增加通道维度 -> (D, H, W, 1)
    data_np = data_np[:, :, :, np.newaxis]
    
    # 转换轴以匹配PyTorch的 (C, D, H, W) 格式
    data_np = np.transpose(data_np, (3, 0, 1, 2))
    
    # 转换为Tensor并增加批次维度 -> (1, C, D, H, W)
    data_tensor = torch.from_numpy(data_np).float()
    data_tensor = data_tensor.unsqueeze(0)
    
    return data_tensor.to(DEVICE)

def main():
    """主推理函数。"""
    print(f"使用设备: {DEVICE}")

    # 加载模型
    model = Generator(in_channels=1, num_filters=64, class_num=1).to(DEVICE)
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['network'])
    model.eval() # 评估模式
    print(f"模型加载成功: {MODEL_WEIGHTS_PATH}")

    if not os.path.exists(INPUT_FILE_PATH):
        print(f"错误: 输入文件未找到 {INPUT_FILE_PATH}")
        return
    input_tensor = preprocess_input(INPUT_FILE_PATH)

    # 执行推理
    print("正在执行推理...")
    with torch.no_grad():
        output_tensor = model(input_tensor)
    print("推理完成。")

    # 后处理并保存结果
    output_tensor = torch.sigmoid(output_tensor)
    output_np = output_tensor.squeeze().cpu().numpy()
    
    # 保存为Nifti格式文件 (.nii.gz)
    nifti_image = nib.Nifti1Image(output_np, affine=np.eye(4))
    nib.save(nifti_image, OUTPUT_FILE_PATH)
    
    print(f"重建结果已保存至: {OUTPUT_FILE_PATH}")

if __name__ == '__main__':
    main()