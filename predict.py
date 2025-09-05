import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib

matplotlib.use('TkAgg')  # 修改后端以支持图像显示
import matplotlib.pyplot as plt

from newmodel import hybrid_attention_transformer as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # 加载图像
    img_path = r"D:\deep-learning-program\Transform_pre\data\val\brown spot\b01.png"
    assert os.path.exists(img_path), f"文件不存在: {img_path}"
    img = Image.open(img_path).convert('RGB')
    plt.imshow(img)  # 显示原始图像

    # 预处理图像
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)  # 增加批次维度

    # 读取类别映射
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"文件不存在: {json_path}"
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = create_model(num_classes=5, has_logits=False).to(device)

    model_weight_path = r"D:\deep-learning-program\Transform_pre\weights\best_model.pth"
    assert os.path.exists(model_weight_path), f"权重文件不存在: {model_weight_path}"

    # 加载权重并过滤不匹配的键
    state_dict = torch.load(model_weight_path, map_location=device)
    # 删除错误提示中提到的不匹配键
    if "layers.0.1.attn_mask" in state_dict:
        del state_dict["layers.0.1.attn_mask"]
    # 允许加载不严格匹配的权重
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    with torch.no_grad():
        # 预测
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 显示预测结果
    print_res = f"类别: {class_indict[str(predict_cla)]}   概率: {predict[predict_cla].numpy():.3f}"
    plt.title(print_res)

    # 打印所有类别的概率
    for i in range(len(predict)):
        print(f"类别: {class_indict[str(i)]:10}   概率: {predict[i].numpy():.3f}")

    plt.show()


if __name__ == '__main__':
    main()