import os
import random
from torchvision import transforms
from my_dataset import MyDataSet

def read_split_data(data_path):
    """
    读取数据集中的图像路径和标签，并划分训练集和验证集
    :param data_path: 数据集根目录
    :return: 训练集和验证集的图像路径与标签
    """
    random.seed(0)  # 保证随机结果可复现
    
    # 遍历数据集根目录下的所有文件夹，每个文件夹对应一个类别
    class_names = [cls for cls in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cls))]
    class_names.sort()  # 排序，保证类别顺序固定
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    train_images_path = []  # 存储训练集图像路径
    train_images_label = []  # 存储训练集图像标签
    val_images_path = []    # 存储验证集图像路径
    val_images_label = []   # 存储验证集图像标签
    every_class_num = []    # 存储每个类别的样本数量
    
    # 遍历每个类别
    for cls_name in class_names:
        cls_path = os.path.join(data_path, cls_name)
        # 获取该类别下所有图像
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) 
                  if img.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # 记录该类别的样本数量
        images_num = len(images)
        every_class_num.append(images_num)
        
        # 按8:2的比例划分训练集和验证集
        val_split = int(images_num * 0.2)
        random.shuffle(images)  # 打乱图像顺序
        
        # 分配训练集和验证集
        train_images = images[val_split:]
        val_images = images[:val_split]
        
        # 存储训练集路径和标签
        train_images_path.extend(train_images)
        train_images_label.extend([class_to_idx[cls_name]] * len(train_images))
        
        # 存储验证集路径和标签
        val_images_path.extend(val_images)
        val_images_label.extend([class_to_idx[cls_name]] * len(val_images))
    
    print(f"训练集样本数: {len(train_images_path)}")
    print(f"验证集样本数: {len(val_images_path)}")
    print(f"类别数: {len(class_names)}, 类别名称: {class_names}")
    
    return train_images_path, train_images_label, val_images_path, val_images_label

def get_data_transforms():
    """
    获取训练集和验证集的数据增强变换
    :return: 训练集和验证集的变换字典
    """
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

def create_dataloaders(args):
    """
    创建训练集和验证集的DataLoader
    :param args: 命令行参数
    :return: 训练集DataLoader和验证集DataLoader
    """
    # 读取并划分数据
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)
    
    # 获取数据变换
    data_transform = get_data_transforms()
    
    # 实例化训练数据集
    train_dataset = MyDataSet(
        images_path=train_images_path,
        images_class=train_images_label,
        transform=data_transform["train"]
    )
    
    # 实例化验证数据集
    val_dataset = MyDataSet(
        images_path=val_images_path,
        images_class=val_images_label,
        transform=data_transform["val"]
    )
    
    # 计算数据加载的线程数
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    print(f'使用 {nw} 个数据加载线程')
    
    # 创建训练集DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=nw,
        collate_fn=train_dataset.collate_fn
    )
    
    # 创建验证集DataLoader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=nw,
        collate_fn=val_dataset.collate_fn
    )
    
    return train_loader, val_loader
