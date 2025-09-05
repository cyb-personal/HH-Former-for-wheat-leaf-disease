import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import xlwt

from newmodel import hybrid_attention_transformer as create_model
from utils import train_one_epoch, evaluate
from data_loader import create_dataloaders  # 导入数据加载函数

def parse_args():
    parser = argparse.ArgumentParser(description='混合注意力Transformer训练脚本')
    parser.add_argument('--name', type=str, default='hybrid_attention_exp',
                        help='TensorBoard实验名称')
    parser.add_argument('--num_classes', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--data-path', type=str,
                        default=r"D:\deep-learning-program\Transform_pre\data")
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    return parser.parse_args()

def main(args):
    best_acc = 0

    # 确定训练设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建权重保存目录
    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    # 创建Excel工作簿和工作表，用于记录训练数据
    book = xlwt.Workbook(encoding='utf-8')
    sheet1 = book.add_sheet(u'Train_data', cell_overwrite_ok=True)
    # 添加数据列标题
    sheet1.write(0, 0, 'epoch')
    sheet1.write(0, 1, 'Train_Loss')
    sheet1.write(0, 2, 'Train_Acc')
    sheet1.write(0, 3, 'Val_Loss')
    sheet1.write(0, 4, 'Val_Acc')
    sheet1.write(0, 5, 'lr')
    sheet1.write(0, 6, 'Best val Acc')

    # 创建TensorBoard写入器
    tb_writer = SummaryWriter(comment=args.name)

    # 通过数据加载模块创建DataLoader
    train_loader, val_loader = create_dataloaders(args)

    # 创建模型
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

    # 加载预训练权重（如果提供）
    if args.weights != "":
        assert os.path.exists(args.weights), f"权重文件不存在: '{args.weights}'"
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            if k in weights_dict:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    # 冻结部分层（如果需要）
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print(f"训练层: {name}")

    # 优化器设置
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # 学习率调度器
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 训练主循环
    for epoch in range(args.epochs):
        # 记录当前epoch
        sheet1.write(epoch + 1, 0, epoch + 1)
        sheet1.write(epoch + 1, 5, str(optimizer.state_dict()['param_groups'][0]['lr']))

        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            optimizer=optimizer,
            data_loader=train_loader,
            device=device,
            epoch=epoch
        )

        # 更新学习率
        scheduler.step()

        # 记录训练结果
        sheet1.write(epoch + 1, 1, str(train_loss))
        sheet1.write(epoch + 1, 2, str(train_acc))

        # 验证
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            epoch=epoch
        )

        # 记录验证结果
        sheet1.write(epoch + 1, 3, str(val_loss))
        sheet1.write(epoch + 1, 4, str(val_acc))

        # 记录到TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")

    # 记录最佳准确率并保存Excel文件
    sheet1.write(1, 6, str(best_acc))
    book.save('./Train_data.xlsx')
    print(f"最佳验证准确率: {best_acc:.4f}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
