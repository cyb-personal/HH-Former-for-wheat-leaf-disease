import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据集类 - 加载小麦叶片图像
class WheatLeafDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image

# 生成器 - 将随机噪声转换为图像
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3, features_g=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),
            
            # 输出: (features_g*8) x 4 x 4
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),
            
            # 输出: (features_g*4) x 8 x 8
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),
            
            # 输出: (features_g*2) x 16 x 16
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),
            
            # 输出: features_g x 32 x 32
            nn.ConvTranspose2d(features_g, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # 输出范围[-1, 1]
            # 最终输出: 3 x 64 x 64
        )
        
    def forward(self, x):
        return self.main(x)

# 判别器 - 判断图像是真实的还是生成的
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features_d=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: 3 x 64 x 64
            nn.Conv2d(img_channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出: features_d x 32 x 32
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出: (features_d*2) x 16 x 16
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出: (features_d*4) x 8 x 8
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出: (features_d*8) x 4 x 4
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出概率值[0, 1]
        )
        
    def forward(self, x):
        return self.main(x)

# 初始化权重
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# 训练GAN
def train_gan(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    
    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化到[-1, 1]
    ])
    
    # 加载数据集
    dataset = WheatLeafDataset(args.dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # 初始化模型
    generator = Generator(args.latent_dim).to(args.device)
    discriminator = Discriminator().to(args.device)
    initialize_weights(generator)
    initialize_weights(discriminator)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
    # 固定噪声，用于观察训练过程中的生成效果变化
    fixed_noise = torch.randn(32, args.latent_dim, 1, 1).to(args.device)
    
    # 训练循环
    generator.train()
    discriminator.train()
    
    for epoch in range(args.epochs):
        loop = tqdm(dataloader, leave=True)
        for batch_idx, real in enumerate(loop):
            real = real.to(args.device)
            batch_size = real.shape[0]
            
            # 训练判别器: max log(D(x)) + log(1 - D(G(z)))
            noise = torch.randn(batch_size, args.latent_dim, 1, 1).to(args.device)
            fake = generator(noise)
            
            # 判别器对真实图像的判断
            disc_real = discriminator(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            
            # 判别器对生成图像的判断
            disc_fake = discriminator(fake.detach()).view(-1)  # detach()防止梯度流向生成器
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            
            # 总判别器损失
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            discriminator.zero_grad()
            loss_disc.backward()
            optimizer_D.step()
            
            # 训练生成器: min log(1 - D(G(z))) 等价于 max log(D(G(z)))
            output = discriminator(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            loss_gen.backward()
            optimizer_G.step()
            
            # 打印训练信息
            loop.set_postfix(
                epoch=epoch,
                disc_loss=loss_disc.item(),
                gen_loss=loss_gen.item()
            )
        
        # 每个epoch保存一次生成的图像
        if epoch % 10 == 0:
            with torch.no_grad():
                fake = generator(fixed_noise)
                save_image(fake * 0.5 + 0.5,  # 反归一化到[0, 1]
                           os.path.join(args.output_dir, "generated", f"epoch_{epoch}.png"),
                           nrow=8, normalize=True)
    
    # 保存模型
    torch.save(generator.state_dict(), os.path.join(args.output_dir, "models", "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.output_dir, "models", "discriminator.pth"))
    
    print("训练完成!")

# 生成新的图像数据集
def generate_dataset(args):
    # 创建输出目录
    os.makedirs(args.generate_dir, exist_ok=True)
    
    # 加载生成器模型
    generator = Generator(args.latent_dim).to(args.device)
    generator.load_state_dict(torch.load(os.path.join(args.output_dir, "models", "generator.pth")))
    generator.eval()  # 切换到评估模式
    
    # 生成新图像
    print(f"开始生成 {args.num_samples} 张图像...")
    with torch.no_grad():
        for i in tqdm(range(args.num_samples)):
            noise = torch.randn(1, args.latent_dim, 1, 1).to(args.device)
            fake_image = generator(noise)
            # 保存图像
            save_image(fake_image * 0.5 + 0.5,  # 反归一化
                       os.path.join(args.generate_dir, f"wheat_leaf_{i}.png"))
    
    print(f"生成完成! 图像已保存至 {args.generate_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用GAN生成小麦叶片病害图像数据集")
    
    # 数据参数
    parser.add_argument("--dataset_dir", type=str, default="D:\deep-learning-program\Transform_pre\data\WPLDD\Leaf rust",
                      help="原始小麦叶片图像数据集目录")
    parser.add_argument("--img_size", type=int, default=64, 
                      help="图像尺寸")
    parser.add_argument("--latent_dim", type=int, default=100, 
                      help="潜在空间维度")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                      help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, 
                      help="批次大小")
    parser.add_argument("--lr", type=float, default=0.0002, 
                      help="学习率")
    parser.add_argument("--beta1", type=float, default=0.5, 
                      help="Adam优化器的beta1参数")
    parser.add_argument("--num_workers", type=int, default=4, 
                      help="数据加载的线程数")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="gan_results/gan_results_r",
                      help="训练结果输出目录")
    parser.add_argument("--generate_dir", type=str, default="generated_dataset/generated_dataset_r",
                      help="生成的数据集保存目录")
    parser.add_argument("--num_samples", type=int, default=1000, 
                      help="要生成的样本数量")
    
    args = parser.parse_args()
    
    # 设置设备
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {args.device}")
    
    # 先训练GAN
    train_gan(args)
    
    # 生成新的数据集
    generate_dataset(args)
    