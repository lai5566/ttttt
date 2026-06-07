#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
类别条件生成脚本 (Class-Conditional Generation Script)

用途：使用训练好的 StyleGAN3 模型生成特定类别的医学图像

示例用法：
    # 生成 64 张 "正常" 类别的图片
    python generate_by_class.py -c 0 -n 64

    # 生成 100 张 "缺血" 类别的图片，使用 truncation
    python generate_by_class.py -c 1 -n 100 --truncation 0.7

    # 生成所有类别，每类 50 张
    python generate_by_class.py --all -n 50
"""

import os
import sys
import argparse
import glob
from os.path import join, exists, dirname, abspath

import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, join(dirname(abspath(__file__)), 'src'))

from config import Configurations
import models.model as model
import utils.ckpt as ckpt
import utils.misc as misc

# 类别名称映射
CLASS_NAMES = {
    0: 'normal',
    1: 'ischemia',
    2: 'bleeding'
}

CLASS_NAMES_ZH = {
    0: '正常',
    1: '缺血性卒中',
    2: '出血性卒中'
}


def load_config(cfg_path):
    """加载配置文件"""
    cfgs = Configurations(cfg_path)

    # 确保 RUN 配置有必需的属性（用于推理）
    if not hasattr(cfgs.RUN, 'mixed_precision'):
        cfgs.RUN.mixed_precision = False
    if not hasattr(cfgs.RUN, 'distributed_data_parallel'):
        cfgs.RUN.distributed_data_parallel = False
    if not hasattr(cfgs.RUN, 'freeze_layers'):
        cfgs.RUN.freeze_layers = -1

    return cfgs


def find_checkpoint_directory(base_dir):
    """
    自动查找实际的检查点目录

    处理两种情况：
    1. 检查点文件直接在 base_dir 中
    2. 检查点文件在 base_dir 下的时间戳子目录中

    返回包含实际检查点文件的目录路径
    """
    if not exists(base_dir):
        raise FileNotFoundError(
            f"检查点目录不存在: {base_dir}\n"
            f"请确保路径正确，或先训练模型。"
        )

    # 检查检查点文件是否直接在 base_dir 中
    if glob.glob(join(base_dir, "model=G-best-weights-step=*.pth")) or \
       glob.glob(join(base_dir, "model=G-current-weights-step=*.pth")):
        print(f"使用检查点目录: {base_dir}")
        return base_dir

    # 搜索时间戳子目录
    try:
        subdirs = [d for d in os.listdir(base_dir)
                   if os.path.isdir(join(base_dir, d))]
    except Exception as e:
        raise FileNotFoundError(
            f"无法读取目录: {base_dir}\n"
            f"错误: {e}"
        )

    # 查找包含检查点的子目录
    valid_dirs = []
    for subdir in subdirs:
        subdir_path = join(base_dir, subdir)
        # 检查是否包含生成器检查点
        if glob.glob(join(subdir_path, "model=G-best-weights-step=*.pth")) or \
           glob.glob(join(subdir_path, "model=G-current-weights-step=*.pth")):
            valid_dirs.append((subdir_path, os.path.getmtime(subdir_path)))

    if not valid_dirs:
        raise FileNotFoundError(
            f"在 {base_dir} 或其子目录中未找到检查点文件。\n"
            f"请确保您已经训练了模型。\n"
            f"期望的文件格式: model=G-best-weights-step=*.pth 或 model=G-current-weights-step=*.pth"
        )

    # 返回最近修改的目录
    valid_dirs.sort(key=lambda x: x[1], reverse=True)
    latest_dir = valid_dirs[0][0]
    print(f"自动发现检查点目录: {latest_dir}")
    return latest_dir


def load_models(cfgs, ckpt_dir, device, load_best=True):
    """
    加载训练好的生成器模型

    Args:
        cfgs: 配置对象
        ckpt_dir: 检查点目录
        device: 设备
        load_best: 是否加载最佳模型

    Returns:
        Gen, Gen_ema, Gen_mapping, Gen_synthesis
    """
    print(f"Loading models from: {ckpt_dir}")

    # 创建模型
    model_outputs = model.load_generator_discriminator(
        DATA=cfgs.DATA,
        OPTIMIZATION=cfgs.OPTIMIZATION,
        MODEL=cfgs.MODEL,
        STYLEGAN=cfgs.STYLEGAN,
        MODULES=cfgs.MODULES,
        RUN=cfgs.RUN,
        device=device,
        logger=None
    )

    # 解包返回值（可能返回 7 或 8 个值）
    if len(model_outputs) == 8:
        Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis, ema = model_outputs
    else:
        Gen, Gen_mapping, Gen_synthesis, Dis, Gen_ema, Gen_ema_mapping, Gen_ema_synthesis = model_outputs

    # 移动到设备（如果还没有在 device 上）
    # 注意：模型已经在 load_generator_discriminator 中移到设备了
    # Gen_ema 已经由 load_generator_discriminator 创建

    # 自动查找实际的检查点目录
    actual_ckpt_dir = find_checkpoint_directory(ckpt_dir)

    # 加载检查点
    when = "best" if load_best else "current"

    try:
        Gen_ckpt_path = glob.glob(join(actual_ckpt_dir, f"model=G-{when}-weights-step=*.pth"))[0]
    except IndexError:
        available_files = os.listdir(actual_ckpt_dir) if exists(actual_ckpt_dir) else []
        raise FileNotFoundError(
            f"生成器检查点未找到: {actual_ckpt_dir}\n"
            f"查找文件: model=G-{when}-weights-step=*.pth\n"
            f"可用文件: {available_files}"
        )

    print(f"Loading Generator from: {Gen_ckpt_path}")
    ckpt.load_ckpt(model=Gen, optimizer=None, ckpt_path=Gen_ckpt_path,
                   load_model=True, load_opt=False, load_misc=False)

    if cfgs.MODEL.apply_g_ema:
        try:
            Gen_ema_ckpt_path = glob.glob(join(actual_ckpt_dir, f"model=G_ema-{when}-weights-step=*.pth"))[0]
        except IndexError:
            available_files = os.listdir(actual_ckpt_dir) if exists(actual_ckpt_dir) else []
            raise FileNotFoundError(
                f"EMA 生成器检查点未找到: {actual_ckpt_dir}\n"
                f"查找文件: model=G_ema-{when}-weights-step=*.pth\n"
                f"可用文件: {available_files}"
            )
        print(f"Loading EMA Generator from: {Gen_ema_ckpt_path}")
        ckpt.load_ckpt(model=Gen_ema, optimizer=None, ckpt_path=Gen_ema_ckpt_path,
                       load_model=True, load_opt=False, load_misc=False)

    # 设置为评估模式
    Gen.eval()
    if Gen_ema is not None:
        Gen_ema.eval()

    # 对于 StyleGAN2/3，分离映射网络和合成网络
    if cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]:
        generator = Gen_ema if Gen_ema is not None else Gen
        Gen_mapping = generator.mapping
        Gen_synthesis = generator.synthesis
    else:
        Gen_mapping = None
        Gen_synthesis = None

    print("Models loaded successfully!")
    return Gen, Gen_ema, Gen_mapping, Gen_synthesis


def sample_latent(batch_size, z_dim, truncation_factor, device):
    """采样潜在向量"""
    if truncation_factor > 0 and truncation_factor < 1:
        # 使用截断正态分布
        from scipy.stats import truncnorm
        values = truncnorm.rvs(-truncation_factor, truncation_factor,
                              size=(batch_size, z_dim))
        z = torch.FloatTensor(values).to(device)
    else:
        # 标准正态分布
        z = torch.randn(batch_size, z_dim, device=device)
    return z


def generate_class_images(cfgs, Gen, Gen_ema, Gen_mapping, Gen_synthesis,
                         class_id, num_images, truncation_factor, device,
                         batch_size=16):
    """
    生成指定类别的图片

    Args:
        cfgs: 配置对象
        Gen: 生成器
        Gen_ema: EMA 生成器
        Gen_mapping: 映射网络 (StyleGAN)
        Gen_synthesis: 合成网络 (StyleGAN)
        class_id: 类别 ID (0, 1, 2)
        num_images: 生成图片数量
        truncation_factor: 截断因子 (-1 表示不使用)
        device: 设备
        batch_size: 批次大小

    Returns:
        生成的图片张量 (num_images, C, H, W)
    """
    is_stylegan = cfgs.MODEL.backbone in ["stylegan2", "stylegan3"]
    generator = Gen_ema if Gen_ema is not None else Gen

    all_images = []
    num_batches = (num_images + batch_size - 1) // batch_size

    print(f"Generating {num_images} images for class {class_id} ({CLASS_NAMES[class_id]})...")

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc=f"Class {class_id}"):
            # 当前批次大小
            current_batch_size = min(batch_size, num_images - i * batch_size)

            # 采样潜在向量
            z = sample_latent(current_batch_size, cfgs.MODEL.z_dim,
                            truncation_factor, device)

            # 创建类别标签
            y_indices = torch.full((current_batch_size,), class_id,
                                  dtype=torch.long, device=device)

            # 对于 StyleGAN2/3，需要 one-hot 编码
            if is_stylegan and cfgs.MODEL.g_cond_mtd == "cAdaIN":
                y = F.one_hot(y_indices, num_classes=cfgs.DATA.num_classes).float()
            else:
                y = y_indices

            # 生成图片
            if is_stylegan:
                # StyleGAN: z -> w -> image
                if truncation_factor > 0 and truncation_factor < 1:
                    # 使用 truncation trick
                    w = Gen_mapping(z, y)
                    # 获取平均 w (可选，这里简化处理)
                    images = Gen_synthesis(w)
                else:
                    w = Gen_mapping(z, y)
                    images = Gen_synthesis(w)
            else:
                # 标准 GAN
                images = generator(z, y)

            all_images.append(images.cpu())

    # 合并所有批次
    all_images = torch.cat(all_images, dim=0)[:num_images]

    return all_images


def save_images_grid(images, save_path, nrow=8):
    """保存图片网格"""
    os.makedirs(dirname(save_path), exist_ok=True)
    save_image(images, save_path, nrow=nrow, normalize=True, value_range=(-1, 1))
    print(f"Saved to: {save_path}")


def save_images_individual(images, save_dir, class_id, start_idx=0):
    """保存单张图片"""
    os.makedirs(save_dir, exist_ok=True)
    class_name = CLASS_NAMES[class_id]

    for i, img in enumerate(images):
        save_path = join(save_dir, f"{class_name}_{start_idx + i:05d}.png")
        save_image(img, save_path, normalize=True, value_range=(-1, 1))

    print(f"Saved {len(images)} individual images to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate class-conditional images')

    # 基本参数
    parser.add_argument('-cfg', '--config', type=str,
                       default='src/configs/medical_imaging/stylegan3_brain_stroke_ir5_256.yaml',
                       help='配置文件路径')
    parser.add_argument('-ckpt', '--checkpoint', type=str,
                       default='./results/brain_stroke_ir5_stylegan3_256/checkpoints',
                       help='检查点目录路径')
    parser.add_argument('-o', '--output', type=str, default='./generated_images',
                       help='输出目录')

    # 生成参数
    parser.add_argument('-c', '--class_id', type=int, choices=[0, 1, 2],
                       help='目标类别: 0=Normal, 1=Ischemia, 2=Bleeding')
    parser.add_argument('-n', '--num_images', type=int, default=64,
                       help='生成图片数量')
    parser.add_argument('--all', action='store_true',
                       help='生成所有类别（忽略 -c 参数）')

    # 高级参数
    parser.add_argument('--truncation', type=float, default=-1.0,
                       help='Truncation factor (0-1, -1 表示不使用). 推荐: 0.7')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--individual', action='store_true',
                       help='保存单张图片（而不是网格）')
    parser.add_argument('--nrow', type=int, default=8,
                       help='网格每行图片数')
    parser.add_argument('--best', action='store_true', default=True,
                       help='加载最佳模型（默认）')
    parser.add_argument('--current', action='store_true',
                       help='加载当前模型（而不是最佳）')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='设备 (cuda:0, cuda:1, cpu)')

    args = parser.parse_args()

    # 检查参数
    if not args.all and args.class_id is None:
        parser.error("必须指定 -c/--class_id 或使用 --all")

    # 加载配置
    print("=" * 80)
    print("类别条件生成脚本")
    print("=" * 80)

    if not exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")

    if not exists(args.checkpoint):
        raise FileNotFoundError(f"检查点目录不存在: {args.checkpoint}")

    cfgs = load_config(args.config)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    load_best = not args.current
    Gen, Gen_ema, Gen_mapping, Gen_synthesis = load_models(
        cfgs, args.checkpoint, device, load_best=load_best
    )

    # 确定要生成的类别
    if args.all:
        class_ids = [0, 1, 2]
        print(f"\n生成所有类别，每类 {args.num_images} 张")
    else:
        class_ids = [args.class_id]
        print(f"\n生成类别 {args.class_id} ({CLASS_NAMES[args.class_id]} / {CLASS_NAMES_ZH[args.class_id]})")

    print(f"数量: {args.num_images}")
    print(f"Truncation: {args.truncation if args.truncation > 0 else '不使用'}")
    print(f"输出目录: {args.output}")
    print()

    # 生成图片
    os.makedirs(args.output, exist_ok=True)

    for class_id in class_ids:
        images = generate_class_images(
            cfgs=cfgs,
            Gen=Gen,
            Gen_ema=Gen_ema,
            Gen_mapping=Gen_mapping,
            Gen_synthesis=Gen_synthesis,
            class_id=class_id,
            num_images=args.num_images,
            truncation_factor=args.truncation,
            device=device,
            batch_size=args.batch_size
        )

        # 保存图片
        class_name = CLASS_NAMES[class_id]

        if args.individual:
            # 保存单张图片
            output_dir = join(args.output, class_name)
            save_images_individual(images, output_dir, class_id)
        else:
            # 保存网格
            output_path = join(args.output, f'{class_name}_grid.png')
            save_images_grid(images, output_path, nrow=args.nrow)

        print(f"✓ 类别 {class_id} ({class_name}) 完成\n")

    print("=" * 80)
    print("生成完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
