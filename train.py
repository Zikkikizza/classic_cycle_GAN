

# 通用的图像到图像转换训练脚本。
# 这个脚本支持多种模型（通过'--model'选项，例如：pix2pix, cyclegan, colorization）和不同的数据集（通过'--dataset_mode'选项，例如：aligned, unaligned, single, colorization）。
# 你需要指定数据集（'--dataroot'）、实验名称（'--name'）和模型（'--model'）。
# 首先，根据给定的选项创建模型、数据集和可视化工具。
# 然后进行标准的网络训练。在训练过程中，它还会可视化/保存图像，打印/保存损失图，并保存模型。
# 该脚本支持继续/恢复训练。使用'--continue_train'来恢复之前的训练。

"""
Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md

--dataroot ./datasets/maps
--name maps_cyclegan
--model cycle_gan

python -m visdom.server

python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
"""


# 导入所需的库和模块
import time
from options.train_options import TrainOptions  # 导入训练选项模块
from data import create_dataset  # 导入创建数据集的函数
from models import create_model  # 导入创建模型的函数
from util.visualizer import Visualizer  # 导入可视化工具

#主函数开始
if __name__ == '__main__':
    opt = TrainOptions().parse()   # 获取训练选项
    dataset = create_dataset(opt)  # 根据选项创建数据集
    dataset_size = len(dataset)    # 获取数据集中图像的数量
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # 创建模型 根据 opt.model and other options
    model.setup(opt)               # 常规设置：加载和打印网络；创建调度器
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # 训练迭代的总次数

    """
    训练通常分为多个"周期"（Epochs）和"迭代"（Iterations）
    外循环用于控制整体的训练过程，而内循环用于具体的模型参数更新。
    
    1、  外循环（Epoch循环）：这个循环遍历整个数据集一次。
        每次遍历整个数据集称为一个"周期"（Epoch）。在每个周期结束后，通常会更新学习率、保存模型状态等。
    """
    # 对不同的训练周期进行外循环
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # 记录整个周期的开始时间
        iter_data_time = time.time()  # 记录每次迭代数据加载的时间
        epoch_iter = 0  # 当前周期内的训练迭代次数，每个周期重置为0
        visualizer.reset()  # 重置可视化工具
        model.update_learning_rate()  # 在每个周期开始时更新学习率

        """
        2、内循环（Iteration循环）：这个循环遍历数据集的一个小批量（Batch）。
            每次遍历一个小批量称为一个"迭代"（Iteration）。
            在每个迭代中，模型的参数会根据损失函数进行更新。
        """
        # 在一个周期内进行内循环
        for i, data in enumerate(dataset): #dataset 是一个批量数据生成器，每次产生一个小批量的数据。
            iter_start_time = time.time()  # 记录每次迭代计算的开始时间
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time  # 计算数据加载时间

            total_iters += opt.batch_size  # 更新总迭代次数
            epoch_iter += opt.batch_size  # 更新当前周期内的迭代次数
            model.set_input(data)  # 从数据集中解包数据并应用预处理
            model.optimize_parameters()  # 计算损失函数，获取梯度，更新网络权重

            # 在Visdom上显示图像并保存图像到HTML文件
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 打印训练损失并将日志信息保存到磁盘
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # 判断是否需要保存最新的模型
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))  # 打印保存模型的信息
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'  # 根据是否按迭代次数保存来设置保存后缀
                model.save_networks(save_suffix)  # 调用模型的保存函数，保存模型

            # 更新数据加载的时间，用于下一次迭代的数据加载时间计算
            iter_data_time = time.time()

        # 每 <save_epoch_freq> 周期缓存我们的模型
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters)) # 打印保存模型的信息
            model.save_networks('latest') # 保存最新的模型
            model.save_networks(epoch)# 以当前周期数作为后缀保存模型

        # 打印当前周期结束后的信息，包括总周期数和该周期所需的时间
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
"""

opt.n_epochs：这是模型进行正常训练（即学习率保持不变）的周期数。
opt.n_epochs_decay：这是模型进行学习率衰减训练的周期数。
为了让模型在训练后期更加稳定。

在第 1 到第 100 个周期（Epoch 1-100）：模型使用初始学习率进行训练（正常训练）。
在第 101 到第 150 个周期（Epoch 101-150）：模型的学习率会逐渐衰减（衰减训练）。

opt.n_epochs + opt.n_epochs_decay  表示模型的总训练周期数，包括正常训练和衰减训练的周期数。
opt.n_epochs + opt.n_epochs_decay = 150，意味着模型将总共进行 150 个训练周期。
"""