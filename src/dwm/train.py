import argparse # 解析命令行参数
import dwm.common
import json
import os
from tqdm import tqdm # 显示训练进度条
import debugpy # 远程调试python程序
import torch
from dwm.utils.sampler import VariableVideoBatchSampler #自定义采样器


def create_parser():
    parser = argparse.ArgumentParser(
        description="The script to finetune a stable diffusion model to the "
        "driving dataset.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="The config to load the train model and dataset.")
    parser.add_argument(
        "-o", "--output-path", type=str, default=None,
        help="The path to save checkpoint files.")
    parser.add_argument(
        "--log-steps", default=100, type=int,
        help="The step count to print log and update the tensorboard.")
    parser.add_argument(
        "--preview-steps", default=200, type=int,
        help="The step count to preview the pipeline result.")# 预览结果 
    parser.add_argument(
        "--checkpointing-steps", default=1000, type=int,# 保存模型
        help="The step count to save the checkpoint.")
    parser.add_argument(
        "--evaluation-steps", default=10000, type=int,
        help="The step count to preview the pipeline result.")# 完整评估
    parser.add_argument(
        "--resume-from", default=None, type=int,#意外情况 恢复训练
        help="The step to resume from")
    parser.add_argument(
        "--wandb", action="store_true",
        help="Use wandb to log the training process.")# 是否使用wandb记录训练过程
    parser.add_argument(
        "--wandb-project", type=str, default="dwm",# 项目名称
        help="The wandb project name.")
    parser.add_argument(
        "--wandb-run-name", type=str, default="train", # 运行名称（模式）
        help="The wandb run name.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args() # 解析命令行参数

    debugpy.listen(("0.0.0.0", 9876)) # 启动远程调试服务器，监听所有IP地址的9876端口
    print("[debugpy] listening on, waiting for VS Code to attach...")
    debugpy.wait_for_client() #在这里设置断点       
    print("attached")
    
    with open(args.config_path, "r", encoding="utf-8") as f:# 读取模型和数据集配置
        config = json.load(f)

    torch.manual_seed(config["generator_seed"]) # 设置随机数种子，确保结果可复现

    # set distributed training (if enabled), log, random number generator, and
    # load the checkpoint (if required).
    #ddp（Distributed Data Parallel）分布式数据并行训练，适用于多GPU训练
    ddp = "LOCAL_RANK" in os.environ # 通过环境变量LOCAL_RANK来确定是否启用分布式训练
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])# 当进入进程时，LOCAL_RANK会被设置为当前进程使用的GPU的索引，例如0、1、2等。通过读取这个环境变量，可以确定当前进程应该使用哪个GPU进行训练。
        device = torch.device(config["device"], local_rank)# 根据配置文件和环境变量设置设备（GPU）
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)

        torch.distributed.init_process_group(backend=config["ddp_backend"])# 各显卡之间实现通信
    else:
        device = torch.device(config["device"])

    # setup the global state
    if "global_state" in config: # 配置文件中包含全局状态的定义
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value) # 根据配置文件创建全局状态实例，并存储在dwm.common.global_state字典中，供训练过程中使用

    should_log = (ddp and local_rank == 0) or not ddp # 主进程才能写日志
    should_save = not torch.distributed.is_initialized() or \
        torch.distributed.get_rank() == 0# 主进程才能保存模型

    # load the pipeline including the models
    output_path = config["output_path"] if args.output_path is None else args.output_path
    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=output_path, config=config,
        device=device, resume_from=args.resume_from)

    if should_log:
        print("The pipeline is loaded.")

    if args.wandb and should_save:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)

    # load the dataset
    training_dataset = dwm.common.create_instance_from_config(
        config["training_dataset"])
    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if ddp:

        if "mix_config" in config.keys(): #要采用分桶策略 将不同尺寸的视频放到一起
            process_group = torch.distributed.group.WORLD # 获取全局通信组，包含所有参与分布式训练的进程

            training_datasampler = VariableVideoBatchSampler(#负责数据排序分组，减少batch内数据尺寸差异
                training_dataset,
                config["mix_config"],
                num_replicas=process_group.size(), # 进程总数
                rank=process_group.rank(),
                shuffle=config["data_shuffle"], # 是否打乱数据顺序
                seed=config["generator_seed"]
            )

            training_dataloader = torch.utils.data.DataLoader(#数据加载，负责按batch采样数据并进行预处理
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                batch_sampler=training_datasampler)

        else: # 没有视频拼接 直接内置函数取样
            training_datasampler = torch.utils.data.distributed.DistributedSampler(
                training_dataset, shuffle=config["data_shuffle"],
                seed=config["generator_seed"])
            training_dataloader = torch.utils.data.DataLoader(
                training_dataset,
                **dwm.common.instantiate_config(config["training_dataloader"]),
                sampler=training_datasampler)

        # make equal sample count for each process to simplify the result
        # gathering
        total_batch_size = int(os.environ["WORLD_SIZE"]) * \
            config["validation_dataloader"]["batch_size"]# 所有进程一起处理的数据量
        dataset_length = len(validation_dataset) // \
            total_batch_size * total_batch_size # 把不足一轮全体batch的部分（余数）丢弃，保证每个进程处理的数据量相同
        validation_dataset = torch.utils.data.Subset(
            validation_dataset, range(0, dataset_length))
        validation_datasampler = \
            torch.utils.data.distributed.DistributedSampler(
                validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"]),
            sampler=validation_datasampler)
    else:
        training_dataloader = torch.utils.data.DataLoader(
            training_dataset,
            **dwm.common.instantiate_config(config["training_dataloader"]),
            shuffle=config["data_shuffle"])
        validation_datasampler = None
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["validation_dataloader"])) #**表示字典解包 把数据加载的相关参数传入DataLoader

    preview_dataloader = torch.utils.data\
        .DataLoader(
            validation_dataset,
            **dwm.common.instantiate_config(config["preview_dataloader"])) if \
        "preview_dataloader" in config else None
    if preview_dataloader is not None:
        preview_data_iterator = iter(preview_dataloader)

    if should_log:
        print("The training dataset is loaded with {} items.".format(
            len(training_dataset)))
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    # train loop
    global_step = 0 if args.resume_from is None else args.resume_from
    for epoch in range(config["train_epochs"]):

        if ddp:
            # Fixing training data order reduces the accessed objects per rank,
            # therefore reduces the upper-bound of memory usage comsumed by the
            # Python reference counting of objects.
            sampler_epoch = 0 if config.get("fix_training_data_order", False) \
                else epoch#这个变量是种子的一个组成部分，决定了每个epoch的数据顺序是否固定。如果fix_training_data_order为True，则sampler_epoch始终为0，保证每个epoch的数据顺序相同；如果为False，则sampler_epoch等于当前的epoch数，使得每个epoch的数据顺序不同。通过设置不同的sampler_epoch值，可以控制训练数据的随机性和多样性，从而影响模型的训练效果和泛化能力。
            training_datasampler.set_epoch(sampler_epoch)
            
        loader = tqdm(
            training_dataloader,
            total=len(training_dataloader),
            desc=f"Epoch {epoch}",
            dynamic_ncols=True#根据终端宽度动态调整进度条的长度
        )

        for batch in loader:
            pipeline.train_step(batch, global_step)
            global_step += 1

            # log
            if global_step % args.log_steps == 0:
                pipeline.log(global_step, args.log_steps)

            # preview
            if global_step % args.preview_steps == 0:
                if preview_dataloader is None:
                    pipeline.preview_pipeline(batch, output_path, global_step)
                else:
                    try:
                        preview_batch = next(preview_data_iterator)
                    except StopIteration:
                        preview_data_iterator = iter(preview_dataloader)
                        preview_batch = next(preview_data_iterator)

                    pipeline.preview_pipeline(
                        preview_batch, output_path, global_step)

            # save step checkpoint
            if global_step % args.checkpointing_steps == 0:
                pipeline.save_checkpoint(output_path, global_step)

            # evaluation
            if (
                args.evaluation_steps > 0 and
                global_step % args.evaluation_steps == 0
            ):
                pipeline.evaluate_pipeline(
                    global_step, len(validation_dataset),
                    validation_dataloader, validation_datasampler)

        if should_log:
            print("Epoch {} done.".format(epoch))

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group() # 销毁分布式训练的通信组，释放资源
