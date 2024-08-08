import os
import torch
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.compat.proto import event_pb2
from tensorboardX import SummaryWriter

def modify_events(logdir, new_logdir, dx):
    if not os.path.exists(new_logdir):
        os.makedirs(new_logdir)
    
    for event_file in os.listdir(logdir):
        if event_file.startswith('events.out.tfevents'):
            ea = event_accumulator.EventAccumulator(os.path.join(logdir, event_file))
            ea.Reload()

            writer = SummaryWriter(new_logdir)

            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                for event in events:
                    new_step = event.step + dx
                    writer.add_scalar(tag, event.value, new_step)

            writer.close()

def truncate_events(logdir, truncated_logdir, max_step):
    if not os.path.exists(truncated_logdir):
        os.makedirs(truncated_logdir)

    writer = SummaryWriter(truncated_logdir)

    for event_file in os.listdir(logdir):
        if event_file.startswith('events.out.tfevents'):
            ea = event_accumulator.EventAccumulator(os.path.join(logdir, event_file))
            ea.Reload()

            for tag in ea.Tags()['scalars']:
                events = ea.Scalars(tag)
                for event in events:
                    if event.step <= max_step:
                        writer.add_scalar(tag, event.value, event.step)

    writer.close()


# 输入和输出日志目录
input_logdir = 'logs_change1'
output_logdir = 'logs_change'
dx = 35800  # 平移的距离
max_step = 35850  # 将其更改为你想截断的 step

# 执行截断操作
truncate_events(input_logdir, output_logdir, max_step)

# # 执行平移操作
# modify_events(input_logdir, output_logdir, dx)
