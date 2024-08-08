import os
from tensorboard.backend.event_processing import event_accumulator
from tensorboardX import SummaryWriter

def merge_events(logdir1, logdir2, merged_logdir):
    if not os.path.exists(merged_logdir):
        os.makedirs(merged_logdir)

    writer = SummaryWriter(merged_logdir)

    def add_events_from_logdir(logdir):
        for event_file in os.listdir(logdir):
            if event_file.startswith('events.out.tfevents'):
                ea = event_accumulator.EventAccumulator(os.path.join(logdir, event_file))
                ea.Reload()

                for tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    for event in events:
                        writer.add_scalar("merged", event.value, event.step) # 此处定义融合后图的名字

    add_events_from_logdir(logdir1)
    add_events_from_logdir(logdir2)

    writer.close()

# 输入日志目录
logdir1 = 'logs_change1'
logdir2 = 'logs_change2'
merged_logdir = 'logs_change'

# 执行合并操作
merge_events(logdir1, logdir2, merged_logdir)