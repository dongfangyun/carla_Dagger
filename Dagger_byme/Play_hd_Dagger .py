"""DQN Synchronous Train Torch version"""
from threading import Thread
import os
import glob # 用于添加carla.egg。环境中装有.whl可忽略
import sys
import random
import time
from collections import deque # 双端队列

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm # 在Python长循环中添加一个进度提示信息，用户只需要封装任意的迭代器tqdm(iterator)

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

from Dagger_CarEnv import CarEnv, IM_WIDTH, IM_HEIGHT,camera_queue1, camera_queue2 

# log_dir = r"Dagger_model/model_Tue_Jul_30_12_57_30_2024.pth"
# log_dir = r"IL_experience_model/model_Sun_Jul_28_16_43_50_2024.pth"
# log_dir = r"Dagger_model/model_Wed_Jul_31_21_38_58_2024.pth"
log_dir = r"Dagger_model/model_Sat_Aug__3_13_04_53_2024.pth"

SHOW_PREVIEW = False # 训练时播放摄像镜头
LOG = False # 训练时向tensorboard中写入记录

REPLAY_MEMORY_SIZE = 1000 # 经验回放池最大容量
MIN_REPLAY_MEMORY_SIZE = 600# 抽样训练开始时经验回放池的最小样本数
MINIBATCH_SIZE = 8 # 每次从经验回放池的采样数（作为训练的同一批次）   此大小影响运算速度/显存
EPISODES = 100 # 游戏进行总次数

if LOG:
    writer = SummaryWriter("./logs_play_hd_DDPG")


class Policynet_cat_fc_pro(nn.Module):
    def __init__(self, IM_HEIGHT, IM_WIDTH):
        super(Policynet_cat_fc_pro, self).__init__() 
        # images的卷积层+全连接层
        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(7),
            nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            # nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.BatchNorm2d(64),
        )
        self.conv_fc1 = nn.Sequential(
            nn.Linear(int(64 * (IM_HEIGHT/8) * (IM_WIDTH/8)), 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.conv_fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )

        # attributes全连接层
        self.bn1 = nn.BatchNorm1d(20)
        self.fc1 = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        
        # 拼接后的全连接层 32 + 32 = 64 --> 32 -->16 -->2
        self.cat_fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc4 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc5 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
        )
        self.cat_fc6 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(16, 2), 
            nn.Tanh()
        )

    def forward(self, images, attributes):
        conv1_out = self.conv1(images) 
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv3_res = conv3_out.reshape(conv3_out.size(0), -1) # --> (76800)
        conv_fc1_out = self.conv_fc1(conv3_res) # 76800 --> (64)
        conv_fc2_out = self.conv_fc2(conv_fc1_out) # (32)
        # print("conv_fc2_out", conv_fc2_out.shape) #torch.Size([64, 32])

        attributes = self.bn1(attributes) # 将20个特征先批归一化
        fc1_out = self.fc1(attributes) # 20 --> 64
        fc2_out = self.fc2(fc1_out) # 64 --> 32
        fc3_out = self.fc3(fc2_out) # 32 --> 32
        # print("fc3_out", fc3_out.shape) # torch.Size([64, 32])

        cat = torch.cat(( conv_fc2_out, fc3_out), 1) # 32 + 32 = 64 --> 32
        # print("cat", cat.shape) # torch.Size([64, 64])
        cat_fc1_out = self.cat_fc1(cat) # 32 --> 16
        cat_fc2_out = self.cat_fc2(cat_fc1_out) # 16 --> 2 
        cat_fc3_out = self.cat_fc3(cat_fc2_out) # 
        cat_fc4_out = self.cat_fc4(cat_fc3_out) # 
        cat_fc5_out = self.cat_fc5(cat_fc4_out) # 
        cat_fc6_out = self.cat_fc6(cat_fc5_out) # 

        return cat_fc6_out # (batch, 2)
    

if torch.cuda.device_count() > 1:
    device = torch.device("cuda:0") 
else:
    device = torch.device("cuda:0")

class Dagger:
    def __init__(self,lr_actor=1e-4):

        self.actor = Policynet_cat_fc_pro(IM_HEIGHT, IM_WIDTH).to("cuda:0")    

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor) 

        # if os.path.exists(log_dir): # 模仿学习模型 加if，路径写错，直接跳过读取了。相当于在用个初始网络在跑
        checkpoint = torch.load(log_dir)
        self.actor.load_state_dict(checkpoint['model'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))    

        # if os.path.exists(log_dir): # 强化学习模型
        #     self.actor = torch.load(log_dir)

        self.sigma = 0.1 #高斯噪声标准差

        self.terminate = False #没结束循环训练，当全部游戏次数跑完后此处会改为True，停止进程中的采样训练
        self.training_initialized = False

    def get_action(self, images, attributes): # state(16,c,h,w)
        self.actor.eval() # 播放模式无需梯度计算，不打开.train()
        with torch.no_grad():
            images = torch.tensor(images, dtype=torch.float).to(device) # (hwc)
            images = images/255 
            images = images.permute(2, 0, 1) # (chw)
            images = images.unsqueeze(dim=0) # (n, c, h, w) 

            action = self.actor( images, attributes)# action(batch, 2)a
            
        return action # action(batch, 2)

def caculate_reward(dist_to_start, dist_to_start_old, kmh_player, done, inva_lane, action):
    reward = 0.0
    reward += np.clip(dist_to_start-dist_to_start_old, -10.0, 10.0) 
    # reward += (new_dis-dis)*1
    # reward +=(new_kmh - kmh)
    reward +=(kmh_player) * 0.05
    if done: #撞击
        reward += -10
    if inva_lane: #跨道
        print("invasion lane") 
        reward += -10
    if kmh_player < 1:
        reward += -1
    if action[0][0] > 0.2:
        reward += 1
    return reward

if __name__ == '__main__':

    # Create agent and environment
    env = CarEnv()    
    agent = Dagger()

    Original_settings = env.original_settings # 将原设置传出来保存
    # sensor_queue1 = Queue()
    
    now = time.ctime(time.time())
    now = now.replace(" ","_").replace(":", "_")

    episode_num = 0 # 游戏进行的次数

    all_average_reward = 0
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'): # 1~100 EPISODE
        env.collision_hist = [] # 记录碰撞发生的列表
        episode_reward = 0 # 每次游戏所有step累计奖励
        
        # Reset environment and get initial state
        env.reset() #旧版处reset里会先tick()一下，往队列里传入初始图像 这里面没tick()
        # 手动tick()一下
        # env.world.tick() 

        # 模仿学习、Dagger及播放模型均无需初始帧以及上一状态

        # current_state1 = camera_queue1.get() # <class 'carla.libcarla.Image'> Image(frame=154824, timestamp=1589.824231, size=640x480)
        # i_1 = np.array(current_state1.raw_data) #(1228800,) = 640 X 480 X 4   	.raw_data：Array of BGRA 32-bit pixels
        # i2_1 = i_1.reshape((IM_HEIGHT, IM_WIDTH, 4))
        # i3_1 = i2_1[:, :, :3] # （h, w, 3）= (480, 640, 3)

        # current_state2 = camera_queue2.get() # <class 'carla.libcarla.Image'> Image(frame=154824, timestamp=1589.824231, size=640x480)
        # current_state2.convert(carla.ColorConverter.CityScapesPalette)
        # i_2 = np.array(current_state2.raw_data) #(1228800,) = 640 X 480 X 4
        # i2_2 = i_2.reshape((IM_HEIGHT, IM_WIDTH, 4))
        # i3_2 = i2_2[:, :, :3] # （h, w, 3）= (480, 640, 3)  # （h, w, 3）= (480, 640, 3)

        # dis_to_start_old = 0
        # current_state = np.concatenate((i3_1, i3_2), axis=2) # （h, w, 3）= (480, 640, 3+3+1 = 7)

        action = torch.tensor([[0.0, 0.0]])  # torch.Size([1, 2]) # 给个初始动作其实就可以了
        # location = np.array([0, 0, 0]) # (3,)
        # start_point = np.array([0, 0, 0]) # (3,)
        # destination = np.array([0, 0, 0]) # (3,)
        # forward_vector = np.array([0, 0, 0]) # (3,)
        # velocity = 0
        # acceleration = np.array([0, 0, 0]) # (3,)
        # angular_velocity = np.array([0, 0, 0]) # (3,)
        # reward =0

        # data = [[*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity, reward]] # 预留batch_sizee维度

        # data = torch.tensor(data).cuda().float() # ( ,20)

        # done = False
        
        #以上为初始帧的

        episode_start = time.time()
        episode_num += 1
        episode_steps = 0

        # Play for given number of seconds only
        while True:
            #synchronous
            env.world.tick()

            new_state1 = camera_queue1.get()
            i_1 = np.array(new_state1.raw_data) # .raw_data：Array of BGRA 32-bit pixels
            #print(i.shape)
            i2_1 = i_1.reshape((IM_HEIGHT, IM_WIDTH, 4))
            i3_1 = i2_1[:, :, :3] # （h, w, 3）= (480, 640, 3)
            if SHOW_PREVIEW:
                cv2.imshow("i3_1", i3_1)
                cv2.waitKey(1)

            new_state2 = camera_queue2.get()
            new_state2.convert(carla.ColorConverter.CityScapesPalette)
            i_2 = np.array(new_state2.raw_data) # .raw_data：Array of BGRA 32-bit pixels
            #print(i.shape)
            i2_2 = i_2.reshape((IM_HEIGHT, IM_WIDTH, 4))
            i3_2 = i2_2[:, :, :3] # （h, w, 3）= (480, 640, 3)
            if SHOW_PREVIEW:
                cv2.imshow("i3_2", i3_2)
                cv2.waitKey(1)

            current_state = np.concatenate((i3_1, i3_2), axis=2) # （h, w, 3+3）= (240, 320, 6)

            # reward, done, _ = env.step(action)
            done, data, act_expert, dis_to_start, inva_lane= env.step(action, episode_steps)
            reward = caculate_reward(dis_to_start, dis_to_start, data[-7], done, inva_lane, action)
            data.append(reward)
            data = [data] # data预留batch_size维度 (, 20)
            # print(data)
            data = torch.tensor(data).cuda().float()

            action = agent.get_action(current_state, data) # 训练的agent开车鉴赏
            # action = act_expert  # 专家开车鉴赏模式

            # 分离损失函数，以便加权损失
            loss_fn_throttle = nn.L1Loss(reduction='mean')
            loss_fn_steer = nn.L1Loss(reduction='mean')
            loss_fn_throttle = loss_fn_throttle.cuda()
            loss_fn_steer = loss_fn_steer.cuda()

            loss_throttle = loss_fn_throttle(action[:, 0], act_expert[:, 0])
            loss_steer = loss_fn_steer(action[:, 1], act_expert[:, 1])
            # loss = loss_throttle + weight_loss_steer * loss_steer # 方向盘损失权重放大100倍

            # print(action, act_expert)
            # print(loss_throttle, 100 * loss_steer)

            episode_reward += reward

            # set the sectator to follow the ego vehicle
            spectator = env.world.get_spectator()
            transform = env.vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                carla.Rotation(pitch=-90)))
            
            episode_steps += 1

            print(episode_steps)

            if done:
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        if LOG:
            writer.add_scalar("avearge_reward_{}".format(now), episode_reward/episode_steps, episode_num)
        
        all_average_reward += episode_reward/episode_steps

    print("all_average_reward", all_average_reward/EPISODES)

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    env.world.apply_settings(Original_settings)
