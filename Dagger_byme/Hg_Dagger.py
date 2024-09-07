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

"""
    Policynet():
    input: state([batch, 7,  height, width])
    return: action([batch, 2])
"""

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

def caculate_reward(dist_to_start, dist_to_start_old, kmh_player, done, inva_lane, action):
    reward = 0.0
    reward += np.clip(dist_to_start-dist_to_start_old, -10.0, 10.0) 
    # reward += (new_dis-dis)*1
    # reward +=(new_kmh - kmh)
    reward +=(kmh_player) * 0.05
    if done: #撞击
        reward += -10
    if inva_lane: #跨道
        # print("invasion lane") 
        reward += -10
    if kmh_player < 1:
        reward += -1
    if action[0][0] > 0.2:
        reward += 1
    return reward

class Dagger:
    def __init__(self,lr_actor=1e-4):

        self.actor = Policynet_cat_fc_pro(IM_HEIGHT, IM_WIDTH).to("cuda:0")

        # if torch.cuda.device_count() > 1:
        #     self.actor= nn.DataParallel(self.actor, device_ids = [0,1], output_device=device)

        #只更新主网络的参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor) 

        #启动预训练模型
        if TRAINED_MODEL:
            checkpoint = torch.load(trained_model_dir )
            self.actor.load_state_dict(checkpoint['model'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 结果成功！'.format(start_epoch))   
        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) #上限12000的经验回放池队列
        self.terminate = False #没结束循环训练，当全部游戏次数跑完后此处会改为True，停止进程中的采样训练

    def get_action(self, images, attributes): # state(16,c,h,w)
        self.actor.eval() # 提取动作时无需反向传播，关闭梯度节省显存。！！！有个问题，双线程时关闭梯度会影响线程2的模型反向传播训练
        with torch.no_grad():
            images = torch.tensor(images, dtype=torch.float).to(device) # (hwc)
            images = images/255 
            images = images.permute(2, 0, 1) # (chw)
            images = images.unsqueeze(dim=0) # (n, h, w, c) 

            action = self.actor(images, attributes)# action(batch, 2)
            
        self.actor.train() # 获取动作后开启梯度计算
        return action # action(batch, 2)

    def update_replay_memory(self, transition): 
        # transition = (current_state, action, act_expert, done)
        self.replay_memory.append(transition)
        # 初始训练集可在此添加

    def train(self):
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) #每次采样16个轨迹作为一批次同时训练
        # transition: (current_state, action, act_expert, done, data)
        # current_state: (240, 320, 6) 
        # data = [*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity, reward]: (, 20)

        current_states = []
        actions = []
        actions_expert = []
        attributes = []

        for transition in minibatch:
            # print(transition[1]) # tensor([[ 0.5837, -0.0024]], device='cuda:0')
            current_states.append(transition[0])
            actions.append(*transition[1].tolist())
            actions_expert.append(*transition[2].tolist())
            attributes.append(*transition[4].tolist()) 

        # 16个随机样本的current_states: 16 * (h, w, c) --> (16, c, h, w)
        current_states = np.array(current_states)/255 #(16, h, w, c)   这里看后期能不能优化掉，直接list[tensor()]/255
        current_states = torch.from_numpy(current_states)  #Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        current_states = current_states.to(torch.float32).to(device)
        current_states = current_states.permute(0,3,1,2) # (16, c, h, w)

        actions= torch.tensor(actions).cuda() # 实际执行的action，没啥用。大循环里可以看实际每一步的loss，# actions: torch.Size([16, 2])
        attributes = torch.tensor(attributes).cuda() # attributes: torch.Size([16, 2])

        actions_agent = self.actor(current_states, attributes)
        actions_expert = torch.tensor(actions_expert).cuda() #
        
        # 分离损失函数，以便加权损失
        loss_fn_throttle = nn.L1Loss(reduction='mean')
        loss_fn_steer = nn.L1Loss(reduction='mean')
        loss_fn_throttle = loss_fn_throttle.cuda()
        loss_fn_steer = loss_fn_steer.cuda()

        loss_throttle = loss_fn_throttle(actions_agent[:, 0], actions_expert[:, 0])
        loss_steer = loss_fn_steer(actions_agent[:, 1], actions_expert[:, 1])
        actor_loss = loss_throttle + 1 * loss_steer # 方向盘损失权重放大100倍

        # print(action, act_expert, loss_throttle, 100 * loss_steer) 

        self.actor_optimizer.zero_grad() # 梯度清零
        actor_loss.backward() # 产生梯度 同时对critic和actor产生梯度！其实不影响。只更critic前会清零
        self.actor_optimizer.step() # 使用target_critic做loss后，actor和critic更新的先后顺序也无所谓了

        return actor_loss.item()
    
    def train_in_loop(self):
        total_train_step = 0 # 在此续训练步数
        while True:
            if self.terminate: # 未完成游戏训练前，此处会一直支线程进行训练
                return
            
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                print(len(self.replay_memory))
                # return

            else: # 当经验回放池样本数量大于MIN_REPLAY_MEMORY_SIZE才会开始采样训练
                actor_loss = self.train()
                # print("loss:", actor_loss)
                total_train_step += 1
                print("total_train_step", total_train_step)

                if total_train_step > 20100: # 超过20000步自动结束
                    self.terminate = True

                if LOG:
                    writer.add_scalar("loss", actor_loss, total_train_step)

                if total_train_step % 1000 == 0:
                    if SAVE:
                        state = {'model':agent.actor.state_dict(), 'optimizer':agent.actor_optimizer.state_dict(), 'epoch': episode}
                        torch.save(state, path)

if __name__ == '__main__':

    now = time.ctime(time.time())
    now = now.replace(" ","_").replace(":", "_")

    SHOW_PREVIEW = False # 训练时播放摄像镜头

    LOG = True # 训练时向tensorboard中写入记录

    SAVE = True # 保存模型

    if SAVE:
        if not os.path.isdir('./Dagger_model'):
            os.makedirs('./Dagger_model')
        path='./Dagger_model/model_{}.pth'.format(now)

    if LOG:
        if not os.path.isdir('./logs/{}'.format(now)):
            os.makedirs('./logs/{}'.format(now))
        writer = SummaryWriter('./logs/{}'.format(now))


    TRAINED_MODEL =False# 是否有预训练模型
    trained_model_dir = r"IL_experience_model/model_Sun_Jul_28_16_43_50_2024.pth" # 
    # trained_model_dir = r"Dagger_model/model_Tue_Aug_13_12_55_57_2024.pth" # 

    REPLAY_MEMORY_SIZE = 10000 # 经验回放池最大容量——足以容纳预训练/或不需要
    MIN_REPLAY_MEMORY_SIZE = 500# 抽样训练开始时经验回放池的最小样本数
    MINIBATCH_SIZE = 256 # 每次从经验回放池的采样数（作为训练的同一批次）   此大小影响运算速度/显存
    EPISODES = 5000 # 游戏进行总次数

    # Create agent and environment
    env = CarEnv()    
    agent = Dagger()

    Original_settings = env.original_settings # 将原设置传出来保存

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True) #创建一个线程，调用的函数是train_in_loop，此处是daemon，理论上主线程结束了会立即结束所有支线程
    trainer_thread.start() #此处会直接往下走，同时线程分支（train_in_loop，训练一批）开始运行

    episode_num = 0 # 游戏进行的次数

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'): # 1~100 EPISODE
        # env.collision_hist = [] # 记录碰撞发生的列表

        # Reset environment and get initial state
        env.reset() #此处reset里会先tick()一下，往队列里传入初始图像
        # env.world.tick()

        action = torch.tensor([[0, 0]]).to(device)  # torch.Size([1, 2])

        episode_start = time.time()
        episode_num += 1
        episode_steps = 0

        # Play for given number of seconds only
        while True:
            #synchronous
            env.world.tick()

            current_state1 = camera_queue1.get()
            i_1 = np.array(current_state1.raw_data) # .raw_data：Array of BGRA 32-bit pixels
            #print(i.shape)
            i2_1 = i_1.reshape((IM_HEIGHT, IM_WIDTH, 4))
            i3_1 = i2_1[:, :, :3] # （h, w, 3）= (480, 640, 3)
            if SHOW_PREVIEW:
                cv2.imshow("i3_1", i3_1)
                cv2.waitKey(1)

            current_state2 = camera_queue2.get()
            current_state2.convert(carla.ColorConverter.CityScapesPalette)
            i_2 = np.array(current_state2.raw_data) # .raw_data：Array of BGRA 32-bit pixels
            #print(i.shape)
            i2_2 = i_2.reshape((IM_HEIGHT, IM_WIDTH, 4))
            i3_2 = i2_2[:, :, :3] # （h, w, 3）= (480, 640, 3)
            if SHOW_PREVIEW:
                cv2.imshow("i3_2", i3_2)
                cv2.waitKey(1)

            current_state = np.concatenate((i3_1, i3_2), axis=2) # （h, w, 3+3+1）= (480, 640, 7) 

            # reward, done, _ = env.step(action)
            done, data, act_expert, dis_to_start, inva_lane= env.step(action, episode_steps)
            reward = caculate_reward(dis_to_start, dis_to_start, data[-7], done, inva_lane, action)
            data.append(reward)
            data = [data] # data预留batch_size维度 (, 20)
            # print(data)
            data = torch.tensor(data).cuda().float()
            # data = [*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity, reward]

            action_a = agent.get_action(current_state, data)
            action_e = act_expert

            waypoint_nearby = env.world.get_map().get_waypoint(env.location_player, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
            # print(len(env.agent._local_planner._waypoints_queue))

            if env.agent.done():  # Check whether the agent has reached its destination
                env.destination = random.choice(env.world.get_map().get_spawn_points()).location
                env.agent.set_destination(env.destination)
                # print("The target has been reached, searching for another target")

            if waypoint_nearby.is_junction:
                # print("juction!")
                action = action_a
            else:
                if waypoint_nearby.lane_id != env.agent._local_planner._waypoints_queue[0][0].lane_id:
                    action = action_e
                else:
                    action = action_a

            # 分离损失函数，以便加权损失
            loss_fn_throttle = nn.L1Loss(reduction='mean')
            loss_fn_steer = nn.L1Loss(reduction='mean')
            loss_fn_throttle = loss_fn_throttle.cuda()
            loss_fn_steer = loss_fn_steer.cuda()

            loss_throttle = loss_fn_throttle(action[:, 0], act_expert[:, 0])
            loss_steer = loss_fn_steer(action[:, 1], act_expert[:, 1])
            # loss = loss_throttle + weight_loss_steer * loss_steer # 方向盘损失权重放大100倍

            # print(action, act_expert, loss_throttle, 100 * loss_steer)

            agent.update_replay_memory((current_state, action, act_expert, done, data)) 

            # set the sectator to follow the ego vehicle
            spectator = env.world.get_spectator()
            transform = env.vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                carla.Rotation(pitch=-90)))
            
            episode_steps += 1

            if done:
                agent.update_replay_memory((current_state, action, act_expert, done, data)) 
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()#将调用join的线程优先执行，当前正在执行的线程阻塞，直到调用join方法的线程执行完毕或者被打断，主要用于线程之间的交互。
    if SAVE:
        state = {'model':agent.actor.state_dict(), 'optimizer':agent.actor_optimizer.state_dict(), 'epoch': episode}
        torch.save(state, path)

    env.world.apply_settings(Original_settings)
