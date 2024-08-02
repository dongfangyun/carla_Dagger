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

from continous_hd_CarEnv import CarEnv, IM_WIDTH, IM_HEIGHT,camera_queue1, camera_queue2 


SHOW_PREVIEW = False # 训练时播放摄像镜头

LOG = True # 训练时向tensorboard中写入记录
writer = SummaryWriter("./logs_hd_DDPG")

SAVE = True # 保存模型
if SAVE:
    if not os.path.isdir('models_hd_DDPG'):
            os.makedirs('models_hd_DDPG')

TRAINED_MODEL = False # 是否有预训练模型
trained_model_dir = r"旧数据集\一万三随机模型\model.pth"

REPLAY_MEMORY_SIZE = 1000 # 经验回放池最大容量
MIN_REPLAY_MEMORY_SIZE = 500# 抽样训练开始时经验回放池的最小样本数
MINIBATCH_SIZE = 8 # 每次从经验回放池的采样数（作为训练的同一批次）   此大小影响运算速度/显存
UPDATE_TARGET_EVERY = 5 # 同步target网络的训练次数
EPISODES = 1000 # 游戏进行总次数
DISCOUNT = 0.99 # 贝尔曼公式中折扣因子γ


"""
    Policynet():
    input: state([batch, 7,  height, width])
    return: action([batch, 2])
"""
class Policynet(nn.Module):
    def __init__(self, IM_HEIGHT, IM_WIDTH):
        super(Policynet, self).__init__() 
        self.conv1 = nn.Sequential(
            # nn.BatchNorm2d(7),
            nn.Conv2d(7, 32, kernel_size=5, stride=1, padding=2),
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
        self.dense = nn.Sequential(
            nn.Linear(int(64 * (IM_HEIGHT/4) * (IM_WIDTH/4)), 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2), # action(batch, 2)
            nn.Tanh()
        )

    def forward(self, x):
        # conv1_out = self.conv1(x) 
        conv2_out = self.conv1(x) 
        # conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.reshape(conv3_out.size(0), -1)
        out = self.dense(res)
        # print(out)
        return out # (batch, 2)

"""
    QValueNet():
    input: state([batch, 7,  height, width]), action([batch, 2])
    return: QValue(1)
"""
class QValueNet(nn.Module):
    def __init__(self, IM_HEIGHT, IM_WIDTH):
        super(QValueNet, self).__init__() 
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.dense = nn.Sequential(
            nn.Linear(int(64 * (IM_HEIGHT/8) * (IM_WIDTH/8)), 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1), # action(batch, 2)
        )


    def forward(self, x, action): #拼接state和action，其中action([batch, 2])，需要拼接为两层
        tensor_list = []
        for i in range(MINIBATCH_SIZE): #x: (b,3,h,w)->(b,5,h,w)
            a0 = action[i][0] #油刹
            a1 = action[i][1] #方向
            a0_array = torch.ones((1, IM_HEIGHT, IM_WIDTH)).to(device)*a0 # （ 1, h, w）
            a1_array = torch.ones((1, IM_HEIGHT, IM_WIDTH)).to(device)*a1 # （ 1, h, w）
            tensor_list.append(torch.cat([x[i], a0_array, a1_array], dim=0)) #[（ 9, h, w）]
        x = torch.stack(tensor_list) # x.shape torch.Size([2, 9, 240, 320]) batch==2时
        
        conv1_out = self.conv1(x) 
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.reshape(conv3_out.size(0), -1)
        out = self.dense(res)
        # print(out)
        return out # (batch, 2)

if torch.cuda.device_count() > 1:
    device = torch.device("cuda:1") 
else:
    device = torch.device("cuda:0")

class DDPG:
    def __init__(self,lr_actor=1e-2, lr_critic=1e-2):

        self.actor = Policynet(IM_HEIGHT, IM_WIDTH).to("cuda:0")
        self.critic = QValueNet(IM_HEIGHT, IM_WIDTH).to("cuda:0")
        self.target_actor = Policynet(IM_HEIGHT, IM_WIDTH).to("cuda:0")
        self.target_critic = QValueNet(IM_HEIGHT, IM_WIDTH).to("cuda:0")
        self.clone_critic = QValueNet(IM_HEIGHT, IM_WIDTH).to("cuda:0")

        if torch.cuda.device_count() > 1:
            self.actor= nn.DataParallel(self.actor, device_ids = [0,1], output_device=device)
            self.critic= nn.DataParallel(self.critic, device_ids = [0,1], output_device=device)
            self.target_actor = nn.DataParallel(self.target_actor, device_ids = [0,1], output_device=device )
            self.target_critic= nn.DataParallel(self.target_critic, device_ids = [0,1], output_device=device)
            self.clone_critic= nn.DataParallel(self.target_critic, device_ids = [0,1], output_device=device)

        #初始化同步目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.clone_critic.load_state_dict(self.critic.state_dict())

        #只更新主网络的参数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor) 
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic) 

        #启动预训练模型
        if TRAINED_MODEL:
            if os.path.exists(trained_model_dir):
                checkpoint = torch.load(trained_model_dir)
                self.actor.load_state_dict(checkpoint['model'])
                self.actor_optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['epoch']
                print('加载 epoch {} 成功！'.format(start_epoch))   

        self.sigma = 0.1 #高斯噪声标准差
        self.tau = 0.005 #软更新参数
        self.gamma = 0.99 #折扣因子

        
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE) #上限5000的经验回放池队列
        self.terminate = False #没结束循环训练，当全部游戏次数跑完后此处会改为True，停止进程中的采样训练
        self.training_initialized = False

    def get_action(self, state): # state(16,c,h,w)
        self.actor.eval() #改完合理，但还是不行
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float).to(device) # (hwc)
            state = state/255 
            state = state.permute(2, 0, 1) # (chw)
            state = state.unsqueeze(dim=0) # (n, h, w, c) 

            action = self.actor(state)# action(batch, 2)
            
            #给动作添加噪声，增加探索
            action = action.clone() + torch.tensor(self.sigma * np.random.randn(2)).to(device) #self.action_dim = 2
        self.actor.train() #########此处为何要train？？？？？？？？
        return action # action(batch, 2)

    #将replay_memory、update_replay_memory()均现在class DQNAgent内部，无需从类外再传入经验回放池队列变量
    def update_replay_memory(self, transition): 
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(self.tau * param.data + (1.0 - self.tau) * param_target.data)

    def train(self):
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE) #每次采样16个轨迹作为一批次同时训练

        # 16个随机样本的current_states，16x(h, w, c) --> (16, c, h, w)
        current_states = np.array([transition[0] for transition in minibatch])/255 #(16, h, w, c)
        current_states = torch.from_numpy(current_states)  #Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        current_states = current_states.to(torch.float32).to(device)
        current_states = current_states.permute(0,3,1,2) # (16, c, h, w)

        # 16个随机样本的new_current_states
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        new_current_states = torch.from_numpy(new_current_states) 
        new_current_states = new_current_states.to(torch.float32).to(device)
        new_current_states = new_current_states.permute(0,3,1,2) # (16, c, h, w)

        rewards = torch.tensor([transition[2] for transition in minibatch]).to(torch.float32).to(device)# (batch)
        rewards = rewards.reshape(-1, 1) #(batch)-->(batch,1)
        actions = torch.stack([transition[1][0] for transition in minibatch]).to(torch.float32).to(device) #包含tensor(1, 2)的列表-->tensor(batch,2)
        # print("actions.shape",actions.shape) actions.shape torch.Size([2, 2]) batch==2时

        # current_qs_list = self.critic(current_states, actions)  #把这16个state运算一次  Qw(s)=>(batch, 1)
        with torch.no_grad():
            future_qs_list = self.target_critic(new_current_states, self.target_actor(new_current_states)) #  Qw-(s', Π-（a'）)=>(batch, 1) target全用target网络计算，再用真网络计算的Q去拟合。
            target_qs_list = rewards + self.gamma * future_qs_list

        torch.autograd.set_detect_anomaly(True) #AddmmBackward0 函数时发生了问题。AddmmBackward0 是PyTorch中执行矩阵和矩阵相乘（mm）和矩阵和矩阵与标量相乘（addmm）的函数的一部分，用于计算梯度。

        critic_loss = torch.mean(nn.functional.mse_loss(self.critic(current_states, actions), target_qs_list)).to(device) #,(torch.Size([2, 1])  (torch.Size([2, 2])

        actor_value = self.clone_critic(current_states, self.actor(current_states)) 
        # actor_value = self.target_critic(current_states, self.actor(current_states)) #改成target_critic可行，但不严谨
        actor_loss = -torch.mean(actor_value).to(device) #Q(s, Π（s）)上升

        self.critic_optimizer.zero_grad() # 梯度清零
        critic_loss.backward() # 产生梯度 同时对critic和actor产生梯度！其实不影响。只更actor前会清零.不应该改变actor梯度！！！！
        self.critic_optimizer.step() # 根据梯度更新

        self.clone_critic.zero_grad() # 梯度清零
        self.actor_optimizer.zero_grad() # 梯度清零
        actor_loss.backward() # 产生梯度 同时对critic和actor产生梯度！其实不影响。只更critic前会清零
        self.actor_optimizer.step() # 使用target_critic做loss后，actor和critic更新的先后顺序也无所谓了

        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.actor, self.target_actor)
        self.clone_critic.load_state_dict(self.critic.state_dict())

        return critic_loss.item(), actor_loss.item()
    
    def train_in_loop(self):
        total_train_step = 0
        while True:
            if self.terminate: #当100批样本收集跑完后此处改为true，只要没跑够100批，此处会一直支线程进行训练
                return
            
            if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
                print(len(self.replay_memory))
                # return

            if len(self.replay_memory) > MIN_REPLAY_MEMORY_SIZE: # 当经验回放池样本数量大于MIN_REPLAY_MEMORY_SIZE才会开始采样训练
                # print("start train")
                critic_loss, actor_loss = self.train()
                print("loss:", critic_loss, actor_loss)
                total_train_step += 1

                if LOG:
                    writer.add_scalar("critic_loss_{}".format(now), critic_loss, total_train_step)
                    writer.add_scalar("actor_loss_{}".format(now), actor_loss, total_train_step)


# def caculate_reward(done, new_kmh, new_dis, kmh, dis, action, env):
#     reward = 0.0

#     reward += np.clip(new_dis-dis, -10.0, 10.0) 
#     # reward += (new_dis-dis)*1
#     # reward +=(new_kmh - kmh)
#     reward +=(kmh) * 0.05
#     if len(env.collision_hist) != 0:
#         reward += -10
#     if kmh < 1:
#         reward += -1
#     if action[0][0] > 0:
#         reward += 1

#     return reward

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
    # For more repetitive results
    # random.seed(1)
    # np.random.seed(1)

    # Create agent and environment
    env = CarEnv()    
    agent = DDPG()

    Original_settings = env.original_settings # 将原设置传出来保存
    # sensor_queue1 = Queue()
    
    now = time.ctime(time.time())
    now = now.replace(" ","_").replace(":", "_")

    # Start training thread and wait for training to be initialized
    trainer_thread = Thread(target=agent.train_in_loop, daemon=True) #创建一个线程，调用的函数是train_in_loop
    trainer_thread.start() #此处会直接往下走，同时线程分支（train_in_loop，训练一批）开始运行
    #线程start后能跑多少是没谱的，可能没咋跑也可能循环了10000次。
    #同步模式主要解决的是listen传回来的图片不连续，是客户端与服务器间的问题。影响的是加入经验回放池的样本轨迹
    #DQN的训练每次都是独立线程进行，用的是缓冲区内的样本，与同异步无关

    episode_num = 0 # 游戏进行的次数

    all_average_reward = 0

    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'): # 1~100 EPISODE
        env.collision_hist = [] # 记录碰撞发生的列表
        episode_reward = 0 # 每次游戏所有step累计奖励

        # Reset environment and get initial state
        env.reset() #此处reset里会先tick()一下，往队列里传入初始图像

        current_state1 = camera_queue1.get() # <class 'carla.libcarla.Image'> Image(frame=154824, timestamp=1589.824231, size=640x480)
        i_1 = np.array(current_state1.raw_data) #(1228800,) = 640 X 480 X 4   	.raw_data：Array of BGRA 32-bit pixels
        i2_1 = i_1.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3_1 = i2_1[:, :, :3] # （h, w, 3）= (480, 640, 3)

        current_state2 = camera_queue2.get() # <class 'carla.libcarla.Image'> Image(frame=154824, timestamp=1589.824231, size=640x480)
        current_state2.convert(carla.ColorConverter.CityScapesPalette)
        i_2 = np.array(current_state2.raw_data) #(1228800,) = 640 X 480 X 4
        i2_2 = i_2.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3_2 = i2_2[:, :, :3] # （h, w, 3）= (480, 640, 3)  # （h, w, 3）= (480, 640, 3)

        action = torch.tensor([[0, 0]]).to(device)  # torch.Size([1, 2])
        reward =0
        done = False
        kmh = 0
        kmh_array = np.ones((IM_HEIGHT, IM_WIDTH, 1))*kmh # （h, w, 1）= (480, 640, 1)
        dis_to_start_old = 0
        current_state = np.concatenate((i3_1, i3_2, kmh_array), axis=2) # （h, w, 3）= (480, 640, 3+3+1 = 7)
        #以上为初始帧的s,a,r,done,kmh,dis

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

            new_state = np.concatenate((i3_1, i3_2, kmh_array), axis=2) # （h, w, 3+3+1）= (480, 640, 7)

            agent.update_replay_memory((current_state, action, reward, new_state, done))

            current_state = new_state.copy() #array直接复制会浅拷贝共用内存，此处需深拷贝保持二者独立性 (480, 640, 3)

            # if len(agent.replay_memory) >= REPLAY_MEMORY_SIZE:
            #     action = agent.get_action(current_state)
            # else:
            #     action = torch.tensor([[torch.rand(1)*2 - 1, torch.rand(1)*2 - 1]]).to(device)
            #     print(len(agent.replay_memory))

            action = agent.get_action(current_state)

            # reward, done, _ = env.step(action)
            done, new_kmh, dis_to_start, inva_lane= env.step(action, episode_steps)
            # reward = caculate_reward(done, new_kmh, new_dis, kmh, dis, action, env)
            reward = caculate_reward(dis_to_start, dis_to_start_old, kmh, done, inva_lane, action)
            kmh = new_kmh
            dis_to_start_old = dis_to_start

            print(action, reward)

            episode_reward += reward

            # set the sectator to follow the ego vehicle
            spectator = env.world.get_spectator()
            transform = env.vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                carla.Rotation(pitch=-90)))
            
            episode_steps += 1

            if done:
                agent.update_replay_memory((current_state, action, reward, new_state, done)) #此处current_state == new_state，new_state不参与Q值拟合训练运算
                break

        # End of episode - destroy agents
        for actor in env.actor_list:
            actor.destroy()

        episode_average_reward = episode_reward/episode_steps

        # 记录每一次游戏的平均奖励
        if LOG:
            writer.add_scalar("average_reward_{}".format(now), episode_average_reward, episode_num)

        all_average_reward += episode_average_reward

        # 每100个episode保存一次模型，输出一下近100次的平均奖励
        if episode_num%100 == 0:
            if SAVE:
                    # torch.save(agent.critic, "./models_hd_DDPG/qnet_{}.pth".format(now))
                    torch.save(agent.actor, "./models_hd_DDPG/qnet_{}.pth".format(now))
            print("all_average_reward", all_average_reward/100)
            all_average_reward = 0

    # Set termination flag for training thread and wait for it to finish
    agent.terminate = True
    trainer_thread.join()#将调用join的线程优先执行，当前正在执行的线程阻塞，直到调用join方法的线程执行完毕或者被打断，主要用于线程之间的交互。

    env.world.apply_settings(Original_settings)
