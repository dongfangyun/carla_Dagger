import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os

load_dict = np.load('./1万全属性step300fps4//record_dic.npy', allow_pickle=True).item()

partion_line = len(load_dict)//4 * 3

list_dic_index = range(len(load_dict))

tarin_data_index = set(random.sample(list_dic_index, partion_line))
test_data_index = set(list_dic_index) - tarin_data_index
tarin_data_index = list(tarin_data_index)
test_data_index = list(test_data_index)

now = time.ctime(time.time())
now = now.replace(" ","_").replace(":", "_")

def get_states_actions(data_index ,batch_size): # 此函数current_state是对的，采集的数据中第四个alpha通道已剔除
    # 采样
    minibatch = random.sample(data_index, batch_size)

    imgs = []
    actions = []
    attributes = []

    for i in minibatch:
        camera1_img = mpimg.imread('./1万全属性step300fps4//camera1//{}.jpg'.format(i))
        camera2_img = mpimg.imread('./1万全属性step300fps4//camera2//{}.jpg'.format(i))
        img = np.concatenate((camera1_img, camera2_img), axis=2) # 将影像叠置拼接
        imgs.append(img)

        location = load_dict["{}".format(i)][1] # (3)
        start_point = load_dict["{}".format(i)][2] # (3)
        destination = load_dict["{}".format(i)][3] # (3)
        forward_vector = load_dict["{}".format(i)][4] # (3)
        velocity = load_dict["{}".format(i)][5] # (1)
        acceleration = load_dict["{}".format(i)][6] # (3)
        angular_velocity = load_dict["{}".format(i)][7] # (3)
        reward = load_dict["{}".format(i)][8] # (1)

        attribute = [*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity, reward]
        attributes.append(attribute) # len(20)

        action = load_dict["{}".format(i)][0]
        actions.append(action)

    imgs = np.array(imgs)/255
    imgs = torch.from_numpy(imgs) 
    imgs = imgs.to(torch.float32).to("cuda")
    imgs = imgs.permute(0,3,1,2) # (b, c, h, w) --> torch.Size([batch_size, 6, 240, 320])

    attributes = np.array(attributes) # torch.Size([batch_size,20])

    actions = np.array(actions)

    return imgs, attributes, actions

# data = [[throttle, steer], location, start_point, destination, forward_vector, velocity, acceleration, angular_velocity, reward]

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
            nn.Dropout(0.5),
        )
        self.conv_fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # attributes全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        
        # 拼接后的全连接层 32 + 32 = 64 --> 32 -->16 -->2
        self.cat_fc1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cat_fc2 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cat_fc3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cat_fc4 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cat_fc5 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cat_fc6 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
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
    

# class Policynet(nn.Module): # base_net
#     def __init__(self, IM_HEIGHT, IM_WIDTH):
#         super(Policynet, self).__init__() 
#         # images的卷积层+全连接层
#         self.conv1 = nn.Sequential(
#             # nn.BatchNorm2d(7),
#             nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.conv2 = nn.Sequential(
#             # nn.BatchNorm2d(32),
#             nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.conv3 = nn.Sequential(
#             # nn.BatchNorm2d(32),
#             nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             # nn.BatchNorm2d(64),
#         )
#         self.conv_fc1 = nn.Sequential(
#             nn.Linear(int(64 * (IM_HEIGHT/8) * (IM_WIDTH/8)), 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )
#         self.conv_fc2 = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )

#         # attributes全连接层
#         self.fc1 = nn.Sequential(
#             nn.Linear(20, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )
#         self.fc3 = nn.Sequential(
#             nn.Linear(32, 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )
        
#         # 拼接后的全连接层 32 + 32 = 64 --> 32 -->16 -->2
#         self.cat_fc1 = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#         )
#         self.cat_fc2 = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(16, 2), 
#             nn.Tanh()
#         )

#     def forward(self, images, attributes):
#         conv1_out = self.conv1(images) 
#         conv2_out = self.conv2(conv1_out)
#         conv3_out = self.conv3(conv2_out)
#         conv3_res = conv3_out.reshape(conv3_out.size(0), -1) # --> (76800)
#         conv_fc1_out = self.conv_fc1(conv3_res) # 76800 --> (64)
#         conv_fc2_out = self.conv_fc2(conv_fc1_out) # (32)
#         # print("conv_fc2_out", conv_fc2_out.shape) #torch.Size([64, 32])

#         fc1_out = self.fc1(attributes) # 20 --> 64
#         fc2_out = self.fc2(fc1_out) # 64 --> 32
#         fc3_out = self.fc3(fc2_out) # 32 --> 32
#         # print("fc3_out", fc3_out.shape) # torch.Size([64, 32])

#         cat = torch.cat(( conv_fc2_out, fc3_out), 1) # 32 + 32 = 64 --> 32
#         # print("cat", cat.shape) # torch.Size([64, 64])
#         cat_fc1_out = self.cat_fc1(cat) # 32 --> 16
#         cat_fc2_out = self.cat_fc2(cat_fc1_out) # 16 --> 2 

#         return cat_fc2_out # (batch, 2)
    
agent = Policynet_cat_fc_pro(240, 320)
agent = agent.cuda()

# 分离损失函数，以便加权损失
loss_fn_throttle = nn.L1Loss(reduction='mean')
loss_fn_steer = nn.L1Loss(reduction='mean')
loss_fn_throttle = loss_fn_throttle.cuda()
loss_fn_steer = loss_fn_steer.cuda()

optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
writer = SummaryWriter("./logs_IL_traning") 

total_train_step = 0
total_test_step = 0
epoch = 100
batch_size = 64

weight_loss_steer = 100

for i in range(epoch):
    agent.train() # 开启训练模式
    print("------------------第{}轮训练开始-----------------".format(i+1))

    epoch_add_loss = 0
    epoch_step = 0

    for j in range(len(tarin_data_index)//batch_size): # 每轮把所有数据的量至少抽样过一遍
        imgs, attributes, actions = get_states_actions(tarin_data_index ,batch_size)
        imgs = imgs.cuda()
        attributes = torch.tensor(attributes).cuda().float()
        actions = torch.tensor(actions).cuda().float()

        outputs = agent(imgs, attributes)

    # #训练开始
    #     images = current_states.cuda()
    #     labels = torch.tensor(actions).cuda()
    #     labels = labels.float() # RuntimeError: Found dtype Double but expected Float,tensor double类型改为tensor float
    #     outputs = agent(images)

        # loss = loss_fn(outputs, labels) #(64,10) , (64)
        loss_throttle = loss_fn_throttle(outputs[:, 0], actions[:, 0])
        loss_steer = loss_fn_steer(outputs[:, 1], actions[:, 1])
        loss = loss_throttle + weight_loss_steer * loss_steer # 方向盘损失权重放大100倍

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        epoch_add_loss += loss
        epoch_step += 1

        writer.add_scalar("tarin_loss_{}".format(now), loss.item(), total_train_step)
        if total_train_step % 100 == 0:
            print("第{}轮，第{}步，loss：{}".format(i+1, total_train_step, loss.item())) #不加item输出的是tensor，加了输出的是数字
    print("训练集上平均loss：{}".format(epoch_add_loss.item()/epoch_step))

    if not os.path.isdir('./IL_experience_model'):
        os.makedirs('./IL_experience_model')
    path='./IL_experience_model/model_{}.pth'.format(now)

    state = {'model':agent.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i}
    torch.save(state, path)

    #测试开始
    agent.eval() # 开启测试模式
    total_test_loss = 0
    with torch.no_grad():
        for j in range(100):
            imgs, attributes, actions = get_states_actions(test_data_index ,batch_size)
            imgs = imgs.cuda()
            attributes = torch.tensor(attributes).cuda().float()
            actions = torch.tensor(actions).cuda().float()
            outputs = agent(imgs, attributes)

            loss_throttle = loss_fn_throttle(outputs[:, 0], actions[:, 0])
            loss_steer = loss_fn_steer(outputs[:, 1], actions[:, 1])
            loss = loss_throttle + weight_loss_steer * loss_steer

            total_test_step += 1
            total_test_loss += loss
            # accuracy = (outputs.argmax(1) == labels).sum()
            # total_accuracy = total_accuracy + accuracy
            writer.add_scalar("test_loss_{}".format(now), loss.item(), total_test_step)

        print("测试集上平均loss：{}".format(total_test_loss.item()/100))
        # print("整体测试集上的准确率：{}".format(total_accuracy/test_data_size))
        
        # writer.add_scalar("test_accuracy",total_accuracy/test_data_size, total_test_step)

writer.close()