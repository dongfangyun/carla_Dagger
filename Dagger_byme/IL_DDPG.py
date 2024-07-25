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

load_dict = np.load('./1万自动驾驶step300fps4//record_dic.npy', allow_pickle=True).item()

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
    # print(minibatch)

    current_states = []
    actions = []
    for i in minibatch:
        actions.append(load_dict["{}".format(i)][0])
        kmh = load_dict["{}".format(i)][-2]
        kmh_array = np.ones((240, 320, 1)) * kmh

        camera1_img = mpimg.imread('./1万自动驾驶step300fps4//camera1//{}.jpg'.format(i))
        camera2_img = mpimg.imread('./1万自动驾驶step300fps4//camera2//{}.jpg'.format(i))
        
        current_state = np.concatenate((camera1_img, camera2_img, kmh_array), axis=2) 
        current_states.append(current_state)

    actions = np.array(actions)
    # print(action.shape) # torch.Size([16, 2])

    current_states = np.array(current_states)/255
    current_states = torch.from_numpy(current_states)  #Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    current_states = current_states.to(torch.float32).to("cuda")
    current_states = current_states.permute(0,3,1,2) # (16, c, h, w)
    # print( current_states.shape) # torch.Size([16, 7, 240, 320])
    return current_states, actions

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
            nn.Linear(int(64 * (IM_HEIGHT/8) * (IM_WIDTH/8)), 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2), # action(batch, 2)
            nn.Tanh()
        )

    def forward(self, x):
        conv1_out = self.conv1(x) 
        # conv2_out = self.conv1(x) 
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.reshape(conv3_out.size(0), -1)
        out = self.dense(res)
        # print(out)
        return out # (batch, 2)
    
agent = Policynet(240, 320)
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

for i in range(epoch):
    agent.train() # 开启训练模式
    print("------------------第{}轮训练开始-----------------".format(i+1))
    total_train_loss = 0
    for j in range(len(tarin_data_index)//batch_size):
        current_states, actions = get_states_actions(tarin_data_index ,batch_size)
    #训练开始
        images = current_states.cuda()
        labels = torch.tensor(actions).cuda()
        labels = labels.float() # RuntimeError: Found dtype Double but expected Float,tensor double类型改为tensor float
        outputs = agent(images)

        # loss = loss_fn(outputs, labels) #(64,10) , (64)
        loss_throttle = loss_fn_throttle(outputs[:, 0], labels[:, 0])
        loss_steer = loss_fn_steer(outputs[:, 1], labels[:, 1])
        loss = loss_throttle + 100 * loss_steer # 方向盘损失权重放大10倍

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        total_train_loss += loss
        writer.add_scalar("tarin_loss_{}".format(now), loss.item(), total_train_step)
        if total_train_step % 100 == 0:
            print("第{}轮，第{}步，loss：{}".format(i+1, total_train_step, loss.item())) #不加item输出的是tensor，加了输出的是数字
    print("训练集上平均loss：{}".format(total_train_loss.item()/(len(tarin_data_index)//batch_size)))

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
            current_states, actions = get_states_actions(test_data_index, batch_size)

            images = current_states.cuda()
            labels = torch.tensor(actions).cuda()
            outputs = agent(images)

            loss_throttle = loss_fn_throttle(outputs[:, 0], labels[:, 0])
            loss_steer = loss_fn_steer(outputs[:, 1], labels[:, 1])
            loss = loss_throttle + 100 * loss_steer

            total_test_step += 1
            total_test_loss += loss
            # accuracy = (outputs.argmax(1) == labels).sum()
            # total_accuracy = total_accuracy + accuracy
            writer.add_scalar("test_loss_{}".format(now), loss.item(), total_test_step)
        print("测试集上平均loss：{}".format(total_test_loss.item()/100))
        # print("整体测试集上的准确率：{}".format(total_accuracy/test_data_size))
        
        # writer.add_scalar("test_accuracy",total_accuracy/test_data_size, total_test_step)

writer.close()