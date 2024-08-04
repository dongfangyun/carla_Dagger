import random
import numpy as np
import torch
import torch.nn as nn

# data = [[throttle, steer], location, start_point, destination, forward_vector, velocity, acceleration, angular_velocity, reward]

# load_dict = np.load('./1万自动驾驶step300fps4//record_dic.npy', allow_pickle=True).item()
path = "1w_300step_4fps_attributes"
load_dict = np.load('{}/record_dic.npy'.format(path), allow_pickle=True).item()

## 查看全字典
# for key, value in load_dict.items():
#     print("{}: {}".format(key, value))

print(load_dict["10"])

minibatch = list(range(0,16))
attributes = []

for i in minibatch:

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

attributes = torch.tensor(attributes).float()
print(attributes.shape) # torch.Size([16, 20])
# print(attributes)

# b = nn.BatchNorm1d(20)
# output = b(attributes)

# print(attributes[0])
# print(output[0])

list_dic_index = range(len(load_dict))
print("字典总长度",len(list_dic_index))

partion_line = len(load_dict)//4 * 3

tarin_data_index = set(random.sample(list_dic_index, partion_line))
test_data_index = set(list_dic_index) - tarin_data_index
tarin_data_index = list(tarin_data_index)
test_data_index = list(test_data_index)

print("训练集长度",len(tarin_data_index))
print("测试集长度",len(test_data_index))


minibatch = random.sample(tarin_data_index, 16)
# print(minibatch)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

current_states = []
action = []
for i in minibatch:
    
    action.append(load_dict["{}".format(i)][0])
    kmh = load_dict["{}".format(i)][-4]
    # print(kmh)
    kmh_array = np.ones((240, 320, 1)) * kmh
    # print(kmh_array.shape)

    camera1_img = mpimg.imread('{}/camera1/{}.jpg'.format(path, i))
    camera2_img = mpimg.imread('{}/camera2/{}.jpg'.format(path, i))
    
    current_state = np.concatenate((camera1_img, camera2_img, kmh_array), axis=2) 
    current_states.append(current_state)

    
# print(len(current_states))
# plt.imshow(current_states[10])
# plt.show()

action = np.array(action)
print("动作空间形状",action.shape)

current_states = np.array(current_states)/255
current_states = torch.from_numpy(current_states)  #Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
current_states = current_states.to(torch.float32).to("cuda")
current_states = current_states.permute(0,3,1,2) # (16, c, h, w)
print("状态空间形状", current_states.shape)


# imagepath='./record//camera1//1000.jpg'
# image = mpimg.imread(imagepath)
# print(type(image)) #结果为<class 'numpy.ndarray'>
# print(image.shape) #结果为(694, 822, 3)

# plt.imshow(image)
# plt.show()

