"""DQN Synchronous Train Torch version"""
import os
import glob # 用于添加carla.egg。环境中装有.whl可忽略
import sys
import random
import time
import math
from queue import Queue # 队列
import numpy as np
import torch

from agents.navigation.basic_agent import BasicAgent 

IM_HEIGHT = 240 # 前置摄像头图像高
IM_WIDTH = 320 # 前置摄像头图像宽

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

class CarEnv:
    im_width = IM_WIDTH
    im_height = IM_HEIGHT

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0] # 主角汽车模型

        # set synchorinized mode
        self.original_settings = self.world.get_settings() # 保存世界异步原设置，训练结束关闭前需还原设置。初始化后得将此参数传出去，不然调用的时候又会从头self.client.get_world().get_settings(), 得到的是当前设置而非原设置
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.25 # 1/0.5 =2，fixed_delta_seconds = 0.5,2次运算在仿真世界内为一秒。每隔多少仿真时间进行一次运算。

        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        

    def reset(self):
        self.collision_hist = []
        self.invasion_hist = []
        self.actor_list = []

        # 每次重置先关交通灯
        self.actors=self.world.get_actors()
        self.light_actor_list=self.actors.filter('*traffic_light*')
        for light_actor in self.light_actor_list:
                light_actor.set_state(carla.TrafficLightState.Green)
                light_actor.freeze(True)

        self.done = False

        self.vehicle = None
        while self.vehicle is None:
            self.transform = random.choice(self.world.get_map().get_spawn_points()) # 尝试生产，位置冲突等失败，就返回None，再选
            self.vehicle = self.world.try_spawn_actor(self.model_3, self.transform)
        self.location_start = self.transform.location
        self.actor_list.append(self.vehicle)

        #设置sensor摄像头，位置绑定actor车辆. world.tick()后将listen到的数据放入sensor_queue_out队列.
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}") # 640
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}") # 480
        self.rgb_cam.set_attribute("fov", f"110")
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(camera_queue1.put)

        #设置sensor摄像头2，位置绑定actor车辆. world.tick()后将listen到的数据放入sensor_queue_out队列.
        self.rgb_cam2 = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.rgb_cam2.set_attribute("image_size_x", f"{self.im_width}") # 640
        self.rgb_cam2.set_attribute("image_size_y", f"{self.im_height}") # 480
        self.rgb_cam2.set_attribute("fov", f"110")
        transform2 = carla.Transform(carla.Location(z=20), carla.Rotation(pitch=-90))
        self.sensor2 = self.world.spawn_actor(self.rgb_cam2, transform2, attach_to=self.vehicle)
        self.actor_list.append(self.sensor2)
        self.sensor2.listen(camera_queue2.put)

        #设置colsensor碰撞雷达，位置绑定actor车辆. world.tick()后将listen到的碰撞信息放入self.collision_hist列表.
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_hist.append(event))

        #设置lane_invasion，位置绑定actor车辆. 
        invsensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.invsensor = self.world.spawn_actor(invsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.invsensor)
        self.invsensor.listen(lambda event: self.invasion_hist.append(event))

        # self.world.tick() #先tick一下将初始摄像头影像传入sensor_queue_out 先取消掉试试，理论上可行

        # 初始布置的expert控制器。重置世界后自动再布置
        self.agent = BasicAgent(self.vehicle, 30) 
        self.agent.follow_speed_limits(True)
        destination = random.choice(self.world.get_map().get_spawn_points()).location
        self.agent.set_destination(destination)

    def step(self, action, episode_steps): #action.shape([1,2]):  油门刹车action[0][0]:(-1,1) 方向盘action[0][1]:(-1,1)
        throttle = float(torch.clip(action[0][0], 0, 1))
        brake = float(torch.abs(torch.clip(action[0][0], -1, 0)))
        
        steer = float(torch.clip(action[0][1], -1, 1))

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=False))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # 发生碰撞时游戏结束
        if len(self.collision_hist) != 0:
            self.done = True
            # print("1",self.done)

        # 每次游戏在现实中超过SECONDS_PER_EPISODE时常后结束
        # print("episode_steps", episode_steps,"PER_EPISODE_max_steps", PER_EPISODE_max_steps)
        if episode_steps > PER_EPISODE_max_steps:
            self.done = True

        # 根据当前这一步执行专家指示
        self.act_expert = self.agent.run_step() # 专家指导动作   
        self.act_expert.manual_gear_shift = False

        # print(self.done)
        return  self.done, kmh, self.act_expert
    
camera_queue1 = Queue()
camera_queue2 = Queue()

PER_EPISODE_max_steps = 300 # 单次游戏最大次数（秒）