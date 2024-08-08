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

        self.total_col_num = 0

        

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

        self.world.tick() # tick一下，让车落下来，再装其他的，这样获得的信息无误。教训：路径规划初始路点位置错误，车还没下来

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

        # 初始布置的expert控制器。重置世界后自动再布置。 这里的问题在于安置车辆和安置控制器间隔太短，车辆还没落下来。加个tick试试
        self.agent = BasicAgent(self.vehicle, 30) 
        self.agent.follow_speed_limits(True)
        self.destination = random.choice(self.world.get_map().get_spawn_points()).location
        self.agent.set_destination(self.destination)
        # self.actor_list.append(self.agent) # 将控制器actor也加入销毁列表！ AttributeError: 'BasicAgent' object has no attribute 'destroy'
        self.location_player = self.location_start

    def step(self, action, episode_steps): #action.shape([1,2]):  油门刹车action[0][0]:(-1,1) 方向盘action[0][1]:(-1,1)

        # 车辆当前位置（x,y,z）和方向(三维矢量)
        transform_player = self.vehicle.get_transform()
        # print(transform_player)
        self.location_player =transform_player.location
        forward_vector_player =transform_player.get_forward_vector() 

        location = (self.location_player.x,self.location_player.y, self.location_player.z)
        destination = (self.destination.x, self.destination.y , self.destination.z)
        forward_vector = (forward_vector_player.x, forward_vector_player.y, forward_vector_player.z)

        # 车辆初始点位置及距初始点距离
        location_start = self.location_start
        dist_to_start = self.location_player.distance(location_start)
        start_point = (location_start.x, location_start.y, location_start.z)

        # 车辆当前速度
        velocity_player = self.vehicle.get_velocity() # Return: carla.Vector3D - m/s
        velocity = int(3.6 * math.sqrt(velocity_player.x**2 + velocity_player.y**2 + velocity_player.z**2)) # --> km/h

        # 车辆当前加速度
        acceleration_player = self.vehicle.get_acceleration()
        acceleration = (acceleration_player.x, acceleration_player.y, acceleration_player.z)

        # 车辆当前角速度
        angular_velocity_player = self.vehicle.get_angular_velocity()
        angular_velocity = (angular_velocity_player.x, angular_velocity_player.y, angular_velocity_player.z)


        # 到达目的地则自动驾驶控制器自动换个目标点，controller中有传出的初始目的地，默认未done
        if self.agent.done():  # Check whether the agent has reached its destination
            self.destination = random.choice(self.world.get_map().get_spawn_points()).location
            self.agent.set_destination(self.destination)
            print("The target has been reached, searching for another target")

        # 设置智能专家，会修正导航
        # 矫正专家，如果最近路点与局部路线规划下一个路点lane_id不符，重新规划路线, 目标点不变（不重新取点）。 得排除交叉路口，不然太善变
        waypoint_nearby = self.world.get_map().get_waypoint(self.location_player, project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
        if waypoint_nearby.is_junction:
            # print("juction!")
            pass
        else:
            if waypoint_nearby.lane_id != self.agent._local_planner._waypoints_queue[0][0].lane_id:
                self.agent.set_destination(self.destination, start_location=True)
                # print("计划有变")

        
        # 根据当前这一步执行专家指示
        self.act_expert = self.agent.run_step() # 专家指导动作 ，这里的指导动作已然有大病 （已修正）
        self.act_expert.manual_gear_shift = False
        self.act_expert = torch.tensor([[self.act_expert.throttle - self.act_expert.brake, self.act_expert.steer]]).cuda()

        throttle = float(torch.clip(action[0][0], 0, 1))
        brake = float(torch.abs(torch.clip(action[0][0], -1, 0)))
        
        steer = float(torch.clip(action[0][1], -1, 1))

        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, reverse=False))


        # 检测是否跨越车道线
        if len(self.invasion_hist) != 0:
            self.invation = True
            self.invasion_hist.clear()
        else:
            self.invation = False

        # 发生碰撞时游戏结束
        if len(self.collision_hist) != 0:
            self.done = True
            self.total_col_num += 1
            # print("1",self.done)

        # 每次游戏在现实中超过SECONDS_PER_EPISODE时常后结束
        # print("episode_steps", episode_steps,"PER_EPISODE_max_steps", PER_EPISODE_max_steps)
        if episode_steps > PER_EPISODE_max_steps:
            self.done = True

        data = [*location, *start_point, *destination, *forward_vector, velocity, *acceleration, *angular_velocity]

        # print(self.act_expert)
        return  self.done, data, self.act_expert, dist_to_start, self.invation
    
camera_queue1 = Queue()
camera_queue2 = Queue()

PER_EPISODE_max_steps = 300 # 单次游戏最大次数（秒）