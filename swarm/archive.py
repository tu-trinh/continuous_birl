import os
import numpy as np
import random
import pybullet as p
import pybullet_data
from car import Racecar
from objects import YCBObject, InteractiveObj, RBOObject


class SimpleEnv():

    def __init__(self, n_cars = 1):
        # create simulation (GUI)
        self.urdfRootPath = pybullet_data.getDataPath()
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)

        # set up camera
        self._set_camera()

        # store car/cube data
        self.cars = []
        self.cubes = []

        # Location of goal
        self.goal = [0.0, 0.0]

        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, 0.1])

        # Load obstacles and cars
        self._load_scene(n_cars)
        
        # Some constraints
        self.dt = 1.0
        self.max_speed = 1.0
        self.min_speed = -0.5
        self.max_angle = 0.5
        self.max_acc = 1.0
        self.max_delta_yaw = 0.1
        self.resolution = 0.01
        self.car_radius = 0.6
        self.goal_cost_gain = 1.0
        self.predict_time = 0.10

    def close(self):
        p.disconnect()

    def step(self, actions):

        # get current state
        state = []
        for car in self.cars:
            state.append(self.cars[0].state)

        for i, action in enumerate(actions):
            self.cars[i].step(speed=action[0], angle=action[1])

        # take simulation step
        p.stepSimulation()

        # return next_state, reward, done, info
        next_state = []
        for car in self.cars:
            next_state = [car.state]
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info
    
    def get_cars(self):
        return self.cars

    def _load_scene(self, instances):
        goal_x = self.goal[0]
        goal_y = self.goal[1]
        h = 0.1
        car_spawn_radius = 1.5
        cube_spawn_dist = 2.5

        # Obstacle generation
        min_angle = 0
        max_angle = 30
        stage_1_cubes = []
        stage_2_cubes = []
        cubes = []
        # for cube_no in range(8):
        #     angle = np.random.randint(min_angle,max_angle) * np.pi/180.0
        #     x = goal_x + cube_spawn_dist * np.sin(angle)
        #     y = goal_y + cube_spawn_dist * np.cos(angle)
        #     cubeId = p.loadURDF(os.path.join("assets/basic/cube_static.urdf"), \
        #                         basePosition=[x, y, 0.5])
        #     if cube_no == 3:
        #         min_angle = 40
        #         max_angle = 60
        #         cube_spawn_dist = 2*cube_spawn_dist
        #     else:
        #         min_angle += 90
        #         max_angle += 90
        #     if cube_no < 4:
        #         stage_1_cubes.append([cubeId, x, y])
        #     else:
        #         stage_2_cubes.append([cubeId, x, y])
        #     self.cubes.append([cubeId, x, y])
        
        # # Get random starting locations for cars
        # starting_cubes = random.sample(stage_2_cubes, instances)

        # for cube in starting_cubes:
        #     angle = np.random.randint(min_angle, max_angle) * np.pi/180.0
        #     car_x =  cube[1] + car_spawn_radius * np.sin(angle)
        #     car_y = cube[2] + car_spawn_radius * np.cos(angle)
            # car = Racecar([car_x, car_y, h])
            # self.cars.append(car)
        # closest_points = p.getClosestPoints(car.get_car_id(),cube,100)
        # x = 5
        # y = 8
        # for i in range(n_cars):
        #     roll = 0
        #     phi = np.arctan2(y,x) - np.pi
        #     theta = 0
        #     car = Racecar([x,y,h])
        #     carId = car.get_car_id()
        #     q = p.getQuaternionFromEuler((roll, theta, phi))
        #     # print((roll, theta, phi))
        #     # p.resetBasePositionAndOrientation(carId,(x,y,h),q)
        #     self.cars.append(car)
        #     x += -8
        #     y -= 5
        #     # closest_points = p.getClosestPoints(car.get_car_id(),cube,100)
    
    def _get_cost(self, pos, car_yaw):
        rot_matrix = np.matrix([[np.cos(car_yaw), -np.sin(car_yaw)],
                                 [np.sin(car_yaw), np.cos(car_yaw)]])
        
        transformed_goal = np.linalg.inv(rot_matrix) * np.array([[-pos[0], -pos[1]]]).T
        # make sure goal and car are at the same height
        cost_to_goal = self.goal_cost_gain *  ((pos[0] - transformed_goal[0])**2 + 
                                                pos[1] - transformed_goal[1]**2)
        return cost_to_goal

    def _get_dynamic_window(self, vels):

        avg_lin_vel = np.sqrt(vels[0][0]*vels[0][0] + vels[0][1]*vels[0][1])
        avg_angular_vel = np.sqrt(vels[1][0]*vels[1][0] + vels[1][1]*vels[1][1])

        # Limits - min vel, max vel, min yaw_rate, max yaw_rate
        limit_win = [self.min_speed, self.max_speed, -self.max_angle, self.max_angle]

        # Window from current state
        state_win = [avg_lin_vel - self.max_acc * self.dt,
                        avg_lin_vel + self.max_acc * self.dt,
                        avg_angular_vel - self.max_delta_yaw * self.dt,
                        avg_angular_vel + self.max_delta_yaw * self.dt]
        # print(state_win)
        window = [max(limit_win[0], state_win[0]), min(limit_win[1], state_win[1]),
                  max(limit_win[2], state_win[2]), min(limit_win[3], state_win[3])]
        return window

    # def _predict_position(self, pos, vel, yaw, cur_yaw):
    #     # print(pos)
    #     pred_pos = [0, 0, 0]
    #     pred_pos[0] = pos[0] + vel * np.cos(yaw) * self.predict_time
    #     pred_pos[1] = pos[1] + vel * np.sin(yaw) * self.predict_time
    #     pred_pos[2] = pos[2]
    #     return pred_pos

    def get_action(self, car):
        carId = car.get_car_id()
        car_pos_orient = p.getBasePositionAndOrientation(carId)
        car_pos = car_pos_orient[0]
        car_quaternion = car_pos_orient[1]
        car_orient = p.getEulerFromQuaternion(car_quaternion)
        car_yaw = car_orient[2]
        # print(p.getEulerFromQuaternion(car_orient))
        # print(car_pos)
        car_vels = p.getBaseVelocity(carId)

        window = self._get_dynamic_window(car_vels)

        # minimum cost of position
        min_cost = float("inf")
        best_action = [0.0, 0.0]
        for vel in np.arange(window[0], window[1], self.resolution):
            for yaw in np.arange(window[2], window[3], self.resolution):

                pred_pos = self._predict_position(car_pos, vel, yaw, car_yaw)
                # transformed_goal = np.linalg.inv(rot_matrix) * \
                # np.array([[-car_pos[0], -car_pos[1]]]).T
                # Get cost of position
                cost = self._get_cost(pred_pos, car_yaw)
                if cost < min_cost:
                    min_cost = cost
                    best_action = [vel, yaw]

        # print(transformed_goal)
        # steer = np.arctan2(transformed_goal[1], transformed_goal[0])
        # print(steer)
        # while steer > np.pi:
        #     steer -= 2 * np.pi
        # while steer < -np.pi:
        #     steer += 2 * np.pi
        # best_action = [0.5, steer]
        return best_action
    def _get_obstacle_cost(self, car_yaw, new_car_pos, car_pos):
        # Get car boundaries
        new_car_boundaries = []
        new_car_boundaries.append([new_car_pos[0], new_car_pos[1] + 0.33/2])
        new_car_boundaries.append([new_car_pos[0], new_car_pos[1] - 0.33/2])
        new_car_boundaries.append([new_car_pos[0] - 0.5, new_car_pos[1] + 0.33/2])
        new_car_boundaries.append([new_car_pos[0] - 0.5, new_car_pos[1] - 0.33/2])

        obs_dists = []
        for boundary in new_car_boundaries:
            transformed_boundary = self._transform_coords(car_yaw, boundary, car_pos)
            for cube in self.cubes:
                cube_x = cube[1]
                cube_y = cube[2]
                transformed_x = transformed_boundary[0]
                transformed_y = transformed_boundary[1]
                distance_cube = (transformed_x - cube_x)**2 + (transformed_y - cube_y)**2
                # if distance_cube < 5:
                #     if distance_cube < 1.1:
                #         distance_cube = -1
                obs_dists.append(distance_cube)
        min_obs_dist = np.min(obs_dists)
        print("min_obs_dist: {}".format(min_obs_dist))
        if min_obs_dist < 0:
            obs_cost = float("inf")
            return obs_cost
        else:
            obs_cost = min_obs_dist
        return 1.0/obs_cost 
            
   
    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                     height=self.camera_height,
                                                                     viewMatrix=self.view_matrix,
                                                                     projectionMatrix=self.proj_matrix)
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=20, cameraPitch=-30,
                                     cameraTargetPosition=[0.5, -0.2, 0.0])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, 0, 0],
                                                               distance=1.0,
                                                               yaw=90,
                                                               pitch=-50,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)

