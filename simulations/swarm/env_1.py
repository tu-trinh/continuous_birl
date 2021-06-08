import os
import numpy as np
import random
import pybullet as p
import pybullet_data
from car import Racecar
from objects import YCBObject, InteractiveObj, RBOObject
import sys

class SimpleEnv():

    def __init__(self, n_cars = 1, theta = "regular"):
        # create simulation (GUI)
        self.urdfRootPath = pybullet_data.getDataPath()
        p.connect(p.GUI)
        # p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81)

        # set up camera
        self._set_camera()

        # store car/cube data
        self.cars = []
        self.obstacles = []
        self.prev_pos = []

        # Location of goal
        self.goal = [0.0, 0.0]

        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, 0.1])

        # Load obstacles and cars
        self._load_scene(n_cars)
        
        # Some constraints
        self.dt = 0.5

        self.goal_cost_gain = 0
        self.obstacle_cost_gain = 0
        
        if theta == "regular":
            self.goal_cost_gain = 1
            self.obstacle_cost_gain = 1
        elif theta == "goal":
            self.goal_cost_gain = 0.0001
            self.obstacle_cost_gain = 0
        elif theta == "obstacle":
            self.goal_cost_gain = 0.0001
            self.obstacle_cost_gain = 1

    def close(self):
        p.disconnect()

    def step(self, actions):

        # get current state
        # state = []
        # for car in self.cars:
        #     state.append(self.cars[0].state)

        for i, action in enumerate(actions):
            self.cars[i].step(speed=action[0], angle=action[1])

        # take simulation step
        p.stepSimulation()

        next_state = []
        done = True
        goal_status = []
        obs_dists = []
        for car in self.cars:
            state = car.state
            pos = state['position']
            next_state.append([pos])
            goal_state = self.goal_reached(car)
            goal_status.append(goal_state)
            done = done and goal_state
        reward = 0.0
        info = {'state':next_state}

        return next_state, reward, done, info

    def get_cars(self):
        return self.cars

    def goal_reached(self, car):
        carId = car.get_car_id()
        car_pos_orient = p.getBasePositionAndOrientation(carId)
        car_pos = car_pos_orient[0]
        dist = np.sqrt((car_pos[0] - self.goal[0])**2 + (car_pos[1] - self.goal[1])**2)
        if  dist < 1.0:
            return True
        else:
            return False

    def get_obs_dist(self, pos, carId):
        min_dist = float("inf")
        for i,obstacle in enumerate(self.obstacles):
            obstacleId = obstacle[0]
            if not (obstacleId == carId):
                cube_x = obstacle[1]
                cube_y = obstacle[2]
                distance_cube = (pos[0] - cube_x)**2 + (pos[1] - cube_y)**2
                if distance_cube < min_dist:
                    idx = i
                    min_dist = distance_cube
        return idx, min_dist

    def get_min_dists(self):
        min_dists = []
        for car in self.cars:
            carId = car.get_car_id()
            car_pos_orient = p.getBasePositionAndOrientation(carId)
            car_pos = car_pos_orient[0]
            _,closest_dist = self.get_obs_dist(car_pos, carId)
            min_dists.append(closest_dist)
        return min_dists

    def get_closest_car(self):
        min_dist = float("inf")
        for i,car in enumerate(self.cars):
            if not self.goal_reached(car):
                carId = car.get_car_id()
                car_pos_orient = p.getBasePositionAndOrientation(carId)
                car_pos = car_pos_orient[0]
                _,closest_dist = self.get_obs_dist(car_pos, carId)
                if closest_dist < min_dist:
                    closest_car = car
                    closest_ind = i
                    min_dist = closest_dist
            # print("car: {}\t dist: {}"\
            #     .format(closest_car.get_car_id(), closest_dist))
        return closest_ind, closest_car

    def get_action(self, index):
        carId = self.cars[index].get_car_id()
        car_pos_orient = p.getBasePositionAndOrientation(carId)
        car_pos = car_pos_orient[0]
        car_quaternion = car_pos_orient[1]
        car_orient = p.getEulerFromQuaternion(car_quaternion)
        car_yaw = car_orient[2]
        # minimum cost of position
        min_cost = float("inf")
        best_vel = 0.0
        best_steer = 0.0

        for vel in [-1.5, 2.0]:
            for steer in np.arange(-0.5,0.5,0.1):
                new_car_x = vel * np.cos(steer) * self.dt
                new_car_y = vel * np.sin(steer) * self.dt
                new_car_pos = [new_car_x, new_car_y]
                transformed_pos = self._transform_coords(car_yaw,\
                                         [new_car_x, new_car_y], car_pos)
                distance_goal = self._get_goal_cost(transformed_pos)
                obs_cost = self._get_obstacle_cost(car_yaw, new_car_pos, car_pos, carId)
                cost = self.goal_cost_gain * distance_goal +\
                         self.obstacle_cost_gain * obs_cost

                if cost < min_cost:
                    best_pos = new_car_pos
                    min_cost = cost
                    best_steer = steer
                    best_vel = vel

        best_action = [best_vel, best_steer]
        return best_action
   
    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(width=self.camera_width,
                                                                     height=self.camera_height,
                                                                     viewMatrix=self.view_matrix,
                                                                     projectionMatrix=self.proj_matrix)
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _transform_coords(self, angle, pos, ref_pos):
        rot_matrix = np.matrix([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
        transformed_pos = rot_matrix * np.array([[pos[0], pos[1]]]).T
        transformed_x = ref_pos[0] + transformed_pos[0]
        transformed_y = ref_pos[1] + transformed_pos[1]
        return [transformed_x, transformed_y]

    def _load_scene(self, n_cars):
        goal_x = self.goal[0]
        goal_y = self.goal[1]
        h = 0.1
        car_spawn_radius = np.random.uniform(1.5, 2.5)
        cube_spawn_dist = 2.25

        # Obstacle generation
        min_angle = 0
        max_angle = 30
        stage_1_cubes = []
        stage_2_cubes = []
        for cube_no in range(8):
            angle = np.random.randint(min_angle,max_angle) * np.pi/180.0
            x = goal_x + cube_spawn_dist * np.sin(angle)
            y = goal_y + cube_spawn_dist * np.cos(angle)
            cubeId = p.loadURDF(os.path.join("assets/basic/cube_static.urdf"), \
                                basePosition=[x, y, 0.5])
            if cube_no == 3:
                min_angle = 40
                max_angle = 60
                cube_spawn_dist = 2*cube_spawn_dist
            else:
                min_angle += 90
                max_angle += 90
            if cube_no < 4:
                stage_1_cubes.append([cubeId, x, y])
            else:
                stage_2_cubes.append([cubeId, x, y])
            self.obstacles.append([cubeId, x, y])

        # Get random starting locations for cars
        starting_cubes = random.sample(stage_2_cubes, n_cars)
        min_angle = 0
        max_angle = 90
        for cube in starting_cubes:
            cube_x = cube[1]
            cube_y = cube[2]
            angle = np.random.randint(min_angle, max_angle) * np.pi/180.0
            car_x =  cube_x + cube_x * car_spawn_radius * np.sin(angle) / abs(cube_x)
            car_y = cube_y + cube_y * car_spawn_radius * np.cos(angle) / abs(cube_y)
            car = Racecar([car_x, car_y, h])
            phi = np.arctan2(car_y, car_x) - np.pi
            q = p.getQuaternionFromEuler((0, 0, phi))
            carId = car.get_car_id()
            car_pos = [car_x, car_y, h]
            p.resetBasePositionAndOrientation(carId,car_pos,q)
            self.cars.append(car)
            self.obstacles.append([carId, car_x, car_y])
            # self.prev_pos.append(car_pos)

    def _get_goal_cost(self, pos):
        # make sure goal and car are at the same height
        cost_to_goal = np.sqrt((pos[0] - self.goal[0])**2 + (pos[1] - self.goal[1])**2)
        return cost_to_goal

    def _get_boundaries(self, car_pos):
        # Get car boundaries
        car_boundaries = []
        car_boundaries.append([car_pos[0], car_pos[1] + 0.4/2])
        car_boundaries.append([car_pos[0], car_pos[1] - 0.4/2])
        car_boundaries.append([car_pos[0] + 0.5, car_pos[1] + 0.4/2])
        car_boundaries.append([car_pos[0] + 0.5, car_pos[1] - 0.4/2])
        return car_boundaries

    def _get_obstacle_cost(self, car_yaw, new_car_pos, car_pos, carId):
        obs_dists = []
        new_car_boundaries = self._get_boundaries(new_car_pos)
        # for boundary in new_car_boundaries:
        transformed_boundary = self._transform_coords(car_yaw, new_car_pos, car_pos)
        _, min_obs_dist = self.get_obs_dist(transformed_boundary, carId)
        if min_obs_dist < 1.2:
            obs_cost = 1000
            return obs_cost
        else:
            obs_cost = min_obs_dist
        return 1.0/obs_cost             

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(cameraDistance=8.5, cameraYaw=0, cameraPitch=-91,
                                     cameraTargetPosition=[-0.34, -0.2, 0.0])
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.5, -0.0, 0],
                                                               distance=9,
                                                               yaw=0,
                                                               pitch=-91,
                                                               roll=0,
                                                               upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(self.camera_width) / self.camera_height,
                                                        nearVal=0.1,
                                                        farVal=100.0)

