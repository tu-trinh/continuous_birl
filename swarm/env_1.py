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

        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, 0.1])

        # Load obstacles and cars
        self._load_scene(n_cars)


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

    def _load_scene(self, instances):
        # Location of goal
        goal_x = 0
        goal_y = 0
        h = 0.1
        car_spawn_radius = 1.5
        cube_spawn_dist = 2.0

        # Obstacle generation
        min_angle = 0
        max_angle = 30
        stage_1_cubes = []
        stage_2_cubes = []
        cubes = []
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
            self.cubes.append([cubeId, x, y])
        
        # Get random starting locations for cars
        starting_cubes = random.sample(stage_2_cubes, instances)

        for cube in starting_cubes:
            angle = np.random.randint(min_angle, max_angle) * np.pi/180.0
            car_x =  cube[1] + car_spawn_radius * np.sin(angle)
            car_y = cube[2] + car_spawn_radius * np.cos(angle)
            car = Racecar([car_x, car_y, h])
            self.cars.append(car)
        # closest_points = p.getClosestPoints(car.get_car_id(),cube,100)
            
   
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

