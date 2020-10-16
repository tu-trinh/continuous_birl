import os
import numpy as np
import pybullet as p
import pybullet_data
from panda import Panda
from objects import YCBObject


class Task():

    def __init__(self):
        # create simulation
        self.urdfRootPath = pybullet_data.getDataPath()
        p.connect(p.GUI) # DIRECT or GUI
        p.setGravity(0, 0, -9.81)

        # set up camera
        self._set_camera()

        # load some scene objects
        p.loadURDF(os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65])
        p.loadURDF(os.path.join(self.urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65])

        # target
        # obj2 = YCBObject('025_mug')
        # obj2.load()
        # p.resetBasePositionAndOrientation(obj2.body_id, [0.7, -0.35, 0.01], [0, 0, 0, 0.75])

        # load a panda robot
        self.panda = Panda()

    def reset(self):
        self.panda.reset()
        return self.panda.state

    def close(self):
        p.disconnect()

    def step(self, action):
        # get current state
        state = self.panda.state

        # action in this example is the joint space velocity
        self.panda.step(mode=0, djoint=action)

        # take simulation step
        p.stepSimulation()

        # return next_state, reward, done, info
        next_state = self.panda.state
        reward = 0.0
        done = False
        info = {}
        return next_state, reward, done, info

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
        p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=20, cameraPitch=-40,
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
