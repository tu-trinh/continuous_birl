import socket
import time
import numpy as np
import pickle
import pygame
import sys


"""
 * teleop the robot, save the state-action pairs
 * includes interaction between modes to make things harder
 * Dylan Losey, September 2020
"""

"""

 * home positions (demos): ./collab/return_home 0.794095 0.684053 0.122566 -1.38936 -1.46769 1.33218 -0.260643
 * home positions (playback): ./collab/return_home 0.841334 1.16946 0.138284 -1.15176 -1.4545 1.33055 -0.00023852
 * goal: x=70, y=35


"""
HOME = np.asarray([0.794095, 0.684053, 0.122566, -1.38936, -1.46769, 1.33218, -0.260643])
Q_MAX = [2.8, 1.7, 2.8, -0.75, 2.8, 3.7, 2.8]
Q_MIN = [-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8]


class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1
        self.timeband = 0.5
        self.lastpress = time.time()

    def input(self):
        pygame.event.get()
        curr_time = time.time()
        dx = self.gamepad.get_axis(0)
        dy = -self.gamepad.get_axis(1)
        dz = -self.gamepad.get_axis(4)
        if abs(dx) < self.deadband:
            dx = 0.0
        if abs(dy) < self.deadband:
            dy = 0.0
        if abs(dz) < self.deadband:
            dz = 0.0
        A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
        B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
        START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
        if A_pressed or B_pressed or START_pressed:
            self.lastpress = curr_time
        return [dx, dy, dz], A_pressed, B_pressed, START_pressed


def connect2robot(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('localhost', PORT))
    s.listen()
    conn, addr = s.accept()
    return conn

def send2robot(conn, state, qdot, limit=1.0):
    qdot = np.asarray(qdot)
    scale = np.linalg.norm(qdot)
    if scale > limit:
        qdot *= limit/scale
    qdot = jointlimits(state, qdot)
    send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())

def send2gripper(conn):
    send_msg = "s"
    conn.send(send_msg.encode())

def listen2robot(conn):
    state_length = 7 + 7 + 7 + 42
    message = str(conn.recv(2048))[2:-2]
    state_str = list(message.split(","))
    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx+1:idx+1+state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
    except ValueError:
        return None
    if len(state_vector) is not state_length:
        return None
    state_vector = np.asarray(state_vector)
    state = {}
    state["q"] = state_vector[0:7]
    state["dq"] = state_vector[7:14]
    state["tau"] = state_vector[14:21]
    state["J"] = state_vector[21:].reshape((7,6)).T
    return state

def readState(conn):
    while True:
        state = listen2robot(conn)
        if state is not None:
            break
    return state

def xdot2qdot(xdot, state):
    J = state["J"]
    W = np.eye(7)
    W[4,4] = 0.1
    J_pinv = W @ J.T @ np.linalg.inv(J @ W @ J.T)
    return J_pinv @ np.asarray(xdot)

def joint2pose(q):
    def RotX(q):
        return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
    def RotZ(q):
        return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    def TransX(q, x, y, z):
        return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
    def TransZ(q, x, y, z):
        return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    H1 = TransZ(q[0], 0, 0, 0.333)
    H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
    H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
    H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
    H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
    H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
    H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
    H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
    H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
    return H[:,3][:3]

def jointlimits(state, qdot):
    for idx in range(7):
        if state["q"][idx] > Q_MAX[idx] and qdot[idx] > 0:
            qdot[idx] = 0.0
        elif state["q"][idx] < Q_MIN[idx] and qdot[idx] < 0:
            qdot[idx] = 0.0
    return qdot


def main():

    usernumber = sys.argv[1]
    filename = sys.argv[2]
    demonstration = []
    steptime = 0.1

    PORT_robot = 8080
    PORT_gripper = 8081
    scaling_linear = 0.1
    scaling_rot = 0.2

    conn = connect2robot(PORT_robot)
    conn_gripper = connect2robot(PORT_gripper)
    interface = Joystick()
    foldername = "demos_task3/user" + usernumber + "/"

    record = False
    gripper_closed = False
    transation_mode = True
    start_time = None
    last_time = None
    state = readState(conn)
    if np.linalg.norm(state["q"] - HOME) > 0.1:
        print("I started in the wrong place!")
        return False

    while True:

        state = readState(conn)
        z, grasp, mode, stop = interface.input()

        if stop and not record:
            record = True
            last_time = time.time()
            start_time = time.time()
            print("I started recording!")
        elif stop and record:
            pickle.dump( demonstration, open( foldername + filename + ".pkl", "wb" ) )
            print(demonstration)
            print("I recorded this many datapoints: ", len(demonstration))
            return True

        if grasp:
            gripper_closed = not gripper_closed
            send2gripper(conn_gripper)

        if mode:
            transation_mode = not transation_mode

        xdot = [0.0]*6

        if transation_mode:
            xdot[0] = scaling_linear * z[0]
            xdot[1] = scaling_linear * z[1]
            xdot[2] = scaling_linear * z[2]
            xdot[4] = scaling_rot * -2 * z[1]
        else:
            xdot[3] = scaling_rot * z[0]
            xdot[4] = scaling_rot * z[1]
            xdot[5] = scaling_rot * z[2]

        x_pos = joint2pose(state["q"])
        if x_pos[2] < 0.1 and xdot[2] < 0:
            xdot[2] = 0

        qdot = xdot2qdot(xdot, state)

        curr_time = time.time()
        if record and curr_time - last_time >= steptime:
            demonstration.append(state["q"].tolist() + qdot.tolist() + [int(gripper_closed)])
            last_time = curr_time

        send2robot(conn, state, qdot)


if __name__ == "__main__":
    main()
