from tkinter import S
import numpy as np
import cv2
import time
from gym import spaces

import gym
from gym.core import ActionWrapper
import numpy as np
from gym import spaces
import os

from numpy.core.defchararray import count
import rospy
from PIL import Image
import math
from collections import deque

from relod.envs.visual_franka_reacher.franka_utils import *

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import time
import logging

from franka_interface import ArmInterface, RobotEnable, GripperInterface
# ids camera lib for use of IDS ueye cameras.
# https://www.ids-imaging.us/files/downloads/ids-peak/readme/ids-peak-linux-readme-1.2_EN.html
#import ids
import time
import signal
import cv2
import multiprocessing
from gym.spaces import Box as GymBox


class FrankaPanda_Visual_Reacher(gym.Env):
    """
    Gym env for the real franka robot. Set up to perform the placement of a peg that starts in the robots hand into a slot
    """
    def __init__(self, 
                 dt=0.04, 
                 image_history_size=3, 
                 image_width=160, 
                 image_height=90, 
                 episode_steps=100, 
                 camera_index=0, 
                 seed=9,
                 experiment_type='min_time',  # options: min_time, dense, eval
                 size_tol=0.10,
                 print_target_info=False):
        np.random.seed(seed)
        self.DT= dt
        self.dt = dt
        self.ep_time = 0
        self.epi_step = episode_steps
        signal.signal(signal.SIGINT, self.exit_handler)
        self.configs = configure('relod/envs/visual_franka_reacher/reacher.yaml')
        self.conf_exp = self.configs['experiment']
        self.conf_env = self.configs['environment']
        rospy.init_node("franka_robot_gym")
        self.init_joints_bound = self.conf_env['reset-bound']
        #self.target_joints = self.conf_env['target-bound']
        self.safe_bound_box = np.array(self.conf_env['safe-bound-box'])
        self.target_box = np.array(self.conf_env['target-box'])
        self.joint_angle_bound = np.array(self.conf_env['joint-angle-bound'])
        self.joint_torque_bound = np.array(self.conf_env['joint-torque-bound'])
        self.return_point = self.conf_env['return-point']
        self.out_of_boundary_flag = False
        self.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']

        self.experiment_type = experiment_type
        self._size_tol = size_tol
        self._print_target_info = print_target_info
        self.robot = ArmInterface(True)
        force = 1e-6
        self.robot.set_collision_threshold(cartesian_forces=[force,force,force,force,force,force])
        self.robot.exit_control_mode(0.1)
        self.robot_status = RobotEnable()
        self.control_frequency = 1/dt
        self.rate = rospy.Rate(self.control_frequency)

        self.ct = dt
        self.tv = time.time()

        self._image_width = image_width
        self._image_height = image_height

        self._image_history_size = image_history_size
        self._image_history = np.zeros((3 * image_history_size, self._image_width, self._image_height))

        self.joint_states_history = deque(np.zeros((5, 21)), maxlen=5)
        self.torque_history = deque(np.zeros((5, 7)), maxlen=5)
        self.last_action_history = deque(np.zeros((5, 7)), maxlen=5)
        self.time_out_reward = False
        action_dim = 7
        self.prev_action = np.zeros(action_dim)
        self.obs_image = None

        # self.camera = camera()
        ####
        self.camera = camera(self._image_width, self._image_height, camera_index)
        
        ####
        self.previous_place_down = None

        # self.joint_action_limit = 0.15 # changed from 0.2
        # self.action_space = GymBox(low=-self.joint_action_limit * np.ones(7), high=self.joint_action_limit*np.ones(7))

        self.joint_action_limit_scale = 4
        self.joint_action_limit = np.array([1, 1, 1, 1, 0.1, 0.1, 0.1]) * self.joint_action_limit_scale
        self.action_space = GymBox(low=-self.joint_action_limit, high=self.joint_action_limit)
        self.joint_angle_low = [j[0] for j in self.joint_angle_bound]
        self.joint_angle_high = [j[1] for j in self.joint_angle_bound]

        self.observation_space = GymBox(
            low=np.array(
                self.joint_angle_low  # q_actual
                + list(-np.ones(7)*self.joint_action_limit)  # qd_actual
                + list(-np.ones(7)*self.joint_action_limit)  # previous action in cont space
            ),
            high=np.array(
                self.joint_angle_high  # q_actual
                + list(np.ones(7)*self.joint_action_limit)  # qd_actual
                + list(np.ones(7)*self.joint_action_limit)    # previous action in cont space
            )
        )

        self.image_space = GymBox(low=0., high=255., 
        shape=[3 * image_history_size, self._image_width, self._image_height],
        dtype=np.uint8)
        
    def reset(self):
        """
        reset robot to random pose
        Returns
        -------
        object
            Observation of the current state of this env in the format described by this envs observation_space.
        """
        self.time_steps = 0
        self.ep_time = 0
        self.robot_status.enable()
        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))

        self.target_pose = [np.random.uniform(box_range[0], box_range[1]) for box_range in self.target_box]
        self.target_pose[2] = 0.5  # set z to string length
        self.reset_ee_quaternion = [0,-1.,0,0]
        
        _ = self.render()  # skip one frame of the camera
        obs = self.get_state()
        
        self.out_of_boundary_flag = False

        reset_pose = dict(zip(self.joint_names, [random.randint(-10, 10) / 100, 0, 0, -1.6, 0, 1.6, 0.8]))
        smoothly_move_to_position_vel(self.robot, self.robot_status, reset_pose, MAX_JOINT_VELs=1.3)
        print(reset_pose)

        reset_pose = dict(zip(self.joint_names, self.return_point))
        reset_pose['panda_joint4'] = np.random.uniform(-2, -1.5) # changed from [-2.25, -1.0]
        reset_pose['panda_joint6'] = np.random.uniform(1.8, 2.1)

        smoothly_move_to_position_vel(self.robot, self.robot_status, reset_pose ,MAX_JOINT_VELs=1.3)

        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))

        # get the observation
        obs_robot = self.get_state()
        obs = np.concatenate((obs_robot["joints"], obs_robot["joint_vels"], [0]*7))
        
        self.time_steps = 0

        self.tv = time.time()

        # Stabilize the image
        for i in range(3):
            self.camera.get_state() 
        
        img = np.transpose(self.camera.get_state(), (2, 1, 0))

        if self._image_history_size > 1:
            self._image_history[:-3, :, :] = self._image_history[3:, :, :]
            self._image_history[-3:, :, :] = img
        
        self.reset_time = time.time()

        return self._image_history.copy(), obs.copy()

    def render(self):
        # get camera image
        
        width, height = self.conf_env['image-width'], self.conf_env['image-width']
        '''
        _, _ = self.cam.next()  # skip one frame
        img, meta = self.cam.next()
        pil_img = Image.fromarray(img).convert('L').crop(self.crop_bbox).resize((width, height))
        if self.conf_env['visualization']:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bgr_img = bgr_img[108:1000 + 108, 468:1000 + 468]
            self.eye_in_hand_cam_pub.publish(self.br.cv2_to_imgmsg(bgr_img, "bgr8"))
        #return np.array(pil_img)
        '''
        return np.zeros((width, height))

    def get_robot_jacobian(self):
        return self.robot.zero_jacobian()
 
    def euler_from_quaternion(self,q):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        w,x,y,z = q        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def get_state(self):
        # get object state
        # self.obs_object = self.camera.get_state()
        
        # get robot states
        joint_angles = extract_values(self.robot.joint_angles(), self.joint_names)
        joint_velocitys = extract_values(self.robot.joint_velocities(), self.joint_names)
        # joint_efforts = extract_values(self.robot.joint_efforts(), self.joint_names)
        ee_pose = self.robot.endpoint_pose()
        ee_quaternion = [ee_pose['orientation'].w, ee_pose['orientation'].x,
                         ee_pose['orientation'].y, ee_pose['orientation'].z]

        image = self.camera.get_state()
        # print("time used", time.time() - t)
        self.last_action_history.append(self.prev_action)
        
        observation = {
            'image': np.array(image),
            'last_action': self.prev_action,
            'joints': np.array(joint_angles),
            'joint_vels': np.array(joint_velocitys)
        }
        # print('orientation',ee_pose['orientation'])
        self.ee_position = ee_pose['position']
        # print(self.ee_position)
        self.ee_position_table = np.array([1.07-self.ee_position[0], 0.605-self.ee_position[1], self.ee_position[2]])
        self.ee_orientation = ee_quaternion
        #return observation['joints']
        return observation

    def get_reward_done(self, image, action):
        """
        Calculates the reward and done based on the current state of the agent and the environment.

        Parameters
        ----------
        image : numpy array

        Returns
        -------
        float
            Value of the reward.
        
        bool
            Done flag
        """

        image = image[:, :, -3:]
        lower = [0, 0, 120]
        upper = [120, 90, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(image, lower, upper)

        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color
        low_red = np.array([151, 155, 84])
        high_red = np.array([179, 255, 255])
        mask = cv2.inRange(hsv_frame, low_red, high_red)

        kernel = np.ones((3, 3), 'uint8')
        mask = cv2.dilate(mask, kernel, iterations=5)
        mask = cv2.erode(mask, kernel, iterations=5)
        
        target_size = np.sum(mask/255.) / mask.size

        size_x, size_y = mask.shape

        if self.experiment_type == 'min_time':
            reward = -1
            done = target_size >= self._size_tol
        elif self.experiment_type == 'dense':
            # reward =  math.floor(target_size * 1000) / 100
            if 255 in mask:
                xs, ys = np.where(mask == 255.)
                reward_x = 1 / 2  - np.abs(xs - int(size_x / 2)) / size_x
                reward_y = 1 / 2 - np.abs(ys - int(size_y / 2)) / size_y
                reward = np.sum(reward_x * reward_y) / self._image_width / self._image_height
            else:
                reward = 0
            reward *= 100
            reward -= (action**2).sum() * 0.1 # changed factor from 0.1

            done = self.time_steps >= self.epi_step

        else:
            reward = 1.0 if target_size >= self._size_tol else 0
            done = self.time_steps >= self.epi_step
        
        if self._print_target_info:
            if self.experiment_type == 'min_time':
                print(round(target_size, 3), end=', ')
            else:
                print(reward, end=', ')

            if done or self.time_steps % 25 == 0:
                print()

        return reward, done

    def out_of_boundaries(self):
        x, y, z = self.robot.endpoint_pose()['position']
        
        x_bound = self.safe_bound_box[0,:]
        y_bound= self.safe_bound_box[1,:]
        z_bound = self.safe_bound_box[2,:]
        if scalar_out_of_range(x, x_bound):
            print('x out of bound, motion will be aborted! x {}'.format(x))
            return True
        if scalar_out_of_range(y, y_bound):
            print('y out of bound, motion will be aborted! y {}'.format(y))
            return True
        if scalar_out_of_range(z, z_bound):
            print('z out of bound, motion will be aborted!, z {}'.format(z))
            return True
        return False

    def apply_joint_vel(self, joint_vels):
        joint_vels = dict(zip(self.joint_names, joint_vels))
        self.robot.set_joint_velocities(joint_vels)
        
        return True

    def apply_joint_torq(self, joint_torqs):
        joint_torqs = dict(zip(self.joint_names, joint_torqs))
        self.robot.set_joint_torques(joint_torqs)
        
        return True
    
    def get_gravity_comp(self):
        return self.robot.gravity_comp()
    
    def get_coriolis_comp(self):
        return self.robot.coriolis_comp()

    def step(self, action): 
        self.ep_time += self.dt
        self.time_steps += 1
        
        self.robot_status.enable()

        # scale output of actor network to joint torque
        if self.joint_action_limit_scale > 1:
            scaled_action = action * self.joint_action_limit_scale
        
        # limit joint action
        # action = action.reshape(-1)
        # action = np.clip(action, -self.joint_action_limit, self.joint_action_limit)
        # for i in range(action.shape[0]):
            # action[i] = np.clip(action[i], -self.joint_action_limit[i], self.joint_action_limit[i])
        # convert joint velocities to pose velocities
        # pose_action = np.matmul(self.get_robot_jacobian(), action)

        # limit action
        # pose_action[:3] = np.clip(pose_action[:3], -pose_vel_limit, pose_vel_limit)

        # safety
        out_boundary = self.out_of_boundaries()
        # pose_action[:3] = self.safe_actions(pose_action[:3])

        # calculate joint actions
        # d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
        # for i in range(3):
        #     if d_angle[i] < -np.pi:
        #         d_angle[i] += 2*np.pi
        #     elif d_angle[i] > np.pi:
        #         d_angle[i] -= 2*np.pi
        # d_angle *= 0.5

        # d_X = pose_action
        
        # handle joint torques for the case that end-effector is out of the Cartesian box
        if out_boundary:
            pose_action = np.zeros((6,))
            pose_action[:3] = self.safe_actions(pose_action[:3])
            scaled_action += self.get_joint_torq_from_pos_torq(pose_action) # changed from = rather than +=
            print("out boundary action: ", scaled_action)

        # handle joint torques in the case that joint angels are out of the joint box
        scaled_action = self.handle_joint_angle_in_bound(scaled_action)

        # handle joint torques to be within their limit
        scaled_action = scaled_action.reshape(-1)
        for i in range(scaled_action.shape[0]):
            scaled_action[i] = np.clip(scaled_action[i], -self.joint_action_limit[i], self.joint_action_limit[i])
        
        self.prev_action = action # this action is part of the observation space

        # it should be checked that the applied joint torque is in the nominal limitation
        scaled_action = self.handle_joint_torque_nominal(scaled_action)

        # the applied joint torque is the torque from actor and the torque related to gravity
        # it seems that the robot is already gravity compensated
        # applied_action = action + self.get_gravity_comp()
        # here applying coreiolios compensation
        applied_action = scaled_action # removed self.get_coriolis_comp()

        print("applied_action: ", applied_action)

        self.apply_joint_torq(applied_action)

        delay = (self.ep_time + self.reset_time) - time.time()
        if delay > 0:
            time.sleep(np.float64(delay))

        # get next observation
        observation_robot = self.get_state()

        # calculate reward and done
        reward, done = self.get_reward_done(observation_robot["image"], scaled_action)
        
        # construct the state
        obs = np.concatenate((observation_robot["joints"], observation_robot["joint_vels"], action))
             
        img = np.transpose(observation_robot["image"], (2, 1, 0))

        if self._image_history_size > 1:
            self._image_history[:-3, :, :] = self._image_history[3:, :, :]
            self._image_history[-3:, :, :] = img
        
        return self._image_history.copy(), obs.copy(), reward, done, {}
    

    def handle_joint_angle_in_bound(self, action):
        current_joint_angle = self.robot.joint_angles()
        in_bound = [False] * 7
        for i, joint_name in enumerate(self.joint_names):
            if current_joint_angle[joint_name] > 0.05 + self.joint_angle_bound[i][1]:
                print("Joint angle out of the bound.")
                 
                action[i] = -self.joint_action_limit[i]
            elif current_joint_angle[joint_name] < -0.05+ self.joint_angle_bound[i][0]:
                print("Joint angle out of the bound.")
                action[i] = self.joint_action_limit[i]
        return action
    
    def handle_joint_torque_nominal(self, scaled_action):
        applied_joint_torque = scaled_action + self.get_coriolis_comp() + self.get_gravity_comp()
        in_bound = [False] * 7
        for i, joint_name in enumerate(self.joint_names):
            if applied_joint_torque[i] > self.joint_torque_bound[i][1]:
                print("Joint torque out of the bound.")
                error = applied_joint_torque[i] - self.joint_torque_bound[i][1]
                scaled_action[i] -= error
            elif applied_joint_torque[i] < self.joint_torque_bound[i][0]:
                print("Joint torque out of the bound.")
                error = applied_joint_torque[i] - self.joint_torque_bound[i][1]
                scaled_action[i] += error
        return scaled_action

    def get_timeout_reward(self):
        if self.time_out_reward:
            reward = -1
            print('call time out reward {:+.3f}'.format(reward))
            return reward
        else:
            return 0

    def move_to_pose_ee(self, ref_ee_pos, pose_vel_limit=0.2):
        counter = 0
        # print('11111', rospy.Time.now())
        
        while True:
            self.robot_status.enable()
            # print(self.robot_status.state())
            counter += 1
            #action = agent.act(observations['ee_states'], ref_ee_pos, self.get_robot_jacobian(), add_noise=False)
            self.get_state()
            action = np.zeros((4,))
            action[:3] = ref_ee_pos-self.ee_position
            action[-1] = 1
            
            #if max(np.abs(action[:3])) < 0.005 or 
            #print(action)
            if max(np.abs(action[:3])) < 0.005 or counter > 100:
                break

            #self.step(action, ignore_safety=True)
            # limit action
            pose_action = np.clip(action[:3], -pose_vel_limit, pose_vel_limit)

            # calculate joint actions
            d_angle =  np.array(self.euler_from_quaternion(self.reset_ee_quaternion)) - np.array(self.euler_from_quaternion(self.ee_orientation))
            for i in range(3):
                if d_angle[i] < -np.pi:
                    d_angle[i] += 2*np.pi
                elif d_angle[i] > np.pi:
                    d_angle[i] -= 2*np.pi
            d_angle *= 0.5
            #print('d_angle', d_angle)
            d_X = np.array([pose_action[0], pose_action[1], pose_action[2], d_angle[0],d_angle[1],d_angle[2]])
            joints_action = self.get_joint_vel_from_pos_vel(d_X)
            # print('joints_action', joints_action)
            self.apply_joint_vel(joints_action)
            
            # action cycle time
            self.rate.sleep()
        self.apply_joint_vel(np.zeros((7,)))

    def get_joint_vel_from_pos_vel(self, pose_vel):
        return np.matmul(np.linalg.pinv( self.get_robot_jacobian() ), pose_vel)
    
    def get_joint_torq_from_pos_torq(self, pose_torq):
        return np.matmul(np.transpose( self.get_robot_jacobian() ), pose_torq)

    def safe_actions(self, action):
        out_boundary = self.out_of_boundaries()
        x, y, z = self.robot.endpoint_pose()['position'] # it gives a 4*4 matrix. the last column is the tip position.
        self.box_Normals = np.zeros((6,3))
        self.box_Normals[0,:] = [1,0,0]
        self.box_Normals[1,:] = [-1,0,0]
        self.box_Normals[2,:] = [0,1,0]
        self.box_Normals[3,:] = [0,-1,0]
        self.box_Normals[4,:] = [0,0,1]
        self.box_Normals[5,:] = [0,0,-1]
        self.planes_d = [   self.safe_bound_box[0][0],
                            -self.safe_bound_box[0][1],
                            self.safe_bound_box[1][0],
                            -self.safe_bound_box[1][1],
                            self.safe_bound_box[2][0],
                            -self.safe_bound_box[2][1]]
        if out_boundary:
            action = np.zeros((3,))
            for i in range(6):
                # action += 0.05 * self.box_Normals[i] * ( (self.box_Normals[i].dot(np.array([x,y,z])) - self.planes_d[i]) < 0 ) 
                ####
                action += self.joint_action_limit_scale * 1.75 * self.box_Normals[i] * ( (self.box_Normals[i].dot(np.array([x,y,z])) - self.planes_d[i]) < 0 ) # changed the factor from 1

        return action
    
    def exit_handler(self,signum):
        self.camera.thread.join()
        self.camera.release()
        exit(signum)
    
    def terminate(self):
        self.close()
        self.exit_handler(1)

    def close(self):
        # stop the robot
        self.apply_joint_vel(np.zeros((7,)))
        self.camera.close()

    def seed(self, seed):
        np.random.seed(seed)


class camera():
    def __init__(self, image_width, image_height, camera_index=0):

        cv2.setNumThreads(1)
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.image_width = image_width
        self.image_height = image_height
        self.frame= None 

    def get_state(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (self.image_width, self.image_height))
            self.frame = frame

        return self.frame
    
    def close(self):
        self.cap.release()
        time.sleep(1)



