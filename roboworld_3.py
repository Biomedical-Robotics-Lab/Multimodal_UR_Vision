#!/usr/bin/env python
# update regrasping algorithm
# update target object flag
import rospy
import sys
import os
import time
import moveit_commander
import moveit_msgs.msg
import math
import numpy as np
import gripper_2F as gripper
from std_msgs.msg import String, Bool
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose, PoseStamped

class RoboWorld_Test(object):
    def __init__(self):
        self.pub_flag = rospy.Publisher('flag_pose', Bool, queue_size=100)
        self.pub_target = rospy.Publisher('target_object', String, queue_size=100)
        self.pub_flag_target = rospy.Publisher('flag_target_str', Bool, queue_size=100)
        self.pub_flag_yolo = rospy.Publisher('flag_yolo', Bool, queue_size=100)
        rospy.init_node('UR_CONTROL', anonymous=True)
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = self.robot.get_group_names()
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name[1])

        self.target_pose = "/target_pose"
        sub_target_pose = rospy.Subscriber(self.target_pose, Pose, self.target_pose_callback)
        sub_flag_finish = rospy.Subscriber("/flag_finish", Bool, self.flag_finish_callback)

    def target_pose_callback(self, msg):
        self.target_pose_msg = msg
    
    def flag_finish_callback(self, msg):
        self.flag_finish = msg.data
    
    def get_flag_finish(self):
        return self.flag_finish

    def get_target_pose(self):
        return self.target_pose_msg

    def get_target_tf(self):
        pose = self.get_target_pose()
        target_position = np.array([[pose.position.x],
                                    [pose.position.y],
                                    [pose.position.z]])
        target_quat = np.array([pose.orientation.x, pose.orientation.y,
                                pose.orientation.z, pose.orientation.w])
        temp_quat = R.from_quat(target_quat)
        target_rot_mat = temp_quat.as_dcm()
        target_transformation_temp = np.hstack((target_rot_mat, target_position))
        dummy = np.array([0.0, 0.0, 0.0, 1.0])
        target_transformation = np.vstack((target_transformation_temp, dummy))
        print("###########target_tf#############")
        print(target_transformation)
        print("#################################")
        return target_transformation
        
    def calc_trans(self):
        cali_mat = np.array([[-1.0, 0.0, 0.0, 0.02],
                            [0.0, -1.0, 0.0, 0.052],
                            [0.0, 0.0, 1.0, 0.016],
                            [0.0, 0.0, 0.0, 1.0]])
        ur_tf = self.get_pose_state()
        target_tf = self.get_target_tf()
        goal_tf = np.dot(ur_tf, cali_mat)
        goal_tf = np.dot(goal_tf, target_tf)
        return goal_tf



    ### UR MOVE
    def move_joints(self, goal):
        self.move_group.set_max_velocity_scaling_factor(0.2)
        self.move_group.set_max_acceleration_scaling_factor(0.2)
        self.move_group.go(goal, wait=True)
        self.move_group.stop()

    def get_joint_state(self):
        joint_state = self.move_group.get_current_joint_values()
        print([joint*180/math.pi for joint in joint_state])

    def get_pose_state(self):
        self.move_group.set_pose_reference_frame('/world')
        temp_pose = self.move_group.get_current_pose(end_effector_link='tool0')
        
        temp_position, temp_quat = temp_pose.pose.position, temp_pose.pose.orientation
        ur_position = np.array([[temp_position.x], [temp_position.y], [temp_position.z]])
        temp_quat = R.from_quat(np.array([temp_quat.x, temp_quat.y, temp_quat.z, temp_quat.w]))
        ur_orientation = temp_quat.as_dcm()
        ur_trans = np.hstack((ur_orientation, ur_position))
        dummy = np.array([0.0, 0.0, 0.0, 1.0])
        ur_trans = np.vstack((ur_trans, dummy))
        return ur_trans
    
    def going_down(self, down):
        self.move_group.set_pose_reference_frame('/world')
        temp_pose = self.move_group.get_current_pose(end_effector_link='tool0')
        temp_position, temp_quat = temp_pose.pose.position, temp_pose.pose.orientation
        temp_position.z -= down

        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_max_acceleration_scaling_factor(0.1)
        pose_goal = Pose()
        goal_position = pose_goal.position
        goal_position.x, goal_position.y, goal_position.z = temp_position.x, temp_position.y, temp_position.z
        goal_ori = pose_goal.orientation
        goal_ori.x, goal_ori.y, goal_ori.z, goal_ori.w = temp_quat.x, temp_quat.y, temp_quat.z, temp_quat.w
        
        self.move_group.set_pose_target(pose_goal, end_effector_link='tool0')
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def going_up(self, up):
        self.move_group.set_pose_reference_frame('/world')
        temp_pose = self.move_group.get_current_pose(end_effector_link='tool0')
        temp_position, temp_quat = temp_pose.pose.position, temp_pose.pose.orientation
        temp_position.z += up

        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_max_acceleration_scaling_factor(0.1)
        pose_goal = Pose()
        goal_position = pose_goal.position
        goal_position.x, goal_position.y, goal_position.z = temp_position.x, temp_position.y, temp_position.z
        goal_ori = pose_goal.orientation
        goal_ori.x, goal_ori.y, goal_ori.z, goal_ori.w = temp_quat.x, temp_quat.y, temp_quat.z, temp_quat.w
        
        self.move_group.set_pose_target(pose_goal, end_effector_link='tool0')
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def going_forward(self):
        self.move_group.set_pose_reference_frame('/world')
        temp_pose = self.move_group.get_current_pose(end_effector_link='tool0')
        temp_position, temp_quat = temp_pose.pose.position, temp_pose.pose.orientation
        temp_position.y += 0.05

        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_max_acceleration_scaling_factor(0.1)
        pose_goal = Pose()
        goal_position = pose_goal.position
        goal_position.x, goal_position.y, goal_position.z = temp_position.x, temp_position.y, temp_position.z
        goal_ori = pose_goal.orientation
        goal_ori.x, goal_ori.y, goal_ori.z, goal_ori.w = temp_quat.x, temp_quat.y, temp_quat.z, temp_quat.w
        
        self.move_group.set_pose_target(pose_goal, end_effector_link='tool0')
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def going_backward(self):
        self.move_group.set_pose_reference_frame('/world')
        temp_pose = self.move_group.get_current_pose(end_effector_link='tool0')
        temp_position, temp_quat = temp_pose.pose.position, temp_pose.pose.orientation
        temp_position.y -= 0.05

        self.move_group.set_max_velocity_scaling_factor(0.1)
        self.move_group.set_max_acceleration_scaling_factor(0.1)
        pose_goal = Pose()
        goal_position = pose_goal.position
        goal_position.x, goal_position.y, goal_position.z = temp_position.x, temp_position.y, temp_position.z
        goal_ori = pose_goal.orientation
        goal_ori.x, goal_ori.y, goal_ori.z, goal_ori.w = temp_quat.x, temp_quat.y, temp_quat.z, temp_quat.w
        
        self.move_group.set_pose_target(pose_goal, end_effector_link='tool0')
        self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def ur_move(self, goal_trans):
        self.move_group.set_pose_reference_frame('/world')
        pose_position = [goal_trans[0][3], goal_trans[1][3], goal_trans[2][3]]
        pose_orientation = np.array(goal_trans[0:-1, 0:-1])
        temp_orientation = R.from_dcm(pose_orientation)
        pose_quat = temp_orientation.as_quat()
        
        self.move_group.set_max_velocity_scaling_factor(0.2)
        self.move_group.set_max_acceleration_scaling_factor(0.2)
        pose_goal = Pose()
        pose_goal.position.x = pose_position[0]
        pose_goal.position.y = pose_position[1]
        pose_goal.position.z = pose_position[2] + 0.35
        pose_goal.orientation.x = pose_quat[0]
        pose_goal.orientation.y = pose_quat[1]
        pose_goal.orientation.z = pose_quat[2]
        pose_goal.orientation.w = pose_quat[3]

        print("#######final result######")
        print(pose_goal)
        print("#########################")
        
        self.move_group.set_pose_target(pose_goal, end_effector_link='tool0')
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        

def main():
    test_obj = RoboWorld_Test()
    #ur3 home
    home = np.array([joint*math.pi/180 for joint in [65.87, -93.02, 62.9, -59.95, -90.3, 156.43]])
    unload = np.array([joint*math.pi/180 for joint in [-23.21,-83.92,37.06,-43.1,-90.44,156.32]])
    
    while True:
        user_input = raw_input("demo_cmd :")
        if user_input == "home":
            test_obj.move_joints(home)
        elif user_input == "unload":
            test_obj.move_joints(unload)
        elif user_input == "down":
            down = 0.05
            test_obj.going_down(down)
        elif user_input == "up":
            up = 0.05
            test_obj.going_up(up)
        elif user_input == "forward":
            test_obj.going_forward()
        elif user_input == "backward":
            test_obj.going_backward()
        elif user_input == "move":
            goal_trans = test_obj.calc_trans()
            test_obj.ur_move(goal_trans)
        elif user_input == "joint":
            test_obj.get_joint_state()
        #####gripper#####
        elif user_input == "init":
            gripper.grip_init()
        elif user_input == "close":
            gripper.grip_close()
        elif user_input == "open":
            gripper.grip_open()
        ####demo####
        elif user_input == 'test':
            test_obj.pub_flag_target.publish(True)
        elif user_input == "demo":
            pub_input = []

            while True:
                target_input = raw_input("target_insert :")
                if target_input == 'q':
                    pub_input = str(pub_input)
                    test_obj.pub_target.publish(pub_input)
                    print(pub_input)
                    test_obj.pub_flag_target.publish(True)
                    time.sleep(1.5)
                    break
                else:
                    pub_input.append(target_input)
                    print(pub_input)
            # if non-target when regrasp process
            while True:
                print('## finish_falg :', test_obj.get_flag_finish())
                if test_obj.get_flag_finish():
                    test_obj.pub_flag_yolo.publish(False)
                    print('finish')
                    break
                test_obj.pub_flag_yolo.publish(True)
                time.sleep(3)
                flag_pose = True
                test_obj.pub_flag_yolo.publish(False)
                test_obj.pub_flag.publish(flag_pose)
                goal_trans = test_obj.calc_trans()
                test_obj.ur_move(goal_trans)
                print("move")
                test_obj.going_down(0.06)
                gripper.grip_close()
                if not(gripper.grip_flag()):
                    print("Fail")
                    test_obj.going_up(0.06)
                    test_obj.move_joints(home)
                    gripper.grip_open()
                    flag_pose = False
                    test_obj.pub_flag.publish(flag_pose)
                    continue
                else:
                    print("Success")
                    test_obj.going_up(0.06)
                    test_obj.move_joints(unload)
                    test_obj.going_down(0.06)
                    gripper.grip_open()
                    test_obj.going_up(0.06)
                    test_obj.move_joints(home)
                    print('move_home')
                    flag_pose = False
                    test_obj.pub_flag.publish(flag_pose)

        elif user_input == "flag":
            flag_cmd = raw_input('flag :')
            if flag_cmd == "True":
                flag_pose = True
                test_obj.pub_flag.publish(flag_pose)
            elif flag_cmd == "False":
                flag_pose = False
                test_obj.pub_flag.publish(flag_pose)

        elif user_input == "q" or "Q":
            break
        else:
            print("invalid input")


if __name__ == '__main__':
    main()