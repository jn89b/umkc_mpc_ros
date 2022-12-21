#!/usr/bin/env python3

import rclpy
import math as m
import os

import casadi as ca
import math as m
import numpy as np
import mavros

from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from umkc_mpc_ros import quaternion_tools

from mavros.base import SENSOR_QOS
from pymavlink import mavutil
from math import cos, sin, pi

## GlOBAL OBSTACLE
OBS_X = -1000.0
OBS_Y = -1000.0
OBS_VEL = 0.00

rob_diam = 1.0
obs_diam = 3.0

N = 10 #number of look ahead steps

v_min = 14
v_max = 16

psi_rate_min = np.deg2rad(-60)
psi_rate_max = np.deg2rad(60)#rad/s

#target parameters
x_target = 0.0
y_target = 3.0
psi_target = np.deg2rad(45.0)
target_velocity = 0.0

# def send_airspeed_command(airspeed,master):
#     master.mav.command_long_send(
#     master.target_system, 
#     master.target_component,
#     mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED, #command
#     0, #confirmation
#     0, #Speed type (0=Airspeed, 1=Ground Speed, 2=Climb Speed, 3=Descent Speed)
#     airspeed, #Speed #m/s
#     -1, #Throttle (-1 indicates no change) % 
#     0, 0, 0, 0 #ignore other parameters
#     )
#     print("change airspeed")

class SimpleModel():
    """
    
    3 States: 
    [x, y, psi]
     
     2 Inputs:
     [v, psi_rate]
    
    """
    def __init__(self):
        self.define_states()
        self.define_controls()
        
    def define_states(self):
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.psi = ca.SX.sym('psi')
        
        self.states = ca.vertcat(
            self.x,
            self.y,
            self.psi
        )
        #column vector of 3 x 1
        self.n_states = self.states.size()[0] #is a column vector 
        
    def define_controls(self):
        self.v_cmd = ca.SX.sym('v_cmd')
        self.psi_cmd = ca.SX.sym('psi_cmd')
        
        self.controls = ca.vertcat(
            self.v_cmd,
            self.psi_cmd
        )
        #column vector of 2 x 1
        self.n_controls = self.controls.size()[0] 
        
    def set_state_space(self):
        #this is where I do the dynamics for state space
        self.x_dot = self.v_cmd * ca.cos(self.psi)
        self.y_dot = self.v_cmd * ca.sin(self.psi)
        self.psi_dot = self.psi_cmd
        
        self.z_dot = ca.vertcat(
            self.x_dot, self.y_dot, self.psi_dot    
        )
        
        #ODE right hand side function
        self.function = ca.Function('f', 
                        [self.states, self.controls],
                        [self.z_dot]
                        ) 
        
        return self.function

class MPC():
    def __init__(self, model, dt_val, N):
        self.model = model
        self.f = model.function        
        
        self.n_states = model.n_states
        self.n_controls = model.n_controls
        
        self.dt_val = dt_val 
        self.N = N

        """REFACTOR THIS ALLOW THE USER TO DEFINE THIS"""
        self.Q = ca.diagcat(1.0, 1.0, 1.0) # weights for states
        self.R = ca.diagcat(1.0, 1.0) # weights for controls
        self.lambda_safe = 1.0 #weight factor for obstacle avoidance
        
        #initialize cost function as 0
        self.cost_fn = 0
        
    def define_bound_constraints(self):
        """define bound constraints of system"""
        self.variables_list = [self.X, self.U]
        self.variables_name = ['X', 'U']
        
        #function to turn decision variables into one long row vector
        self.pack_variables_fn = ca.Function('pack_variables_fn', self.variables_list, 
                                             [self.OPT_variables], self.variables_name, 
                                             ['flat'])
        
        #function to turn decision variables into respective matrices
        self.unpack_variables_fn = ca.Function('unpack_variables_fn', [self.OPT_variables], 
                                               self.variables_list, ['flat'], self.variables_name)

        ##helper functions to flatten and organize constraints
        self.lbx = self.unpack_variables_fn(flat=-ca.inf)
        self.ubx = self.unpack_variables_fn(flat=ca.inf)

        """
        REFACTOR THIS
        """

        self.lbx['U'][0,:] = v_min
        self.ubx['U'][0,:] = v_max

        self.lbx['U'][1,:] = psi_rate_min
        self.ubx['U'][1,:] = psi_rate_max

    def init_decision_variables(self):
        """intialize decision variables for state space models"""
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        self.U = ca.SX.sym('U', self.n_controls, self.N)
        
        #column vector for storing initial and target locations
        self.P = ca.SX.sym('P', self.n_states + self.n_states)

        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        
    def compute_cost(self, obstacle_x, obstacle_y, obstacle_vel):
        """this is where we do integration methods to find cost"""
        #tired of writing self
        #dynamic constraints 
        self.g = []
        self.g = self.X[:,0] - self.P[:self.n_states]
        
        P = self.P
        Q = self.Q
        R = self.R
        n_states = self.n_states
        
        """
        going to add a stationary object and see what happens
        I need to abstract this someway to insert this into the MPC 
        """
        for k in range(N):
            states = self.X[:, k]
            controls = self.U[:, k]
            state_next = self.X[:, k+1]
            
            #penalize states and controls for now, can add other stuff too
            self.cost_fn = self.cost_fn \
                + (states - P[n_states:]).T @ Q @ (states - P[n_states:]) \
                + controls.T @ R @ controls                 

            # self.cost_fn =             
            ##Runge Kutta
            k1 = self.f(states, controls)
            k2 = self.f(states + self.dt_val/2*k1, controls)
            k3 = self.f(states + self.dt_val/2*k2, controls)
            k4 = self.f(states + self.dt_val * k3, controls)
            state_next_RK4 = states + (self.dt_val / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, state_next - state_next_RK4) #dynamic constraints

        """REFACTOR THIS WHAT HAPPENS IF I HAVE NO OBSTACLES"""
        obs_cost = 0 
        for k in range(N):
            #penalize obtacle distance
            x_pos = self.X[0,k]
            y_pos = self.X[1,k]

            #the Jonathan Benson method
            pred_obs_x = obstacle_x + (k*self.dt_val * obstacle_vel)
            obs_distance = ca.sqrt((x_pos - pred_obs_x)**2 + \
                                       (y_pos - obstacle_y)**2)

            
            self.g = ca.vertcat(self.g, obs_distance) 

            obs_cost = obs_cost + obs_distance            

    def init_solver(self):
        """init the NLP solver utilizing IPOPT this is where
        you can set up your options for the solver"""
        
        nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }

        opts = {
            'ipopt': {
                'max_iter': 4000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            # 'jit':True,
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def init_goals(self, start, goal):
        """init goals"""
        self.state_init = ca.DM(start)        # initial state
        self.state_target = ca.DM(goal)  # target state
        
           
    def solve_mpc(self,start,goal):
        """solve the mpc based on initial and desired location"""
        n_states = self.model.n_states
        n_controls = self.model.n_controls
        
        self.state_init = ca.DM(start)        # initial state
        self.state_target = ca.DM(goal)  # target state
        
        # self.t0 = t0
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)         # initial state full

        """REFACTOR THIS NEED TO UPDATE THE OBSTACLE INFORMATION"""
        obs_velocity = 0.0
        target_velocity = 0.0
        
        # origin_obs_x = OBS_X
        # origin_obs_y = OBS_Y
        
        self.origin_obs_x = OBS_X + self.dt_val * obs_velocity
        self.origin_obs_y = OBS_Y #- self.dt_val * obs_velocity/2

        """Jon's advice consider velocity of obstacle at each knot point"""
        #moving target in the y direction
        self.state_target = ca.DM(goal) 

        self.compute_cost(self.origin_obs_x, self.origin_obs_y, obs_velocity)

        """REFACTOR THIS NEED TO ADD OBSTACLES IN THE LBG AND UBG"""
        lbg =  ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))  # constraints lower bound
        lbg[self.n_states*N+n_states:] = rob_diam/2 + obs_diam/2 # -infinity to minimum marign value for obs avoidance
        
        ubg  =  ca.DM.zeros((self.n_states*(self.N+1)+self.N, 1))  # constraints upper bound
        ubg[self.n_states*N+n_states:] = ca.inf#rob_diam/2 + obs_diam/2 #adding inequality constraints at the end

        args = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': self.pack_variables_fn(**self.lbx)['flat'],
            'ubx': self.pack_variables_fn(**self.ubx)['flat'],
        }

        #this is where you can update the target location
        args['p'] = ca.vertcat(
            self.state_init,    # current state
            self.state_target   # target state
        )
        
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(self.X0, n_states*(self.N+1), 1),
            ca.reshape(self.u0, n_controls*self.N, 1)
        )

        #this is where we solve
        sol = self.solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        #unpack as a matrix
        self.u = ca.reshape(sol['x'][self.n_states * (self.N + 1):], 
                            self.n_controls, self.N)
        
        self.X0 = ca.reshape(sol['x'][: n_states * (self.N+1)], 
                                self.n_states, self.N+1)        
        
        #return the controls and states of the system
        return (self.u, self.X0)


class FWOffboardNode(Node):
    def __init__(self):
        super().__init__('fixedwing_node')
        
        self.mpc_traj_publisher = self.create_publisher(
            Path, 'mpc_trajectory', 1
        )

        self.position_sub = self.create_subscription(
            mavros.local_position.Odometry, 
            'mavros/local_position/odom', 
            self.position_callback, 
            qos_profile=SENSOR_QOS)

        self.local_vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 1
        )

        self.local_pos_pub = self.create_publisher(PoseStamped, 
        '/mavros/setpoint_position/local', 10)

        self.attitude_rate_pub = self.create_publisher(
            AttitudeTarget, '/mavros/setpoint_raw/attitude', 1)

        self.body_velocity_pub = self.create_publisher(
            Float32, 'airspeed', 1
        )

        self.z_position = None        
        self.current_state = [None,None,None]
        self.pixhawk_sp  = PoseStamped()

    def publish_body_velocity(self, body_velocity):
        msg = Float32()
        msg.data = float(body_velocity)
        self.body_velocity_pub.publish(msg)

    def position_callback(self, msg):
        # self.get_logger().info('Position: {}'.format(self.odom))
        current_position = msg.pose.pose.position 
        self.current_state[0] = current_position.x
        self.current_state[1] = current_position.y
        self.z_position = current_position.z
 
        #quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll,pitch,yaw = quaternion_tools.euler_from_quaternion(qx, qy, qz, qw)
        #wrap the angles
        self.current_state[2] = (yaw+ (2*np.pi) ) % (2*np.pi);

    def publish_velocity(self, vx, vy, vz, psi_rate):
        vel_msg = TwistStamped()
        # vel_msg.header.stamp = self.get_clock().now().to_msg()
        # vel_msg.header.frame_id = 'map'
        vel_msg.twist.linear.x = float(vx)
        vel_msg.twist.linear.y = float(vy)
        vel_msg.twist.linear.z = float(vz)
        vel_msg.twist.angular.z = float(psi_rate)
        self.local_vel_pub.publish(vel_msg)

    def publish_position(self, x, y, z, yaw_d):
        self.pose = PoseStamped()
        self.pose.pose.position.x = float(x)
        self.pose.pose.position.y = float(y)
        self.pose.pose.position.z = 9.7#float(z)

        if (yaw_d > np.deg2rad(180.0)):
            yaw_d = (-yaw_d - np.rad2deg(90.0))
            self.pixhawk_sp.pose.orientation.w = -sin(yaw_d/2)
            self.pixhawk_sp.pose.orientation.x = 0.0
            self.pixhawk_sp.pose.orientation.y = 0.0
            self.pixhawk_sp.pose.orientation.z = cos(yaw_d/2)
        else:
            yaw_d = (-yaw_d - np.rad2deg(90.0))
            self.pixhawk_sp.pose.orientation.w = sin(yaw_d/2)
            self.pixhawk_sp.pose.orientation.x = 0.0
            self.pixhawk_sp.pose.orientation.y = 0.0
            self.pixhawk_sp.pose.orientation.z = -cos(yaw_d/2)
        
        self.local_pos_pub.publish(self.pose)


    def publish_path(self, x_waypoints, y_waypoints):
        """Visualize path from MPC"""
        path = Path()
        path.poses = []
        # header_name = 'map'
        
        x_waypoints = np.array(x_waypoints)
        y_waypoints = np.array(y_waypoints)

        # x_waypoints = x_waypoints.flatten()
        # y_waypoints = y_waypoints.flatten()

        x_waypoints = x_waypoints[-1]
        y_waypoints = y_waypoints[-1]

        for x,y in zip(x_waypoints, y_waypoints):
            point_pose = PoseStamped()
            # point_pose.header = self.create_header('base_link')
            point_pose.pose.position.x = float(x)
            point_pose.pose.position.y = float(y)
            point_pose.pose.position.z = 9.7
            # point_pose.pose.orientation.w = 1.0
            path.poses.append(point_pose)
            
        self.mpc_traj_publisher.publish(path)

    def publish_rate_commands(self, vx, vy, vz, thrust):
        """publish rate commands"""
        att = AttitudeTarget()
        att.type_mask = 7 #ignore body rate
        att.header.frame_id = 'base_link'
        orientation = quaternion_tools.get_quaternion_from_euler(-0.25, 0.15,
                                                                 0)
        att.orientation.x = orientation[0]
        att.orientation.y = orientation[1]
        att.orientation.z = orientation[2]
        att.orientation.w = orientation[3]

        # att.header.stamp = self.get_clock().now().to_msg()
        # att.type_mask =  128#0b00000000 #64 #64 #ignore orientation#0b00000000
        # att.body_rate.x = 0.0#float(np.deg2rad(vx))
        # att.body_rate.y = float(np.deg2rad(15))
        # att.body_rate.z = 0.0
        att.thrust = 0.75#abs(float(thrust))
        self.attitude_rate_pub.publish(att)


def compute_error(init , end):
    return m.dist(init, end)


def get_time_in_secs(some_node:Node) -> float:
    return some_node.get_clock().now().nanoseconds /1E9

def main(args=None):
    rclpy.init(args=args)
    fixedwing_node = FWOffboardNode()
    current_state = fixedwing_node.current_state

    print("starting")
    
    simplefixed_model = SimpleModel()
    function = simplefixed_model.set_state_space()

    step_horizon = 0.05

    start = [0.0, 0.0, 0.0] #x,y,psi
    end = [x_target, y_target, psi_target]

    fw_mpc = MPC(simplefixed_model, step_horizon, N)
    fw_mpc.init_decision_variables()
    # fw_mpc.init_goals(start, end)
    
    fw_mpc.compute_cost(OBS_X, OBS_Y, OBS_VEL)
    fw_mpc.init_solver()
    fw_mpc.define_bound_constraints()
    controls, states = fw_mpc.solve_mpc(start, end)    
    
    x_traj = states[0,:]
    y_traj = states[1,:]
    psi_traj = states[2,:]

    idx_projection = 2

    while rclpy.ok():
        
        fw_mpc.init_solver()
        rclpy.spin_once(fixedwing_node)

        current_state = fixedwing_node.current_state
        
        current_state = [x_traj[idx_projection], 
                    y_traj[idx_projection], psi_traj[idx_projection]]

        curr_pos = [current_state[0], current_state[1], current_state[2]]
        end_pos = [end[0], end[1], end[2]]

        #compute error
        error = compute_error(curr_pos[:-1], end_pos[:-1])
        print("lateral position: ", curr_pos[:-1])
        if error <= 1.0:
            print("reached goal")
            #rclpy.shutdown()
            #return

        time_1 = get_time_in_secs(fixedwing_node)

        rclpy.spin_once(fixedwing_node)
        current_state = fixedwing_node.current_state

        controls, states = fw_mpc.solve_mpc(current_state, end) 
        time_2 = get_time_in_secs(fixedwing_node)
        # print("time to solve: ", time_2 - time_1)
        
        x_traj = states[0,:]
        y_traj = states[1,:]
        psi_traj = states[2,:]

        #unpack this matrix and publish the controls and trajectory
        cmd_vel = float(controls[0])
        ang_vel_cmd = float(controls[1])

        #publish the position and velocity
        print("cmd_vel: ", cmd_vel)
        vx = cmd_vel * cos(psi_traj[-1])
        vy = cmd_vel * sin(psi_traj[-1])        
        # fixedwing_node.publish_velocity(vx, vy, 0.0, ang_vel_cmd)
        # fixedwing_node.publish_body_velocity(cmd_vel)

        fixedwing_node.publish_position(x_traj[-1], 
        y_traj[-1], 
        fixedwing_node.z_position, 
        psi_traj[-1])

        fixedwing_node.publish_path(x_traj, y_traj)

        rclpy.spin_once(fixedwing_node)

    # fixedwing_node.destroy_node()
    # rclpy.shutdown()

if __name__ == '__main__':
    main()