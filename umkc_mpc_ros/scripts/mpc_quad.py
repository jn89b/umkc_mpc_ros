#!/usr/bin/env python3

import rclpy
import math as m
import os

import casadi as ca
import math as m
import numpy as np
import mavros

from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, AttitudeTarget, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from umkc_mpc_ros import quaternion_tools

from mavros.base import SENSOR_QOS

vel_min = 15
vel_max = 15

thrust_max = vel_max
thrust_min = vel_min

psi_min = -np.deg2rad(45)
psi_max = np.deg2rad(45)

class FlatQuadcopterModel():
    def __init__(self):
        
        #model constants for dj100 from paper
        self.k_x = 1 
        self.k_y = 1 
        self.k_z = 1 
        self.k_psi = np.pi/180
        
        #tau constaints
        self.tau_x = 0.8355
        self.tau_y = 0.7701
        self.tau_z = 0.5013
        self.tau_psi = 0.5142 
        
        self.define_states()
        self.define_controls()
        
    def define_states(self) -> None:
        """
        define the 8 flat states of the quadcopter
        
        [x, y, z, psi, vx, vy, vz, psi_dot]
        
        """
        
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.z = ca.SX.sym('z')
        self.psi = ca.SX.sym('psi')
        
        self.vx = ca.SX.sym('vx')
        self.vy = ca.SX.sym('vy')
        self.vz = ca.SX.sym('vz')
        self.psi_dot = ca.SX.sym('psi')
        
        self.states = ca.vertcat(
            self.x, 
            self.y,
            self.z,
            self.psi,
            self.vx,
            self.vy,
            self.vz, 
            self.psi_dot
        )
        
        #column vector of 3 x 1
        self.n_states = self.states.size()[0] #is a column vector 
        
        
    def define_controls(self) -> None:
        """4 motors"""
        self.u_0 = ca.SX.sym('u_0')
        self.u_1 = ca.SX.sym('u_1')
        self.u_2 = ca.SX.sym('u_2')
        self.u_3 = ca.SX.sym('u_3')
        
        self.controls = ca.vertcat(
            self.u_0,
            self.u_1,
            self.u_2,
            self.u_3
        )
        
        #column vector of 2 x 1
        self.n_controls = self.controls.size()[0] 
        
    def set_state_space(self) -> None:
        #this is where I do the dynamics for state space
        self.z_0 = self.vx * ca.cos(self.psi) - self.vy * ca.sin(self.psi)
        self.z_1 = self.vy * ca.sin(self.psi) + self.vy * ca.cos(self.psi)
        self.z_2 = self.vz
        self.z_3 = self.psi_dot
        
        self.x_ddot = (-self.vx + (self.k_x * self.u_0))
        self.y_ddot = (-self.vy + (self.k_y * self.u_1))
        self.z_ddot = (-self.vz + (self.k_z * self.u_2))
        self.psi_ddot = (-self.psi_dot + (self.k_psi * self.u_3))

        #renamed it as z because I have an x variable, avoid confusion    
        self.z_dot = ca.vertcat(
            self.z_0, 
            self.z_1, 
            self.z_2, 
            self.z_3,
            self.x_ddot, 
            self.y_ddot, 
            self.z_ddot, 
            self.psi_ddot
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
        
        """this needs to be changed, let user define this"""
        self.Q = ca.diagcat(1.0, 
                            1.0, 
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0)
        
        self.R = ca.diagcat(1.0, 
                            1.0,
                            1.0,
                            1.0) # weights for controls
        
        #initialize cost function as 0
        self.cost_fn = 0        

    def init_decision_variables(self):
        """intialize decision variables for state space models"""
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)
        self.U = ca.SX.sym('U', self.n_controls, self.N)
        
        #column vector for storing initial and target locations
        self.P = ca.SX.sym('P', self.n_states + self.n_states)
        
        #dynamic constraints 
        self.g = self.X[:,0] - self.P[:self.n_states]


    def define_bound_constraints(self):
        """define bound constraints of system"""
        self.variables_list = [self.X, self.U]
        self.variables_name = ['X', 'U']
        
        #function to turn decision variables into one long row vector
        self.pack_variables_fn = ca.Function('pack_variables_fn', 
                                             self.variables_list, 
                                             [self.OPT_variables], 
                                             self.variables_name, 
                                             ['flat'])
        
        #function to turn decision variables into respective matrices
        self.unpack_variables_fn = ca.Function('unpack_variables_fn', 
                                               [self.OPT_variables], 
                                               self.variables_list, 
                                               ['flat'], 
                                               self.variables_name)

        ##helper functions to flatten and organize constraints
        self.lbx = self.unpack_variables_fn(flat=-ca.inf)
        self.ubx = self.unpack_variables_fn(flat=ca.inf)

        """
        REFACTOR THIS
        """
        #right now still coupled with state space system, 0 means velcoity 1 is psi rate
        self.lbx['X'][4,:] = vel_min
        self.ubx['X'][4,:] = vel_max
        
        self.lbx['X'][5,:] = vel_min 
        self.ubx['X'][5,:] = vel_max
        
        self.lbx['X'][6,:] = vel_min 
        self.ubx['X'][6,:] = vel_max
        
        
        self.lbx['X'][7,:] = psi_min
        self.ubx['X'][7,:] = psi_max
        
        
        self.lbx['U'][0,:] = thrust_min
        self.ubx['U'][0,:] = thrust_max

        self.lbx['U'][1,:] = thrust_min
        self.ubx['U'][1,:] = thrust_max

        self.lbx['U'][2,:] = thrust_min
        self.ubx['U'][2,:] = thrust_max

        self.lbx['U'][3,:] = thrust_min
        self.ubx['U'][3,:] = thrust_max

    def compute_cost(self):
        """this is where we do integration methods to find cost"""
        #tired of writing self
        P = self.P
        Q = self.Q
        R = self.R
        n_states = self.n_states
        N = self.N
        
        for k in range(N):
            states = self.X[:, k]
            controls = self.U[:, k]
            state_next = self.X[:, k+1]
            
            #penalize states and controls for now, can add other stuff too
            self.cost_fn = self.cost_fn \
                + (states - P[n_states:]).T @ Q @ (states - P[n_states:]) \
                + controls.T @ R @ controls
            
            ##Runge Kutta
            k1 = self.f(states, controls)
            k2 = self.f(states + self.dt_val/2*k1, controls)
            k3 = self.f(states + self.dt_val/2*k2, controls)
            k4 = self.f(states + self.dt_val * k3, controls)
            state_next_RK4 = states + (self.dt_val / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, state_next - state_next_RK4) #dynamic constraints

    def init_solver(self):
        """init the NLP solver utilizing IPOPT this is where
        you can set up your options for the solver"""
        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        
        nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }

        opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            # 'jit':True,
            'print_time': 0
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def solve_mpc(self, start,goal):
        
        self.state_init = ca.DM(start)        # initial state
        self.state_target = ca.DM(goal)  # target state

        #control and state as long row
        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)  # initial state full

        time_history = []


        ## the arguments are what I care about
        args = {
            'lbg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints lower bound
            'ubg': ca.DM.zeros((self.n_states*(self.N+1), 1)),  # constraints upper bound
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
            ca.reshape(self.X0, self.n_states*(self.N+1), 1),
            ca.reshape(self.u0, self.n_controls*self.N, 1)
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
        
        self.X0 = ca.reshape(sol['x'][: self.n_states * (self.N+1)], 
                                self.n_states, self.N+1)

        #return the controls and states of the system
        return (self.u, self.X0)
    

def get_time_in_secs(some_node:Node) -> float:
    return some_node.get_clock().now().nanoseconds /1E9

class QuadNode(Node):
    def __init__(self):
        super().__init__('quad_node')
        # self.state_sub = self.create_subscription(State, '/mavros/state', self.state_callback, 10)

        self.global_position_sub = self.create_subscription(mavros.global_position.NavSatFix, 
        '/mavros/global_position/raw/fix', self.global_position_cb, qos_profile=SENSOR_QOS)

        self.state_sub = self.create_subscription(mavros.local_position.Odometry, 
        'mavros/local_position/odom', self.position_callback, qos_profile=SENSOR_QOS)

        self.state_info = [0,0,0,0,0,0,0,0] #x, y, z, psi, vx, vy, vz, psi_dot

        #publish position
        self.local_pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', 1)

        self.local_vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 1
        )

        self.attitude_rate_pub = self.create_publisher(
            AttitudeTarget, '/mavros/setpoint_raw/attitude', 1)

        self.local_vel_raw = self.create_publisher(
            PositionTarget, '/mavros/setpoint_raw/local', 1)

        self.position_info = [0, 0, 0, 0]


    def global_position_cb(self, topic):
            #self.get_logger().info("x %f: y: %f, z: %f" % (topic.latitude, topic.longitude, topic.altitude))
            pass


    def position_callback(self, msg):
        #positions
        self.state_info[0] = msg.pose.pose.position.x
        self.state_info[1] = msg.pose.pose.position.y
        self.state_info[2] = msg.pose.pose.position.z

        #quaternion attitudes
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        roll,pitch,yaw = quaternion_tools.euler_from_quaternion(qx, qy, qz, qw)
        self.state_info[3] = (yaw+ (2*np.pi) ) % (2*np.pi);

        #velocities 
        self.state_info[4] = msg.twist.twist.linear.x
        self.state_info[5] = msg.twist.twist.linear.y
        self.state_info[6] = msg.twist.twist.linear.z

        #angular psi
        self.state_info[7] = msg.twist.twist.angular.z

    def publish_velocity(self, vx, vy, vz, psi_rate):
        vel_msg = TwistStamped()
        # vel_msg.header.stamp = self.get_clock().now().to_msg()
        # vel_msg.header.frame_id = 'map'
        vel_msg.twist.linear.x = float(vx)
        vel_msg.twist.linear.y = float(vy)
        vel_msg.twist.linear.z = float(vz)

        # vel_msg.twist.angular.z = float(psi_rate)

        self.local_vel_pub.publish(vel_msg)


    def publish_pos_cmd(self, x, y, z):
        """publish position command"""
        print("publishing position", x, y, z)
        pose = PoseStamped()
        # pose.header.stamp = self.get_clock().now().to_msg()
        # pose.header.frame_id = 'map'
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        self.local_pos_pub.publish(pose)

    def publish_rate_commands(self, vx, vy, vz, thrust):
        """publish rate commands"""
        att = AttitudeTarget()
        att.header.stamp = self.get_clock().now().to_msg()
        att.header.frame_id = 'map'
        att.type_mask =  0b10000000 #64 #ignore orientation#0b00000000
        att.body_rate.x = float(vx)
        att.body_rate.y = float(vy)
        att.body_rate.z = float(vz)
        att.thrust = abs(float(thrust))
        self.attitude_rate_pub.publish(att)

    def publish_raw_vel(self, vx, vy, vz, yaw):
        """publish raw velocity commands"""
        # print("publishing raw vel", vx, vy, vz)
        pos = PositionTarget()
        pos.header.stamp = self.get_clock().now().to_msg()
        pos.header.frame_id = 'map'
        pos.coordinate_frame = 1
        pos.type_mask = 64#2503 #yaw command#3576 
        pos.position.x = float(vx)
        pos.position.y = float(vy)
        pos.position.z = float(vz)
        # pos.yaw = yaw

        self.local_vel_raw.publish(pos)

def compute_error(init , end):
    return m.dist(init, end)

def main(args=None):

    #start
    rclpy.init(args=args)
    print("starting")
    quad_node = QuadNode()

    #mpc 
    N = 10
    flat_quad_model = FlatQuadcopterModel()
    flat_quad_function = flat_quad_model.set_state_space()

    x_init = 0
    y_init = 0
    z_init = 1
    psi_init = 0

    x_target = 25
    y_target = 25 
    z_target = 25
    psi_target = np.deg2rad(45)

    step_horizon = 0.05

    #mpc parameters
    # start = [x_init, y_init, z_init, psi_init,
    #          0, 0, 0, 0]
    start = quad_node.state_info
    
    end = [x_target, y_target, z_target, psi_target,
           0, 0, 0, 0 ]

    #warm up the solver
    OBS_VEL = 0.0
    quad_mpc = MPC(flat_quad_model, step_horizon, N)
    quad_mpc.init_decision_variables()
    # quad_mpc.init_goals(start, end)
    quad_mpc.compute_cost()
    # quad_mpc.compute_cost(OBS_X, OBS_Y, OBS_VEL)
    quad_mpc.init_solver()
    quad_mpc.define_bound_constraints()
    controls, states = quad_mpc.solve_mpc(start, end)         
    print(states[4,:])

    init_time = get_time_in_secs(quad_node)
    duration_time = 5.0
    
    while rclpy.ok():


        quad_mpc.init_solver()
        current_state = quad_node.state_info
        
        curr_pos = [current_state[0], current_state[1], current_state[2]]
        end_pos = [end[0], end[1], end[2]]

        print("current position", curr_pos)
        #compute error
        error = compute_error(curr_pos, end_pos)

        # if error <= 0.2:
        #     print("reached goal")
        #     rclpy.shutdown()
        #     return
        
        rclpy.spin_once(quad_node)
            
        #solve mpc
        controls, states = quad_mpc.solve_mpc(current_state, end)
        #unpack states and trajectory
        x_traj = states[0,:]
        y_traj = states[1,:]
        z_traj = states[2,:]
        psi_traj = states[3,:]

        vx_traj = states[4,:]
        vy_traj = states[5,:]
        vz_traj = states[6,:]
        psi_dot_traj = states[7,:]

        control_x = controls[1,:]
        # quad_node.publish_pos_cmd(x_traj[-1], y_traj[-1], z_traj[-1])
        # quad_node.publish_pos_cmd(x_traj[0], y_traj[-1], z_traj[-1])
        
        # quad_node.publish_velocity(vx_traj[1], vy_traj[1], vz_traj[1], 0)
        # quad_node.publish_rate_commands(vx_traj[1], vy_traj[1], vz_traj[1], 1)        
        #quad_node.publish_raw_vel(150, 150, 25, np.deg2rad(-180))
        #curr_time = get_time_in_secs(quad_node)
        #quad_node.publish_rate_commands(0.1, 0, 0, 1)

        # if curr_time - init_time > duration_time:
        #     print("time up")
        #     rclpy.shutdown()
        #     return


if __name__=='__main__':
    main()
