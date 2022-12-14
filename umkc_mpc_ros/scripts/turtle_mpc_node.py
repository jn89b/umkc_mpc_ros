#!/usr/bin/env python3

import rclpy
import math as m
import os

import casadi as ca
import numpy as np

from time import time
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Header
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from umkc_mpc_ros import TurtleBotNode
from umkc_mpc_ros.msg import TurtleCmd

"""

Decoupling the problem:
- Generate a trajectory based on current state
- Publish the trajectory to the topic of nav_msgs.msg Path
- Use the trajectory to generate a control input
- Publish the control input to the topic of geometry_msgs.msg Twist
- Use the control input to generate a new trajectory
- Repeat

To do: 
    - Implement MPC for Turtlebot by:
        - Creating a class for MPC
        - Allow us to input dynamics 
        - Allow us define the cost function 
        = Allow us to define the constraints
        - Allow us to define the initial state
        - Allow us to define the reference trajectory
        
Future I might just inheritc MPC and then override the methods such as :
    - Constraints
    - Cost function
    

Publishing trajectory:
- Publish to the topic of nav_msgs.msg Path 

"""


## GlOBAL OBSTACLE
OBS_X = 2.0
OBS_Y = 2.0

rob_diam = 1.0
obs_diam = 0.65

N = 40 #number of look ahead steps

v_min = 0.05
v_max = 0.2

psi_rate_min = np.deg2rad(-30)
psi_rate_max = np.deg2rad(30)#rad/s


#target parameters
x_target = 4.0
y_target = 4.0
psi_target = np.deg2rad(5.0)
target_velocity = 0.0

class TurtleBotModel():
    """
    TurtleBotModel
    
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
        self.Q = ca.diagcat(1.0, 1.0, 0.0) # weights for states
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
            pred_obs_x = obstacle_x + (self.dt_val * obstacle_vel)
            
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
                'print_level': 1,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            # 'jit':True,
            'print_time': 1
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
        
        origin_obs_x = OBS_X
        origin_obs_y = OBS_Y

        origin_obs_x = origin_obs_x + self.dt_val * obs_velocity
        origin_obs_y = origin_obs_y #- self.dt_val * obs_velocity/2

        """Jon's advice consider velocity of obstacle at each knot point"""
        #moving target in the y direction

        self.state_target = ca.DM(goal) 

        self.compute_cost(origin_obs_x, origin_obs_y, obs_velocity)
        t1 = time()
        self.init_solver()
        print("time diff is ", time()- t1)

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
    
class MPCTurtleNode(TurtleBotNode.TurtleBotNode):
    def __init__(self, node_name, namespace =''):
        super().__init__(node_name)
        self.ns = namespace
        
        self.mpc_traj_publisher = self.create_publisher(
            Path, self.ns+'/mpc_trajectory', 1
        )
        
        self.mpc_cmd_publisher = self.create_publisher(
            TurtleCmd, self.ns+'/mpc_turtle_cmd', 1
        )
        

    def create_header(self, frame_id):
        """create header information to publish messaage"""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()

        header.frame_id = frame_id
        return header

        
    def publish_path(self, x_waypoints, y_waypoints, path_pub):
        """Visualize path from MPC"""
        path = Path()
        path.poses = []
        # header_name = 'map'
        
        for x,y in zip(x_waypoints, y_waypoints):
            point_pose = PoseStamped()
            point_pose.header = self.create_header('odom')
            point_pose.pose.position.x = x
            point_pose.pose.position.y = y
            point_pose.pose.orientation.w = 1.0
            path.poses.append(point_pose)
            
        path_pub.publish(path)
             
    
    def publish_cmd(self, cmd_vel, ang_vel):
        cmd = TurtleCmd()
        cmd.vel = cmd_vel
        cmd.ang_vel = ang_vel
        
        self.mpc_cmd_publisher.publish(cmd)
    
             
        
def compute_error(init , end):
    return m.dist(init, end)
        
def main_mpc():
    """performe the mpc actions"""
    rclpy.init(args=None)
    print("starting")
    
    #turtlebot
    turtle_model = TurtleBotModel()
    turtle_model_function = turtle_model.set_state_space()
    print("function", turtle_model.function)

    namespace = ''
    
    #mpc parameters
    start = [0.0, 0.0, 0.0] #x,y,psi
    goal = [x_target, y_target, psi_target]

    step_horizon = 0.1
    
    #warm up the solver
    turtle_mpc = MPC(turtle_model, step_horizon, N)
     
    turtle_mpc.init_decision_variables()
    turtle_mpc.init_goals(start,goal)
    turtle_mpc.compute_cost(OBS_X, OBS_Y, 0.5)
    turtle_mpc.define_bound_constraints()
    
    controls, states = turtle_mpc.solve_mpc(start, goal) 
    #turtlebot_node = TurtleBotNode.TurtleBotNode('pursuer')    
    mpc_turtle_node = MPCTurtleNode('help')
    
    while rclpy.ok():
        
        curr_pos = [start[0], start[1]]
        end_pos =  [x_target, y_target]

        rclpy.spin_once(mpc_turtle_node)
        #call out mpc solver and return controls and states
        current_postion = mpc_turtle_node.current_position
        euler_angles = mpc_turtle_node.orientation_euler
        
        psi = (euler_angles[2] + (2*np.pi) ) % (2*np.pi);
        # psi = euler_angles[2]
        start = [current_postion[0], current_postion[1], psi]


        controls, states = turtle_mpc.solve_mpc(start, goal) 
        print("size of g", turtle_mpc.OPT_variables.size())

        
        #unpack this matrix and publish the controls and trajectory
        cmd_vel = np.float(controls[0])
        ang_vel_cmd = np.float(controls[1])
              
        #publish the trajectory
        x_traj = np.array(states[0,:])
        y_traj = np.array(states[1,:])
        psi_traj = states[2,:]
        
        # mpc_turtle_node.publish_path(x_traj[0], y_traj[0],
        #                              mpc_turtle_node.mpc_traj_publisher)

        #publish the controls to send to the robot
        mpc_turtle_node.publish_cmd(cmd_vel, ang_vel_cmd)
        
        rclpy.spin_once(mpc_turtle_node)

        
    
if __name__=='__main__':
    main_mpc()    