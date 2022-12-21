#!/usr/bin/env python3

import casadi as ca
import math as m
import numpy as np

class MPC:
    def __init__(self, model, dt_val, N,
                    Q,R):
    
        self.model = model
        self.dt_val = dt_val 
        self.N = N 
        self.Q = Q
        self.R = R

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

    def get_bounds(self):
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

    def set_bounds(self):
        """
        REFACTOR THIS
        """

        self.lbx['U'][0,:] = v_min
        self.ubx['U'][0,:] = v_max

        self.lbx['U'][1,:] = psi_rate_min
        self.ubx['U'][1,:] = psi_rate_max

