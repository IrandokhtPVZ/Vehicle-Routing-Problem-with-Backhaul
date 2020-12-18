# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:31:52 2020

@author: Irandokht
"""

import numpy as np
np.random.seed(1000)
import pandas as pd
import random
random.seed(1000)
import warnings
warnings.filterwarnings('ignore')
import time
from docplex.mp.model import Model
##############################################################################
'''
This dataset corresponds to our small-scale dataset with 6 nodes including depot
and 5 customers/nodes in which distance between customer i and j is asymmetric.

We use the CPLEX optimization solver, the academic version, to solve this problem.
'''
##############################################################################
# function that solves VRPB
def CVRPB():

    mdl = Model('CVRPB')

    # linhaul decision Variables
    x = mdl.binary_var_dict (A_L,name='x')
    u = mdl.continuous_var_dict (L, name = 'u')
    
    #bakchaul decision variables
    y = mdl.binary_var_dict (A_B,name='y')
    w = mdl.continuous_var_dict (B,name = 'w')
    
    #connection decision variables
    z = mdl.binary_var_dict (A_C,name='z')
    #####################################
    ##### linhaul constraints
    #truck constraint
    mdl.add_constraint(mdl.sum(x[0,j]for j in L)==k )
    #degree constraints
    mdl.add_constraints(mdl.sum(x[i,j]for i in L_0 if i!=j)==1 for j in L )     
    #truck capacity 
    mdl.add_constraints(u[i]-u[j]+Q*x[i,j]<= Q-d_L[j] for i,j in A_L if i!=0 and j!=0)
        
    ##### bakchaul constraints
    #truck constraints
    mdl.add_constraint(mdl.sum(y[i,0]for i in B)+mdl.sum(z[i,0] for i in L)==k )
    #mdl.add_constraint(mdl.sum(y[i,0]for i in B)<=k )
    #degree constraints
    mdl.add_constraints(mdl.sum(y[i,j]for j in B_0 if j!=i)==1 for i in B)    
    #truck capacity
    mdl.add_constraints(w[i]-w[j]+Q*y[i,j]<=Q-d_B[j] for i,j in A_B if i!=0 and j!=0)
           
    ####### connection constraints
    #truck constraints
    mdl.add_constraint(mdl.sum(z[i,j]for i,j in A_C )==k)
    #degree constraints
    mdl.add_constraints(mdl.sum(x[i,j] for j in L if j!=i )+ mdl.sum(z[i,j] for j in B_0 )==1 for i in L)
    mdl.add_constraints(mdl.sum(y[i,j] for i in B if i!=j )+ mdl.sum(z[i,j] for i in L )==1 for j in B)
    ###################################
    ####objective functions
    obj_Linhaul = mdl.sum(c[i,j]*x[i,j]for i,j in A_L)
    mdl.add_kpi(obj_Linhaul, 'Linehaul Cost')
    
    obj_Backhaul = mdl.sum(c[i,j]*y[i,j]for i,j in A_B)
    mdl.add_kpi(obj_Backhaul, 'Backhaul Cost')
    
    obj_Connection = mdl.sum(c[i,j]*z[i,j]for i,j in A_C)
    mdl.add_kpi(obj_Connection, 'Connection Cost')
    
    objective = obj_Linhaul+obj_Backhaul+obj_Connection
    mdl.add_kpi(objective, 'Total Cost')
    #######################################
    #solving model
    mdl.minimize(objective)
    #mdl.parameters.timelimit=5 
    solution =mdl.solve(log_output=False) #true if you need to see the steps of slover
    #mdl.report_kpis()
    if not solution:
        print('fail in solving, there is no feasible solution')
    #mdl.export_as_lp()
    ########################################
    #Collect optimal solution
    x_opt=[a for a in A_L if x[a].solution_value> 0.9]
    y_opt=[a for a in A_B if y[a].solution_value> 0.9]
    z_opt=[a for a in A_C if z[a].solution_value> 0.9]
    obj_opt=round(solution.objective_value,2)
    
    return obj_opt, x_opt, y_opt, z_opt
##############################################################################
global V, L, L_0, B, B_0
global A, A_L, A_B, A_C
global c
global d_L, d_B
global Q, k

#### step 1: preparing data
# vehicles capacity and number of available vehicles at the depot
Q = 2
k = 2

# all nodes, where 0 states the depot. all links and symetric distance matrix
V = [0, 1, 2, 3, 4, 5]
A = [(i,j) for i in V for j in V]
c = {(0,0):0,   (0,1):2,    (0,2):4,    (0,3):7,   (0,4):1.5,   (0,5):2.5,
     (1,0):3,   (1,1):0,    (1,2):2,    (1,3):4,   (1,4):2.5,   (1,5):1.5,
     (2,0):5,   (2,1):3,    (2,2):0,    (2,3):5,   (2,4):3,     (2,5):1.5,
     (3,0):3,   (3,1):5,    (3,2):2,    (3,3):0,   (3,4):3,     (3,5):1.5,
     (4,0):2.5, (4,1):1.5,  (4,2):4,    (4,3):2.5, (4,4):0,     (4,5):1,
     (5,0):3,   (5,1):2.5,  (5,2):2.5,  (5,3):3,   (5,4):1.5,   (5,5):0}

# all nodes and links related to linehaul customers
L = [1, 2, 3]
L_0 = [0] +L
d_L = {1:1, 2:1, 3:1}
A_L = [(i,j) for i in L_0 for j in L if i!=j]

# all nodes and links related to backhaul customers
B = [4, 5]
B_0 = B + [0]
d_B = {4:1, 5:1}
A_B = [(i,j) for i in B for j in B_0 if i!=j]

# all links related to connecting linehaul routes to backhaul routes or to the depot
A_C= [(i,j) for i in L for j in B_0]

#### step 2:  solve the problem
start = time.clock()
obj_opt, x_opt, y_opt, z_opt = CVRPB()
stop = time.clock()
CPU_running_time = round(stop - start,2)

#### step 3:  print the solution
print ('The solution for VRPB with asymmetric distance for small-scale dataset is:',
       '\nObjective value = ', obj_opt, '\nCPU running time in second = ', CPU_running_time)


