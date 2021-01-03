# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:47:57 2020

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
from datetime import datetime
import threading, queue
from docplex.mp.model import Model
##############################################################################
'''
This program is designed to solve the standard VRPB benchmark instances proposed by Toth and Vigo (1997).
This program presents the Lagrangian relaxation algorithm with a sequential layout.
To implement this program following packages are required:
    CPLEX optimization solver, academic version
    docplex
    threading and queue
    time
'''
##############################################################################
TV=['eil22_50', 'eil22_66', 'eil22_80', 'eil23_50', 'eil23_66', 'eil23_80',
    'eil30_50', 'eil30_66', 'eil30_80', 'eil33_50', 'eil33_66', 'eil33_80',
    'eil51_50', 'eil51_66', 'eil51_80', 
    'eilA76_50', 'eilA76_66', ###########'eilA76_80',
    'eilB76_50', 'eilB76_66', 'eilB76_80', 'eilC76_50', 'eilC76_66', 'eilC76_80',
    'eilD76_50', 'eilD76_66', 'eilD76_80', 'eilA101_50', 'eilA101_66', #######'eilA101_80',
    'eilB101_50']#, ###################'eilB101_66', 'eilB101_80']
max_iteration=30
max_iteration_without_improvement = 3
Gap=1
time_limit = 500
##############################################################################
##############################################################################
# read data
def data(path):
    print('Toth, P. and Vigo, D., 1997. An exact algorithm for the vehicle routing problem with backhauls. Transportation science, 31(4), pp.372-385. instance', path)
    global L, L_0, B, B_0
    global A_L, A_B, A_C
    global c, c_L, c_B, c_C
    global d_L, d_B
    global Q, k
    
    data = pd.read_csv(str(path)+'.csv')
    
    # all nodes ID, including depot. 1 represents depot in this dataset
    V = list(data['node_id'])
    
    # linehaul customers ID
    L = list(data[data['type']==0]['node_id'])
    L_0 = [1] + L
    # backhaul customers ID
    B = list(data[data['type']==1]['node_id'])
    B_0 = B + [1]
    
    # truck capacity and No. of avaiable trucks
    Q = data['Q'].iloc[0]    
    k = data['k'].iloc[0]
    
    # all links, completed graph
    A = [(i,j) for i in V for j in V]
    A_L = [(i,j) for i in L_0 for j in L_0 if i!=j]
    A_B = [(i,j) for i in B_0 for j in B_0 if i!=j]
    A_C= [(i,j) for i in L for j in B_0]
    
    # Euclidean distance of all nodes, based on Toth and Vigo it is rounded to closest integer
    c = {(i,j):round(np.sqrt((data[data['node_id']==j]['x'].iloc[0]-data[data['node_id']==i]['x'].iloc[0])**2+
                           (data[data['node_id']==j]['y'].iloc[0]-data[data['node_id']==i]['y'].iloc[0])**2)) for i,j in A}
    
    for i in V:
        c[(i,i)] = 0
        
    c_L= {(i,j): c[(i,j)] for i,j in A_L}
    for i in L_0:
        c_L[(i,1)] = 0

    c_B= {(i,j): c[(i,j)] for i,j in A_B}
    for i in B_0:
        c_B[(1,i)] = 0
    
    c_C= {(i,j): c[(i,j)] for i,j in A_C}
    
    # demand of all nodes, demand is zero for depot
    d = {i: data[data['node_id']==i]['demand'].iloc[0] for i in V} 
    d_L = {i: d[i] for i in L}
    d_B = {i: d[i] for i in B}
################################################
# solve linehaul problem as an open VRP in which ci1=0 for i in linehaul customers' set
def linehaul(landa, x_LB, obj):
    
    mdl = Model('Linehaul Probplem')
    
    # decision variables
    x = mdl.binary_var_dict (A_L,name='x')
    u = mdl.continuous_var_dict (L,name = 'u')       
    
    # out/inbounds constraints
    mdl.add_constraint(mdl.sum(x[1,j]for j in L)==k)
    mdl.add_constraint(mdl.sum(x[i,1]for i in L)==k) 
    #degree constraints
    mdl.add_constraints(mdl.sum(x[i,j]for i in L_0 if i!=j)==1 for j in L )
    mdl.add_constraints(mdl.sum(x[i,j]for j in L_0 if i!=j)==1 for i in L )
    # truck capacity and subtour elimination constraints
    mdl.add_constraints(u[i]-u[j]+Q*x[i,j]<= Q-d_L[j] for i,j in A_L if i!=1 and j!=1 and i!=j)
    
    # objective function
    obj_linhaul = mdl.sum(c_L[i,j]*x[i,j]for i,j in A_L)
    mdl.add_kpi(obj_linhaul, 'Linehaul Cost')   
    penalty_linehaul = mdl.sum(landa[i]*x[i,j]for i in L for j in L if i!=j)
    mdl.add_kpi(penalty_linehaul, 'Linehaul Penalty')    
    objective = obj_linhaul + penalty_linehaul

    mdl.minimize(objective)
    
    mdl.parameters.timelimit = time_limit
    mdl.export_as_lp()
 
    solution = mdl.solve(log_output=False)
    #mdl.report_kpis()
    
    if not solution:
        print('fail in solving, there is no feasible solution')
    else:   
        x_LB.put( [a for a in A_L if x[a].solution_value> 0.9])
        obj.put( solution.objective_value)
################################################
# solve backhaul problem as an open VRP in which c1j=0 for j in backhaul customers' set
def backhaul(beta, y_LB, obj):

    mdl = Model('Backhaul Probplem')
    
    # decision variables
    y = mdl.binary_var_dict (A_B,name='y')
    w = mdl.continuous_var_dict (B,name = 'w')
   
    # out/inbounds constraints
    mdl.add_constraint(mdl.sum(y[1,j]for j in B)<=k) 
    mdl.add_constraint(mdl.sum(y[i,1]for i in B)<=k)
    #degree constraints
    mdl.add_constraints(mdl.sum(y[i,j]for i in B_0 if i!=j)==1 for j in B ) ####extra constraint to obtain valid backhaul tour
    mdl.add_constraints(mdl.sum(y[i,j]for j in B_0 if i!=j)==1 for i in B )      
    # truck capacity and subtour elimination constraints
    mdl.add_constraints(w[i]-w[j]+Q*y[i,j]<= Q-d_B[j] for i,j in A_B if i!=1 and j!=1 and i!=j)
    
    # objective function
    obj_backhaul = mdl.sum(c_B[i,j]*y[i,j]for i,j in A_B)
    mdl.add_kpi(obj_backhaul, 'Backhaul Cost')   
    penalty_backhaul = mdl.sum(beta[j]*y[i,j]for i in B for j in B if i!=j)
    mdl.add_kpi(penalty_backhaul, 'Backhaul Penalty')            
    objective = obj_backhaul + penalty_backhaul

    mdl.minimize(objective)
    
    mdl.parameters.timelimit = time_limit
    mdl.export_as_lp()
     
    solution = mdl.solve(log_output=False)
    #mdl.report_kpis()
    if not solution:
        print('fail in solving, there is no feasible solution')
    else:
        y_LB.put( [a for a in A_B if y[a].solution_value> 0.9])
        obj.put(solution.objective_value) 
################################################
# solve connection problem as an assignment problem to connect linehaul routes to backhaul routes or to the depot
def connection(mini_L, mini_B, landa, beta, z_LB, obj):
    mini_B_0 = mini_B + [1]
    link = [(i,j) for i in mini_L for j in mini_B_0]
    
    mdl = Model('Connection Problem')
    
    # decision variable
    z = mdl.binary_var_dict (link,name='z')
    
    # constraints
    mdl.add_constraint(mdl.sum(z[i,j]for i,j in link )==k)
    mdl.add_constraints(mdl.sum(z[i,j] for j in mini_B_0)<=1 for i in mini_L)
    mdl.add_constraints(mdl.sum(z[i,j] for i in mini_L)<=1 for j in mini_B)
    
    # objective function
    obj_connection = mdl.sum(c[i,j]*z[i,j]for i,j in link)
    mdl.add_kpi(obj_connection, 'Connection Cost')
    penalty_linehaul_connection = mdl.sum(landa[i]*z[i,j]for i in mini_L for j in mini_B_0 if i!=j)
    mdl.add_kpi(penalty_linehaul_connection, 'Linehaul-Connection Penalty')
    penalty_backhaul_connection = mdl.sum(beta[j]*z[i,j]for i in mini_L for j in mini_B if i!=j)
    mdl.add_kpi(penalty_backhaul_connection, 'Backhaul-Connection Penalty')
    objective = obj_connection + penalty_linehaul_connection + penalty_backhaul_connection

    mdl.minimize(objective)
    
    mdl.parameters.timelimit = time_limit
    mdl.export_as_lp()
     
    solution =mdl.solve(log_output=False)
    #mdl.report_kpis()
    
    if not solution:
        print('fail in solving, there is no feasible solution')
    else:
       z_LB.put( [a for a in link if z[a].solution_value> 0.9])
       obj.put(solution.objective_value) 
################################################
# compute actual cost/objective
def objective_value(x,y,z):
    
    x = [(i,j) for i,j in x if j!=1]
    y = [(i,j) for i,j in y if i!=1]
    
    objective_linhaul = sum(c[i,j] for i,j in x)
    objective_backhaul = sum(c[i,j]for i,j in y)
    objective_connection = sum(c[i,j] for i,j in z)    
    obj = objective_linhaul + objective_backhaul + objective_connection   
    return obj
################################################
# solve upper bound problem to get feasible solution
def upper_bound(x_LB, y_LB):

    x_tail = list(set([i[0] for i in x_LB if i[1]==1]))
    y_head = list(set([i[1] for i in y_LB if i[0]==1]))
    y_head_0 = [1]+y_head
    
    A_UB = [(i,j) for i in x_tail for j in y_head_0]
    c_UB = {(i,j):c[i,j] for i in L for j in B_0} 
    
    mdl = Model('Upper Bound')
    
    # decision variable
    z = mdl.binary_var_dict (A_UB,name='z')
    
    # constraints
    mdl.add_constraint(mdl.sum(z[i,j]for i,j in A_UB )==k)
    mdl.add_constraints(mdl.sum(z[i,j]for i in x_tail )==1 for j in y_head)
    mdl.add_constraints(mdl.sum(z[i,j]for j in y_head_0 )==1 for i in x_tail)
    
    # objective function
    obj = mdl.sum(c_UB[i,j]*z[i,j]for i,j in A_UB)
    mdl.add_kpi(obj, 'Upper Bound Cost')
    
    objective = obj

    mdl.minimize(objective)
    
    mdl.parameters.timelimit = time_limit
    mdl.export_as_lp()
    
    solution =mdl.solve(log_output=False)
    #mdl.report_kpis()
    if not solution:
        print('fail in solving, there is no feasible solution')
    
    else:
        z_UB = [a for a in A_UB if z[a].solution_value> 0.9]
        x_UB = [(i,j) for i,j in x_LB if j!=1]
        y_UB = [(i,j) for i,j in y_LB if i!=1]
        obj = objective_value(x_UB, y_UB, z_UB)
        return x_UB, y_UB, z_UB, obj
################################################
################################################
# update Lagrangian multipliers
def update_multiplier(theta, global_UB, local_LB, landa, beta, x, y, z):
    
    # determining the step size and updating the multiplier
    # for linehaul
    residual_L = {i:sum(x[i,j] for j in L)+sum(z[i,j] for j in B_0)-1 for i in L if i!=j}
    norme_L = sum(residual_L[i]**2 for i in L)
    
    if norme_L == 0:
        landa = landa
    else:
        t_L = round(theta * (global_UB - local_LB) / sum(residual_L[i]**2 for i in L),2)
        landa = {i:landa[i]+t_L*residual_L[i] for i in L}
    
    # for backhaul     
    residual_B = {j:sum(y[i,j] for i in B)+sum(z[i,j] for i in L)-1 for j in B if i!=j}
    norme_B = sum(residual_B[i]**2 for i in B)
    
    if norme_B == 0:
        beta = beta
    else:
        t_B = round(theta * (global_UB - local_LB) / sum(residual_B[i]**2 for i in B),2)
        beta = {i:beta[i]+t_B*residual_B[i] for i in B}
    
    norme = norme_L + norme_B    
    return  norme, landa, beta
##############################################################################
##############################################################################
solution = list()
# main function
for path in TV:
    # read data
    data(str(path))
    
    # algorithm initialization
    global_UB = float('inf')
    global_LB = float('-inf')
    
    theta = 0.4
    iter = 0
    stepsize_controller = 0
    
    landa = {i:0 for i in L}
    beta = {i:0 for i in B}
    
    start = time.clock()
    
    # compute initial solution
    r_B = queue.Queue()
    out_B = queue.Queue()
    t2 = threading.Thread(target=backhaul, args=(beta, r_B, out_B))
    t2.start()
    t2.join()
    obj_B = out_B.get()
    y_LB = r_B.get()
    B_head = list(set([i[1] for i in y_LB if i[0]==1]))
    
    iter = 1
    
    while iter<=max_iteration:
        if iter%2 == 1:
            # solve linehaul and connection sub-problems concurrenlty, while the tails of assignment problem are fixed
            r_L = queue.Queue()
            r_C = queue.Queue()
            
            out_L = queue.Queue()
            out_C = queue.Queue()
            
            t1 = threading.Thread(target=linehaul, args=(landa, r_L, out_L)) 
            t3 = threading.Thread(target=connection, args=(L, B_head, landa, beta, r_C, out_C))
        
            t1.start()
            t3.start()
            
            t1.join()
            t3.join()
            
            obj_L = out_L.get()
            obj_C = out_C.get()
    
            x_LB = r_L.get()
            z_LB = r_C.get()
            L_tail = list(set([i[0] for i in x_LB if i[1]==1]))
            
        else:
            # solve backhaul  and connection sub-problems concurrenlty, while the heads of assignment problem are fixed
            r_B = queue.Queue()
            r_C = queue.Queue()
            
            out_B = queue.Queue()
            out_C = queue.Queue()
            
            t2 = threading.Thread(target=backhaul, args=(beta, r_B, out_B))
            t3 = threading.Thread(target=connection, args=(L_tail, B, landa, beta, r_C, out_C))
        
            t2.start()
            t3.start()
            
            t2.join()
            t3.join()
            
            obj_B = out_B.get()
            obj_C = out_C.get()
    
            y_LB = r_B.get()
            z_LB = r_C.get()
            B_head = list(set([i[1] for i in y_LB if i[0]==1]))
        
        # compute lower bound objective value           
        penalty_L = round(sum(landa[i] for i in L))
        penalty_B = round(sum(beta[i] for i in B))
        local_LB = round(obj_L + obj_B + obj_C - penalty_L - penalty_B) 

        if local_LB > global_LB:
            global_LB = local_LB
            stepsize_controller = 0
        else:
            stepsize_controller += 1
    
        # solve upper bound problem to obtain feasible solution based on lower bound solution
        x_UB, y_UB, z_UB , local_UB= upper_bound(x_LB, y_LB)  
    
        if local_UB < global_UB:
          global_UB = local_UB
          x_best = x_UB
          y_best = y_UB
          z_best = z_UB
    
        # compute the optimality gap, if it meets the algorithm terminates, update multipliers, otherwise.
        opt_gap = round(100*(global_UB - global_LB) / global_UB, 2)
       
        if opt_gap <= Gap:
            break
        else:
            # update theta if the solution did not improve after a certain No. of iterations
            if stepsize_controller == max_iteration_without_improvement:
                theta = round(theta/2, 4)
                stepsize_controller = 0 
            
            # update multipliers
            x_D = {(i,j):0 for i in L_0 for j in L}
            y_D = {(i,j):0 for i in B for j in B_0}
            z_D = {(i,j):0 for i in L for j in B_0}
            for (i,j) in x_LB:
                x_D[i,j] = 1
            for (i,j) in y_LB:
                y_D[i,j] = 1
            for (i,j) in z_LB:
                z_D[i,j] = 1
        
            norme, landa, beta = update_multiplier(theta, global_UB, local_LB, landa, beta, x_D, y_D, z_D)
            
        if norme == 0:
            print('both multipliers are zero, the LB solution is feasible')
            break
        else:
            iter+=1
    
    stop = time.clock()
    # compute CPU running time in second
    cpu_time = round(stop - start,2)
    
    # collect solution
    solution.append(dict(zip(['instance','L+B','L','B','Q','k', 'global LB', 'global UB', 'gap', 'CPU running time (Second)'],
                             [path, len(L)+len(B), len(L), len(B), Q, k, global_LB, global_UB, opt_gap, cpu_time])))        
##############################################################################    
Solution = pd.DataFrame(solution)
Solution.to_csv('Solution_Sequential_TV'+str(datetime.now().strftime('%Y-%m-%d_%H-%M'))+'.csv', index=False)
##############################################################################