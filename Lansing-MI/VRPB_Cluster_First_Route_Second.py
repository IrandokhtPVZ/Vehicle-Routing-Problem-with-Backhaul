# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:12:10 2020

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
This program is designed to solve the standard VRPB for randomly generated dataset from the real-world Lansing transportation network.
This program presents the Lagrangian relaxation algorithm with cluster-first route-second layout.
To implement this program following packages are required:
    CPLEX optimization solver, academic version
    docplex
    threading and queue
    time
'''
##############################################################################
# initialization
path =['Lansing_100', 'Lansing_250']#, 'Lansing_500'] #'Lansing_30',
sheet = ['50', '66', '80']
capacity = [10, 30, 50]
max_iteration = 30
Gap = 1
time_limit = 500
max_iteration_without_improvement = 3
##############################################################################
# read data
def data(path, sheet):
    print('Lansing Transportation Network', path, sheet, Q)
    global L, L_0, B, B_0
    global A, A_C
    global c, c_C
    global k
    
    data = pd.read_excel(str(path)+'.xlsx', sheet_name=str(sheet))
    distance = pd.read_excel(str(path)+'.xlsx', sheet_name='distance', index_col=0)

    # all nodes ID, including depot. 0 represents depot
    V = list(data['node_id'])

    # linehaul customers ID
    L = list(data[data['type']==1]['node_id'])
    L_0 = [2705] + L
    # backhaul customers ID
    B = list(data[data['type']==2]['node_id'])
    B_0 = B + [2705]
    
    # truck capacity and No. of avaiable trucks
    k = max(np.ceil(len(L)/Q), np.ceil(len(B)/Q))
    
    # all links, completed graph
    A = [(i,j) for i in V for j in V]
    A_C= [(i,j) for i in L for j in B_0]
    
    # Distance is computed as the shorthest path for every two nodes based on Haversine distance metric in miles and rounded to 2 decimals 
    c = {(i,j):round(distance[i][j],2) for i in V for j in V}
    for i in V:
        c[(i,i)] = float('inf')
        
    c_C= {(i,j): c[(i,j)] for i,j in A_C}
    
    # demand of all nodes is one unit, demand is zero for depot
################################################
# cluster linehaul/backhaul customers
def clustering(node, penalty):
    
    link = [(i,j) for i in node for j in node]
    
    mdl = Model('Clustering')
    
    # decision variables
    s = mdl.binary_var_dict (link,name='s')
    p = mdl.binary_var_dict (node,name = 'p')
    

    # cluster size
    mdl.add_constraints(mdl.sum(s[i,q]for i in node)<=Q*p[q] for q in node)        
    # each node is assigned to one cluster
    mdl.add_constraints(mdl.sum(s[i,q]for q in node)==1 for i in node)   
    # No. of clusters
    mdl.add_constraint(mdl.sum(p[q] for q in node)<=k)
    
    # objective function
    obj = mdl.sum(c[i,j]*s[i,j] for i,j in link)
    mdl.add_kpi(obj, 'Clustering Cost')
    penalty = mdl.sum(penalty[i]*s[i,j]for i in node for j in node if i!=j)
    mdl.add_kpi(penalty, 'Penalty')
    objective = obj + penalty
    
    mdl.minimize(objective)
    
    mdl.parameters.timelimit = time_limit
    mdl.export_as_lp()
 
    solution = mdl.solve(log_output=False)
    #mdl.report_kpis()
    
    if not solution:
        print('fail in solving, there is no feasible solution')
    else:   
        s = [a for a in link if s[a].solution_value> 0.9]
        obj = solution.objective_value
        return s
################################################
# route each cluster
def routing(node, penalty, route):
    
    node_0 = [2705] + node 
    link = [(i,j) for i in node_0 for j in node_0 if i!=j]
    cost = {(i,j): c[(i,j)] for i,j in link}
    n = len (node_0) 
    if route == 'linehaul':
        for i in node_0:
            cost[(i,2705)] = 0
    elif route == 'backhaul':
        for i in node_0:
            cost[(2705, i)] = 0
    
    mdl = Model('Routing')
    
    # decision variables
    r = mdl.binary_var_dict (link,name='r')
    u = mdl.continuous_var_dict (node,name = 'u')
    
    # traveling sales man problem
    mdl.add_constraints(mdl.sum(r[i,j]for i in node_0 if i!=j)==1 for j in node_0 )
    mdl.add_constraints(mdl.sum(r[i,j]for j in node_0 if i!=j)==1 for i in node_0 )
    # truck capacity and subtour elimination constraints
    mdl.add_constraints(u[i]-u[j]+n*r[i,j]<= n-1 for i,j in link if i!=2705 and j!=2705 and i!=j)
        
    # objective function
    obj = mdl.sum(cost[i,j]*r[i,j]for i,j in link)
    mdl.add_kpi(obj, 'Routing Cost')   
    penalty = mdl.sum(penalty[i]*r[i,j]for i in node for j in node if i!=j)
    mdl.add_kpi(penalty, 'Penalty')    
    objective = obj + penalty

    mdl.minimize(objective)
    
    mdl.parameters.timelimit = time_limit
    mdl.export_as_lp()

    solution = mdl.solve(log_output=False)
    #mdl.report_kpis()
    
    if not solution:
        print('fail in solving, there is no feasible solution')
    else:   
        r = [a for a in link if r[a].solution_value> 0.9]
        obj = solution.objective_value
        return r, obj
################################################
# solve connection problem as an assignment problem to connect linehaul to backhaul or depot
def connection(landa, beta, z_LB, obj_C):
    
    mdl = Model('Connection Assignment Problem')
    
    # decision variable
    z = mdl.binary_var_dict (A_C,name='z')
    
    # constraints
    mdl.add_constraint(mdl.sum(z[i,j]for i,j in A_C )==k)
    mdl.add_constraints(mdl.sum(z[i,j] for j in B_0)<=1 for i in L)
    mdl.add_constraints(mdl.sum(z[i,j] for i in L)<=1 for j in B)
    
    # objective function
    obj_connection = mdl.sum(c_C[i,j]*z[i,j]for i,j in A_C)
    mdl.add_kpi(obj_connection, 'Connection Cost')
    penalty_linehaul_connection = mdl.sum(landa[i]*z[i,j]for i in L for j in B_0 if i!=j)
    mdl.add_kpi(penalty_linehaul_connection, 'Linehaul-Connection Penalty')
    penalty_backhaul_connection = mdl.sum(beta[j]*z[i,j]for i in L for j in B if i!=j)
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
        z_LB.put( [a for a in A_C if z[a].solution_value> 0.9])
        obj_C.put( solution.objective_value)
################################################
# compute actual cost/objective
def objective_value(x,y,z):
    
    x = [(i,j) for i,j in x if j!=2705]
    y = [(i,j) for i,j in y if i!=2705]
    
    objective_linhaul = sum(c[i,j] for i,j in x)
    objective_backhaul = sum(c[i,j]for i,j in y)
    objective_connection = sum(c[i,j] for i,j in z)
    obj = round(objective_linhaul + objective_backhaul + objective_connection)   

    return obj
################################################
# solve lower bound problem
# solve linehaul problem as an open TSP in which ci2705=0 for i in linehaul customers' set
def linehaul(landa, x_LB, obj_L):
    
    x_Lin = list()
    obj_Lin=0
    
    # cluster & route linehaul
    cluster_L = clustering(L, landa)
    linehaul = pd.DataFrame()
    linehaul['node_id'] = L
    for i,j in cluster_L:
        linehaul.loc[linehaul['node_id']==i, 'cluster']=j

    for i in list((linehaul['cluster'].unique())):
        
        mini_L = list(linehaul[linehaul['cluster']==i]['node_id'])
        x, obj = routing(mini_L, landa, 'linehaul')
        x_Lin += x
        obj_Lin += obj 
    
    x_LB.put(x_Lin)
    obj_L.put(obj_Lin)
# solve backhaul problem as an open TSP in which c2705j=0 for j in backhaul customers' set      
def backhaul(beta, y_LB, obj_B):
    
    y_Back = list()
    obj_Back=0
      
    # cluster & route backhaul    
    cluster_B = clustering(B, beta)
    beckhaul = pd.DataFrame()
    beckhaul['node_id'] = B
    for i,j in cluster_B:
        beckhaul.loc[beckhaul['node_id']==i, 'cluster']=j

    for i in list((beckhaul['cluster'].unique())):
        mini_B = list(beckhaul[beckhaul['cluster']==i]['node_id'])
        y, obj = routing(mini_B, beta, 'backhaul')
        y_Back += y
        obj_Back += obj  

    y_LB.put(y_Back)
    obj_B.put(obj_Back)
################################################
# solve upper bound problem to get feasible solution
def upper_bound(x_LB, y_LB):

    x_tail = list(set([i[0] for i in x_LB if i[1]==2705]))
    y_head = list(set([i[1] for i in y_LB if i[0]==2705]))
    y_head_0 = [2705]+y_head

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
        x_UB = [(i,j) for i,j in x_LB if j!=2705]
        y_UB = [(i,j) for i,j in y_LB if i!=2705]
        obj = objective_value(x_UB, y_UB, z_UB)
        return x_UB, y_UB, z_UB, obj
################################################
################################################
def update_multiplier(theta ,global_UB,local_LB, landa,beta, x, y, z):
    
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
for p in path:
    for s in sheet:
        for c in capacity:
            global Q
            Q = c
            # read data
            data(p,s)
           
            # algorithm initialization
            global_UB = float('inf')
            global_LB = float('-inf')
            
            theta = 0.4
            iter = 0
            stepsize_controller = 0
            
            landa = {i:0 for i in L}
            beta = {i:0 for i in B}
            
            start = time.clock()
            while iter<=max_iteration:
                
                # solve clustering & LR decomposed problems to get lower bound solution using multithreading 
                # multithreading setting
                r_L=queue.Queue()
                r_B=queue.Queue()
                r_C=queue.Queue()
                
                out_L=queue.Queue()
                out_B=queue.Queue()
                out_C=queue.Queue()
                
                # threads solving linehaul, backhaul, and connection phases in parallel to obtain a lower bound solution
                t1 = threading.Thread(target=linehaul, args=(landa, r_L, out_L)) 
                t2 = threading.Thread(target=backhaul, args=(beta, r_B, out_B))
                t3 = threading.Thread(target=connection, args=(landa, beta, r_C, out_C))
                
                t1.start()
                t2.start()
                t3.start()
                
                t1.join()
                t2.join()
                t3.join()
                
                # collect lower bound solution
                obj_L = out_L.get()
                obj_B = out_B.get()
                obj_C = out_C.get()
        
                x_LB = r_L.get()
                y_LB = r_B.get()
                z_LB = r_C.get()
                
                # compute lower bound objective value
                penalty_L = round(sum(landa[i] for i in L))
                penalty_B = round(sum(beta[i] for i in B))
                local_LB = round(obj_L + obj_B + obj_C - penalty_L - penalty_B)
                
                
                if local_LB > global_LB:
                    global_LB = local_LB
                    stepsize_controller = 0    
                else:
                    stepsize_controller += 1
                
                # solve the upper bound problem to obtain a feasible solution
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
                    # update theta if the slution did not improve after certain No. of iterations
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
               
                #if norme == 0:
                 #   print('both multipliers are zero, the LB solution is feasible')
                  #  break
                #else:
                    iter+=1
            
            stop = time.clock()
            # compute CPU running time in second
            cpu_time = round(stop - start,2)
            
            # collect solution
            solution.append(dict(zip(['instance','L+B','L','B','Q','k', 'global LB', 'global UB', 'gap', 'CPU running time (Second)', 'x', 'z', 'y'],
                                     [str(p)+'_'+str(s), len(L)+len(B), len(L), len(B), Q, k, global_LB, global_UB, opt_gap, cpu_time, x_best, z_best, y_best])))        
##############################################################################    
Solution = pd.DataFrame(solution)
Solution.to_csv('Solution_Cluster_First_Route_Second_'+str(datetime.now().strftime('%Y-%m-%d_%H-%M'))+'.csv', index=False)
##############################################################################

