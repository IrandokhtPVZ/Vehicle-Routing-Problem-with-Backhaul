# Vehicle-Routing-Problem-with-Backhaul
Vehicle Routing Problem with Backhaul (VRPB), Lagrangian relaxation algorithm, CPLEX optimization solver, Python implementation

This repository is created to publicly share the datasets and Python codes for the VRPB model proposed in:
Parviziomran, I., Mahmoudi, M., 2020. A mixed-integer programming model for the standard vehicle routing problem with backhauls.

In this repository you will find the following folders:

1. "Six Nodes" constitutes of three python codes corresponding to Section 5.1 in the foregoing reference.
2. "GJ" constitutes of 68 benchmark instances for the standard VRPB proposed by Goetschalckx and Jacobs-Blecha (1989) as well as two python codes corresponding to the Lagrangian relaxation algorithm in parallel and sequential layout (see Section 5.2).
3. "TV" constitutes of 33 benchmark instances for the standard VRPB proposed by Toth and Vigo (1997) as well as two python codes corresponding to the Lagrangian relaxation algorithm in parallel and sequential layout (see Section 5.2).
4. "Lansing transportation network" constitutes of three randomly generated datasets with 100, 250, and 500 customers/nodes as well as python code related to the Lagrangian relaxation algorithm with cluster-first, route-second layout (see Section 5.3 and Appendix B).
5. "GJ_Solution" presents our computational results as well as results reported by exact algorithms in the extant literature for GJ dataset.
6. "TV_Solution" presents our computational results as well as results reported by exact algorithms in the extant literature for TV dataset.


References
Goetschalckx, M. and Jacobs-Blecha, C., 1989. The vehicle routing problem with backhauls. European Journal of Operational Research, 42(1), pp.39-51.
Toth, P., Vigo, D., 1997. An exact algorithm for the vehicle routing problem with backhauls. Transportation Science. 31(4), 372-385.

Cite our paper as:
Parviziomran, I., Mahmoudi, M., 2020. A mixed-integer programming model for the standard vehicle routing problem with backhauls.
