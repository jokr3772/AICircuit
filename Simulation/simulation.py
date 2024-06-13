# This is the top entry of the codebase

from Simulation.simulator import Simulator
from Simulation.param import get_circ_params, get_circ_path, get_dataset_path
from Utils import sim_result
from args import args

circuit_params = get_circ_params(args.circuit)
circuit_path, circuit_path_docker = get_circ_path(args.circuit)
params_path = get_dataset_path(args.circuit, args.model)

simulator = Simulator(circuit_path, circuit_path_docker, circuit_params, params_path)

simulator.run_all(n=args.npoints)

# print simulation results
sim_result.calc_hist(simulator.sim_results)

#for item in simulator.sim_results:
#    print(item)
