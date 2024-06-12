# Simulation class given analog circuit parameters
#
# Author: Yue (Julien) Niu

import csv
import math
import subprocess
from utils import bw, text


class Simulator:
    
    def __init__(self, circuit_path, circuit_path_docker, circuit_params, params_path):
        """
        :param circuit_path: circuit path in a host system
        :param circuit_path_docker: circuit path inside a docker container
        :param circuit_params: defined circuit parameters
        :param params_path: circuit parameter value path
        """
        self.circuit_path = circuit_path
        self.circuit_def = circuit_path + '/oceanScript.ocn'
        self.circuit_params = circuit_params
        self.params_path = params_path
        
        # construct simulation command with docker
        self.circuit_path_docker = circuit_path_docker
        self.docker_cmd = 'docker exec --user=julien rlinux8-1 /bin/tcsh -c'
        
        # store simulation results
        self.sim_results = []
        
    def run_sim(self):
        """Start a simulation
        Note that all simulation parameters are defined in input.scs file.
        If additional simulation functions need to be added, you should directly edit input.scs file.
        """
        
        bash_cmds = f'\"cd {self.circuit_path_docker}; ocean -nograph -replay oceanScriptNew.ocn"'
        sim_cmd = f'{self.docker_cmd} {bash_cmds}'
        # print(sim_cmd)
        ret = subprocess.call(sim_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        if ret:
            print('[ERROR] cmd is not properly executed!!!')
    
    def decode_results(self, sim_functions: list):
        """Decode simulation results from output saved files
        The simulation results are saved in a special format, we need first convert it to a text format,
        and then extract the data we need
        """
        
        cur_sim_result = {}
        
        for func in sim_functions:
            if 'ac' in func:
                # convert results to text
                cmd = f'{self.docker_cmd} \"cd {self.circuit_path_docker}; psf -i input_new.raw/ac.ac -o ac.txt\"'
                ret = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                
                if ret:
                    print('[ERROR] cmd is not properly executed!!!')
                
                # extract data needed
                text_ac_path = self.circuit_path + '/ac.txt'
                freq, vout_mag = text.dec_ac_text(text_ac_path)
                
                # cur_sim_result['ac'] = [freq, vout_mag]
                cur_sim_result['Bandwidth/Hz'] = bw.bw_by_iterp(vout_mag, freq)
                cur_sim_result['VoltageGain/dB'] = 20 * math.log10(vout_mag[0])
                cur_sim_result['Error_Bandwidth/Hz'] = 0.0
                cur_sim_result['Error_VoltageGain/dB'] = 0.0
                
            if 'dcOp' in func:
                cmd = f'{self.docker_cmd} \"cd {self.circuit_path_docker}; psf -i input_new.raw/dcOp.dc -o dcOp.txt\"'
                ret = subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

                if ret:
                    print('[ERROR] cmd is not properly executed!!!')

                text_dc_path = self.circuit_path + '/dcOp.txt'
                pw = text.dec_dc_text(text_dc_path)

                cur_sim_result['dcPowerConsumption/W'] = pw
                cur_sim_result['Error_dcPowerConsumption/W'] = 0.0

        self.sim_results.append(cur_sim_result)
                
    def get_results(self):
        """Simply extract result from results.txt generated from ocean
        """
        cur_sim_result = {}
        result_path = self.circuit_path + '/results.txt'
        result_file = open(result_path, 'r')
        lines = result_file.readlines()
        for line in lines:
            if ':' in line:
                try:
                    metric, value = line.split(':')[0], float(line.split(':')[1])
                except ValueError:
                    return
                else:
                    cur_sim_result[metric] = value
                    cur_sim_result['Error_'+metric] = 0
        self.sim_results.append(cur_sim_result)

    def run_all(self, n=10, display=True):
        """Run all simulations by sweeping paramters defined in the .csv file
        """
        with open(self.params_path, mode='r') as param_file:
            param_dict = csv.DictReader(param_file)
            for i, line in enumerate(param_dict):
                for p in self.circuit_params:
                    if p not in line: continue
                    
                    self.circuit_params[p] = line[p]
            
                # edit circuit parameters in .scs file
                text.alter_circ_param(self.circuit_params, self.circuit_def)

                # start simulation
                self.run_sim()
                
                # get simulation results
                self.get_results()

                # calculate relative error
                self.calc_error(line)

                if display and i > 10 and i % 100 == 0:
                    print('{} points simulated.'.format(i+1))

                if n != -1 and i == n: break

    def calc_error(self, perf_ref):
        """Calculate error compared to reference values
        :param perf_ref: reference performance
        """
        for key in self.sim_results[-1]:
            if 'Error' not in key:  # only check actual values, not error
                val_actual = self.sim_results[-1][key]
                val_ref = float(perf_ref[key])

                if 'VoltageGain' in key:
                    val_actual = 10 ** (val_actual / 20)
                    val_ref = 10 ** (val_ref / 20)
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)
                elif 'ConversionGain' in key or 'PowerGain' in key or 'NoiseFigure' in key or 'S11' in key or 'S22' in key:
                    val_actual = 10 ** (val_actual / 10)
                    val_ref = 10 ** (val_ref / 10)
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)
                else:
                    rel_error = abs(val_ref - val_actual) / abs(val_ref)

                self.sim_results[-1]['Error_' + key] = rel_error

