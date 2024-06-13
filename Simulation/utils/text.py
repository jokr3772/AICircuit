# Decode results from result text file
#
# Desc:
# Cadence generae simulation results in text format.
# This script contains necessary functions for decoding results from text files.
#
# Author: Yue (Julien) Niu

import math
import numpy as np


def alter_circ_param(new_params_values, ocean_path):
    scs_file = open(ocean_path, 'r')
    lines = scs_file.readlines()
    
    # format for set variable values
    format_var = 'desVar(   \"{}\" {} )\n'

    # locate the line of circuit parameters
    for i, line in enumerate(lines):
        if 'desVar' in line:
            var = line.split('\"')[1]
            if var in new_params_values:
                if new_params_values[var] == 0: continue
                
                lines[i] = format_var.format(var, new_params_values[var])

    ocean_path_new = '/'.join(ocean_path.split('/')[0:-1]) + '/oceanScriptNew.ocn'
    ocean_file_new = open(ocean_path_new, 'w')
    ocean_file_new.writelines(lines)
            


def dec_ac_text(path):
    ac_file = open(path, 'r')
    lines = ac_file.readlines()
    
    start = False
    is_differential = False
    freq, voutp, voutn, vout, vinp, vinn = [], [], [], [], [], []
    for line in lines:
        if 'VALUE' in line:
            start = True
    
        # find the frequency point
        if start and 'freq' in line:
            freq.append(float(line.split(' ')[1]))
    
        # find Vout+ and Vout-
        if start and 'Vout+' in line:
            is_differential = True
            i, j = line.index('('), line.index(')')
            out_str = line[i+1:j].split(' ')
            voutp.append([float(out_str[0]), float(out_str[1])])

        if start and 'Vout-' in line:
            i, j = line.index('('), line.index(')')
            out_str = line[i+1:j].split(' ')
            voutn.append([float(out_str[0]), float(out_str[1])])

        # find Vin+ and Vin- (if exists)
        if start and 'Vin+' in line:
            i, j = line.index('('), line.index(')')
            out_str = line[i+1:j].split(' ')
            vinp.append([float(out_str[0]), float(out_str[1])])

        if start and 'Vin-' in line:
            i, j = line.index('('), line.index(')')
            out_str = line[i+1:j].split(' ')
            vinn.append([float(out_str[0]), float(out_str[1])])

        # find Vout if it is not differential output
        if start and 'Vout' in line and 'Vout+' not in line:
            i, j = line.index('('), line.index(')')
            out_str = line[i+1:j].split(' ')
            vout.append([float(out_str[0]), float(out_str[1])])
    
    vout_mag = []
    if is_differential:
        for voutp_i, voutn_i in zip(voutp, voutn):
            mag_i = math.sqrt((voutp_i[0] - voutn_i[0]) ** 2 + (voutp_i[1] - voutn_i[1]) ** 2)
            vout_mag.append(mag_i)
    else:
        for vout_i in vout:
            mag_i = math.sqrt(vout_i[0] ** 2 + vout_i[1] **2)
            vout_mag.append(mag_i)

    vin_mag = []
    if len(vinp) > 0:  # if Vin exits, voltage gain is the ratio between vout_mag and vin_mag
        for vinp_i, vinn_i in zip(vinp, vinn):
            mag_i = math.sqrt((vinp_i[0] - vinn_i[0]) ** 2 + (vinp_i[1] - vinn_i[1]) ** 2)
            vin_mag.append(mag_i)

    vgain = []
    if len(vinp) > 0:
        for vout_i, vin_i in zip(vout_mag, vin_mag):
            vgain.append(vout_i / vin_i)
    else:
        vgain = vout_mag

    return np.array(freq), np.array(vgain)


def dec_dc_text(path: str):
    dc_file = open(path, 'r')
    lines = dc_file.readlines()

    for line in lines:
        if '\":pwr\"' in line:
            pw_str = line.split(' ')[2]

            return float(pw_str)