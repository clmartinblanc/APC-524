""" An added attribute of phase (dictionary) to the Case object. "t" is when the phase is installed, "index" is the how many array elements we shift based on
    the hilbert transform from the simulation. "index_theo" is the same but based on theoretical wave phase velocity.
    Usage:
        quantity = np.roll(quantity, -(idx), axis=0)
    where axis=0 corresponds to the x axis.    
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from defs import Case, Interface2D
from prepare import load_object, save_object

def create_new_phase(case, tsimu, PRE=False, ADDITIVE=False, phase_old=None):
    case.phase = {"t":[], "idx":[], "idx_theo":[], "eta":[]}
    # Stationary waves. All the same phase.
    if PRE == True:
        interface = Interface2D(L0 = case.L0, N = case.N, 
                                path = case.path, t=None, PRUNING=True, pruningz=case.h+0.4/case.k, filename=case.path+'eta/eta_pre') 
        for (j,t) in enumerate(tsimu):
            case.phase['t'].append(t-case.tstart)
            case.phase['idx'].append(0)  
            case.phase['eta'].append(interface.eta)  
    # Moving waves. Read phase from interface file.
    else:      
        if ADDITIVE == True: 
            #If additively adding time
            t_old = np.array(phase_old['t'])
            for (j,t) in tqdm(enumerate(tsimu)):
                case.phase['t'].append(t-case.tstart)
                i = np.where(np.isclose(t_old, t-case.tstart))[0]
                print(i)
                if len(i) != 0:
                    print('Reuse %g' %(t_old[i[0]]+case.tstart))
                    case.phase['idx'].append(phase_old['idx'][i[0]])
                    case.phase['idx_theo'].append(phase_old['idx_theo'][i[0]])
                    case.phase['eta'].append(phase_old['eta'][i[0]])
                else:
                    print ('Additional t = %g' %t)
                    interface = Interface2D(L0 = case.L0, N = case.N, 
                                            path = case.path, pre='eta/eta_loc_t', t = t, PRUNING=True, pruningz=case.h+0.4/case.k)    
                    # TODO: append other field in interface too
                    case.phase['idx'].append(interface.idx)
                    idx_theo = int(round(case.wave.c*(t-case.tstart)/(2*np.pi/case.k)*int(case.N/case.k)) % int(case.N/case.k))
                    case.phase['idx_theo'].append(idx_theo)
                    case.phase['eta'].append(interface.eta) 
        else: 
            # If creating from scratch 
            for (j,t) in tqdm(enumerate(tsimu)):
                case.phase['t'].append(t-case.tstart)
                interface = Interface2D(L0 = case.L0, N = case.N, 
                                        path = case.path, pre='eta/eta_loc_t', t = t, PRUNING=True, pruningz=case.h+0.4/case.k)    
                case.phase['idx'].append(interface.idx)
                idx_theo = int(round(case.wave.c*(t-case.tstart)/(2*np.pi/case.k)*int(case.N/case.k)) % int(case.N/case.k))
                case.phase['idx_theo'].append(idx_theo)
                case.phase['eta'].append(interface.eta)
            
    # Save the new phase dictionary
    picklename = case.path + 'eta/' + 'phase_info' +'.pkl'
    save_object(case.phase, picklename)

def extract_phase (case, tsimu, PRE=False):      
    picklename = case.path + 'eta/' + 'phase_info' +'.pkl'
    exists = os.path.exists(picklename)
    # If the pickle is there read in the pickles, and check that time array agrees.
    if exists:
        case.phase = load_object(picklename)
        print('pickle restored!')
        if len(np.array(case.phase['t'])) != len(tsimu):
            print('But not the same time array!')
            phase_old = case.phase.copy()
            create_new_phase (case, tsimu, PRE=PRE, ADDITIVE=True, phase_old=phase_old)
    # Else initiate the array from scratch 
    else:
        create_new_phase (case, tsimu, PRE=PRE, ADDITIVE=False)
