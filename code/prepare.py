""" Functions related to pickle. """
import sys
import os
import numpy as np
sys.path.append('/home/jiarong/research/postprocessing/functions/')
sys.path.append('/projects/DEIKE/jiarongw/jiarongw-postprocessing/jupyter_notebook/functions/')
sys.path.append('/projects/DEIKE/jiarongw/jiarongw-postprocessing/jupyter_notebook/project_specific/windwave/')
from fio import ensemble_pickle
from tqdm import tqdm

""" Helper functions """
import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
def load_object(filename):
    with open(filename, 'rb') as input:  # Overwrites any existing file.
        obj = pickle.load(input)
    return obj

""" Helper functions (readin): 
    For the case objects, with already specified self.field_t, read in fields and store as 
    spanwise averaged time sequence of 2D array. If the pickle is available, read in the pickle;
    if not, store the pickle. (NOTICE: need to have a sub-directory of pickle_tiger/ first!)
    Output: case.ux_2D, case.uy_2D, case.f_2D (dimension time*x*z)                   
"""
def read_fields (case):
    case.ux_2D = []
    case.uy_2D = []
    case.f_2D = []

    for i in tqdm(range(0,np.size(case.field_t))):

        NSLICE = 256    
        NGRID = 512
        ux_3D = {'name':'ux', 'value':[]} # axis0 in z, axis1 in x, axis2 in y  (in the code)
        uy_3D = {'name':'uy', 'value':[]}
        f_3D = {'name':'f', 'value':[]}
        tsimu = case.field_t[i] + case.tstart
        print(tsimu)
        phasei = np.where(np.isclose(np.array(case.phase['t']), case.field_t[i]))[0][0]
        idx = case.phase['idx'][phasei]

        # Read in the fields either from pickle or from slice data
        for field in (ux_3D,uy_3D,f_3D):         
            """NOTICE: to accomodate different pickle versions"""
            picklename = case.path + 'field/' + 'pickle_tiger/' + field['name']+'_t%g' % tsimu +'.pkl'
    #             picklename = working_dir + 'field/' + 'pickle_desktop/' + field['name']+'_t%g' % t +'.pkl'
            exists = os.path.exists(picklename)
            # If the pickle is there read in the pickles
            if exists:
                field['value'] = load_object(picklename)
                print('pickle restored!')
            # If no pickle read in from the slice files and pickle dump
            if not exists:
                for sn in range (0, NSLICE-1):
                    filename = case.path + 'field/'+field['name']+'_t%g_slice%g' % (tsimu,sn)
                    snapshot = np.loadtxt(filename, dtype = np.str, delimiter='\t')
                    snapshot.reshape([NGRID,NGRID+1])
                    field['value'].append(snapshot[:,0:NGRID].astype(np.float))
                field['value'] = np.array(field['value'])
                save_object(field['value'], picklename)

            # Shift the values along x axis
            field['value'] = np.roll(field['value'], -idx, axis=1)

        case.ux_2D.append(np.average(ux_3D['value'], axis=0))
        case.uy_2D.append(np.average(uy_3D['value'], axis=0))
        case.f_2D.append(np.average(f_3D['value'], axis=0))
        
""" Helper function (readin): 
    With already specified p['t'], read in aerodynamic pressure and store as p_2D spanwise averaged time sequence of 2D array.
    The pressure here has not been processed (meaning remove the average because of the incompressible flow absolute value of pressure
    does not have meaning.) It still needs to be passed in processing_energy1 to clean the data.
    Attributes: 
        p['t']
        p['p_2D']: instantaneous pressure in x-z plane, already averaged spanwise.
        p['phat'], p['amp']: phase and amplitude of the pressure signal
"""



def field (case, time, PRE=False, PLOT=False):
    """Put the field slices together and compute some turbulent statistics.
    Pickle the fileds if it's the first time read-in.
    
    Args:
    case: the case instance providing metadata
    time: the times where we want to compute the field statistics
    
    Returns:
    Write the following fields as case attributes.
    
    TODO: add coordinate transformation.
    
    """
    NGRID = 512; NSLICE = 256; L0 = 2*np.pi
    working_dir = case.path
    case.tstat = time # time that the statistics are computed
    case.eta =[]; case.eta_shift = [] # 1D eta (shift or not)
    case.ux = []; case.uy = []; case.f = [] # 2D ux uy # 2D ux uy
    case.re_stress = []
    """Instead of recording shifted fields, record shifting index
    Then do field_shift = np.roll(field, -idx) in x direction when needed."""
    case.shift_index = []
    case.ux_shift = []; case.uy_shift = []; case.f_shift = [] # 2D ux uy shifted, leave blank
    case.re_stress_shift = [] # 2D re shifted
    case.uxmean = [] # 1D ux mean
    case.ux_center = []; case.uy_center = []; case.f_center = [] # The center non-averaged slice to show turbulence
    case.ux_yzcrest = []; case.uy_yzcrest = []; case.f_yzcrest = [] # The y-z plane slice at crest x
    case.ux_yztrough = []; case.uy_yztrough = []; case.f_yztrough = [] # The y-z plane slice at trough x
    
    for t in tqdm(time):
        # Read in eta (utilize the Interface class)
#         filename = working_dir + 'eta/eta_t%g' % 59
#         snapshot = pd.read_table(filename, delimiter = ',')
#         eta_raw = {'x':np.array(snapshot.x), 'z':np.array(snapshot.z), 'eta':np.array(snapshot.pos)}
#         xarray = np.linspace(-L0/2.,L0/2.,NGRID,endpoint=False)+L0/2**NGRID/2 
#         zarray = np.linspace(-L0/2.,L0/2.,NGRID,endpoint=False)+L0/2**NGRID/2
#         x_tile, z_tile = np.meshgrid(xarray,zarray)
#         eta_tile = griddata((eta_raw['x'].ravel(), eta_raw['z'].ravel()), eta_raw['eta'].ravel(), 
#                                  (x_tile, z_tile), method='nearest')
#         eta_1D = np.average(eta_tile, axis=0)
        """For precursor cases just read the fixed shape"""
        """For moving wave cases just read the fixed shape"""
        if (PRE == True): 
            interface = Interface2D(L0 = 2*np.pi, N = 512, path = case.path, 
                                    pre='eta/eta_pre', t = None, PRUNING=True, filename=case.path+'eta/eta_pre')
            eta_1D = np.average(interface.eta, axis=0)        
        else: 
            interface = Interface2D(L0 = 2*np.pi, N = 512, path = case.path, 
                                    pre='eta/eta_loc_t', t = t, PRUNING=True)
            eta_1D = np.average(interface.eta, axis=0)
        # Filter the data (subtract the mean)
        eta_1D_filtered = butter_lowpass_filter(eta_1D-np.average(eta_1D))
        analytic_signal = hilbert(eta_1D_filtered)
        phase = np.angle(analytic_signal)
        # Shift the velocity field along x axis so that phase starts at 0
        idx = (np.abs(phase - 0)).argmin()
#         eta_1D_shift = np.roll(eta_1D, -idx)
        ux_3D = {'name':'ux', 'value':[]} # axis0 in z, axis1 in x, axis2 in y  (in the code)
        uy_3D = {'name':'uy', 'value':[]}
        f_3D = {'name':'f', 'value':[]}

        # Read in the fields either from pickle or from slice data
        for field in (ux_3D,uy_3D,f_3D):         
            """NOTICE: to accomodate different pickle versions"""
            picklename = working_dir + 'field/' + 'pickle_tiger/' + field['name']+'_t%g' % t +'.pkl'
#             picklename = working_dir + 'field/' + 'pickle_desktop/' + field['name']+'_t%g' % t +'.pkl'
            exists = os.path.exists(picklename)
            # If the pickle is there read in the pickles
            if exists:
                field['value'] = load_object(picklename)
                print('pickle restored!')
            # If no pickle read in from the slice files and pickle dump
            if not exists:
                for sn in range (0, NSLICE-1):
                    filename = working_dir + 'field/'+field['name']+'_t%g_slice%g' % (t,sn)
                    snapshot = np.loadtxt(filename, dtype = np.str, delimiter='\t')
                    snapshot.reshape([NGRID,NGRID+1])
                    field['value'].append(snapshot[:,0:NGRID].astype(np.float))
                field['value'] = np.array(field['value'])
                save_object(field['value'], picklename)
                
        # Compute Reynolds stress (not shifted)
        ux_mean = np.tile(np.average(ux_3D['value'], axis=(0,1)), (ux_3D['value'].shape[1], 1)) 
        uy_mean = np.tile(np.average(uy_3D['value'], axis=(0,1)), (uy_3D['value'].shape[1], 1))
        re_stress_3D = (ux_3D['value']-ux_mean)*(uy_3D['value']-uy_mean)*(1-f_3D['value'])
        # Compute wave coherent stress (TODO!)
        
        # Append z direction averaged 2D profile
        case.eta.append(eta_1D)
        case.ux.append(np.average(ux_3D['value'], axis=0))
        case.uy.append(np.average(uy_3D['value'], axis=0))
        case.f.append(np.average(f_3D['value'], axis=0))
        # A few fields for visualization
        # First is the center field
        SAMPLE = int(ux_3D['value'].shape[0]/2)
        print(SAMPLE) # center in z
        case.ux_center.append(ux_3D['value'][SAMPLE].copy())
        case.uy_center.append(uy_3D['value'][SAMPLE].copy())
        case.f_center.append(f_3D['value'][SAMPLE].copy())        
        # Second is at crest and trough (require shifted field)
        # These are only for plotting
        if PLOT == True:
            ux_3D_shift = {'name':'ux_shift', 'value':[]} # axis0 in z, axis1 in x, axis2 in y  (in the code)
            uy_3D_shift = {'name':'uy_shift', 'value':[]}
            f_3D_shift = {'name':'f_shift', 'value':[]}
            for (field,field_shift) in zip((ux_3D,uy_3D,f_3D),(ux_3D_shift,uy_3D_shift,f_3D_shift)):
                field_shift['value'] = np.roll(field['value'], -idx, axis=1)
            case.ux_yzcrest.append(ux_3D_shift['value'][:,256,:].copy()) # z, x, y
            case.uy_yzcrest.append(uy_3D_shift['value'][:,256,:].copy())
            case.f_yzcrest.append(f_3D_shift['value'][:,256,:].copy())
            case.ux_yztrough.append(ux_3D_shift['value'][:,192,:].copy()) # z, x, y
            case.uy_yztrough.append(uy_3D_shift['value'][:,192,:].copy())
            case.f_yztrough.append(f_3D_shift['value'][:,192,:].copy())
        case.shift_index.append(idx)
        case.re_stress.append(np.average(re_stress_3D, axis=0))        
        # Additional 1D profile
        case.uxmean.append(np.average(ux_3D['value'], axis=(0,1)))
        case.yarray = np.linspace(0,case.L0,case.N,endpoint=False)+case.L0/2**case.N/2 # Centered grid for interpolation
        del(ux_3D, uy_3D, f_3D, ux_mean, uy_mean, re_stress_3D)
        gc.collect()
        

def read_p (case):
    case.p_2D = []
    NSLICE = 256
    NGRID = 512

    for i in tqdm(range(0, np.size(case.p['t']))):    
        pair_3D = {'name':'pair', 'value':[]}
        f_3D = {'name':'f', 'value': []}
        tsimu = case.p['t'][i] + case.tstart
        print(tsimu)
        phasei = np.where(np.isclose(np.array(case.phase['t']), case.p['t'][i]))[0][0] # Because t is a float
        idx = case.phase['idx'][phasei]

        # Read in the fields either from pickle or from slice data
        field = pair_3D
        """NOTICE: to accomodate different pickle versions"""
        picklename = case.path + 'field/' + 'pickle_tiger/' + field['name']+'_t%g' % tsimu +'.pkl'
    #             picklename = working_dir + 'field/' + 'pickle_desktop/' + field['name']+'_t%g' % t +'.pkl'
        exists = os.path.exists(picklename)
        # If the pickle is there read in the pickles
        if exists:
            field['value'] = load_object(picklename)
            print('pickle restored!')
        # If no pickle read in from the slice files and pickle dump
        if not exists:
            for sn in range (0, NSLICE-1):
                filename = case.path + 'field/'+field['name']+'_run'+'_t%g_slice%g' % (tsimu,sn)
                snapshot = np.loadtxt(filename, dtype = np.str, delimiter='\t')
                snapshot.reshape([NGRID,NGRID+1])
                field['value'].append(snapshot[:,0:NGRID].astype(np.float))
            field['value'] = np.array(field['value'])
            save_object(field['value'], picklename) 
        field['value'] = np.roll(field['value'], -idx, axis=1)

        case.p['p_2D'].append(np.average(pair_3D['value'], axis=0))


""" A go-to filter function, by default N=512, CUT=4 """
from scipy.signal import butter,filtfilt
def butter_lowpass_filter(data, CUT=4, N=512):
    T = 4     # Sample Period
    fs = 512/T      # Sample rate, Hz
    cutoff = CUT    # desired cutoff frequency of the filter, Hz, slightly higher than actual 1.2 Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# os.chdir('/home/jiarong/research/projects/turbulence/preliminary_cluster/stopforcing_restore_second')

# def main():
#     clock = np.arange(900, 950)
#     ensemble_pickle(clock, picklename="ensemble")
#     print(os.getcwd())

# if __name__ == "__main__":
#     main()

# print("DONE")





