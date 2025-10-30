import numpy as np
def amber_get_E(file):
    i=0;E_list=[]
    with open(file)as f:
        for line in f:
            if i==1:
                s=line.split()
                E=float(s[1])
                E_list.append(E)
                break
            if (line.find(' RMS ')>-1):
                i=1
    return E_list

def dft_get_E(path_file):
    with open(path_file,'r') as f:
        lines=f.readlines()
    lines_E=[]
    for line in lines:
        if line.find('E(R')>-1 and len(line.split())>5:
            lines_E.append(float(line.split()[4])*627.510)
    E_list=[lines_E[5*i]-lines_E[5*i+1]-lines_E[5*i+2] for i in range(int(len(lines_E)/5))]
    return E_list

def Rod(n,theta_in):
    nx,ny,nz=n
    theta_t=np.radians(theta_in)
    Rod=np.array([[np.cos(theta_t)+(nx**2)*(1-np.cos(theta_t)),nx*ny*(1-np.cos(theta_t))-nz*np.sin(theta_t),nx*nz*(1-np.cos(theta_t))+ny*np.sin(theta_t)],
                [nx*ny*(1-np.cos(theta_t))+nz*np.sin(theta_t),np.cos(theta_t)+(ny**2)*(1-np.cos(theta_t)),ny*nz*(1-np.cos(theta_t))-nx*np.sin(theta_t)],
                [nx*nz*(1-np.cos(theta_t))-ny*np.sin(theta_t),ny*nz*(1-np.cos(theta_t))+nx*np.sin(theta_t),np.cos(theta_t)+(nz**2)*(1-np.cos(theta_t))]])
    return Rod

def R2atom(R):
    if R==1.8:
        return 'S'
    elif R==1.7:
        return 'C'
    elif R==1.2:
        return 'H'
    elif R==1.47:
        return 'F'
    else:
        return 'X'

_VDW = {'H':1.20,'C':1.70,'N':1.55,'O':1.52,'F':1.47,'P':1.80,'S':1.80,'CL':1.75,'BR':1.85,'I':1.98}

def vdw_radius(sym: str) -> float: #sym→symbol(元素記号)
    s = sym.strip().upper()
    if s in _VDW: return _VDW[s]
    return _VDW['C']

def check_calc_status(df_cur,A1,A2,A3,a,b):
    try:        
        return df_cur.loc[
                        (df_cur['A1']==A1)&
                        (df_cur['A2']==A2)&
                        (df_cur['A3']==A3)&
                        (df_cur['a']==a)&
                        (df_cur['b']==b), 'status'].values[0] == 'Done'
    except IndexError:
        return False

