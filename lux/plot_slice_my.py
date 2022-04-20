import numpy as np
from matplotlib import pyplot as plt
import pathlib
import h5py

run_dir = pathlib.Path(".")
scalar_dir = run_dir / "snapshots2"
# print(scalar_dir)
kx0_part = True

with h5py.File(scalar_dir / "snapshots2_s1.h5", mode='r') as file:
    # print(list(file['/scales']))
    # print(list(file['/tasks']))
    sim_time = np.array(file['/scales/sim_time'])  # or np.array(file['scales']['sim_time'])
    T = np.array(file['/tasks/T'])[:,:,:]
    x = np.array(file['/scales/x/1.0'])
    z = np.array(file['/scales/z/1.0'])

 
print(np.shape(T))
print(np.shape(x))
print(np.shape(z))
print(np.shape(sim_time))

plt.pcolormesh(x,z,T[64].T)
plt.colorbar()
plt.savefig('Tsnap.png')
   
    

