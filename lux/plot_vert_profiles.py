import numpy as np
from matplotlib import pyplot as plt
import pathlib
import h5py

run_dir = pathlib.Path(".")
scalar_dir = run_dir / "vertical_profiles"
# print(scalar_dir)
kx0_part = True

with h5py.File(scalar_dir / "vertical_profiles_s1.h5", mode='r') as file:
    # print(list(file['/scales']))
    # print(list(file['/tasks']))
    sim_time = np.array(file['/scales/sim_time'])  # or np.array(file['scales']['sim_time'])
    T = np.array(file['/tasks/Temp_HA_VP'])[:,:]
    C = np.array(file['/tasks/Comp_HA_VP'])[:,:]
    z = np.array(file['/scales/z/1.0'])
    rho = np.array(file['/tasks/Density'])[:,:]
 
print(np.shape(T))
print(np.shape(C))
print(np.shape(z))
print(np.shape(rho))
#print(np.shape(sim_time))

plot1 = plt.figure(1)
plt.plot(z,T[28,0])
plt.xlabel(r'$z$')
plt.ylabel(r'Horizontally averaged Temperature')
plt.savefig('Temp_vertical_profile.png')
plt.show()

plot2 = plt.figure(2)
plt.plot(z,C[28,0])
plt.xlabel(r'$z$')
plt.ylabel(r'Horizontally averaged Composition')
plt.savefig('Comp_vertical_profile.png')
plt.show()
plot3 = plt.figure(3)
plt.plot(z,rho[28,0])
plt.xlabel(r'$z$')
plt.ylabel(r'density')
plt.savefig('density.png')
plt.show()

   
    

