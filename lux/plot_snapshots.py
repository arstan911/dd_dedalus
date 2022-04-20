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
    T = np.array(file['/tasks/T'])[:,0,0]
    C = np.array(file['/tasks/Comp'])[:,0,0]
    u = np.array(file['/tasks/u'])[:,0,0]
    w = np.array(file['/tasks/w'])[:,0,0]


plot1 = plt.figure(1)
plt.plot(sim_time, T, c='C0', ls='-', label=r'$temperature$')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'temp')
plt.savefig('temperature.png')
plt.show()


plot2 = plt.figure(2)
plt.semilogy(sim_time, w, c='C0', ls='-', label=r'$w$')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'w')
plt.savefig('w.png')
plt.show()

plot2 = plt.figure(3)
plt.semilogy(sim_time, u, c='C0', ls='-', label=r'$u$')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'u')
plt.savefig('u.png')
plt.show()

plot2 = plt.figure(4)
plt.semilogy(sim_time, C, c='C0', ls='-', label=r'$Composition$')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'Composition')
plt.savefig('comp.png')
plt.show()