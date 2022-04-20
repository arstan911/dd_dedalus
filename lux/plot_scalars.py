import numpy as np
from matplotlib import pyplot as plt
import pathlib
import h5py

run_dir = pathlib.Path(".")
scalar_dir = run_dir / "scalars"
# print(scalar_dir)
kx0_part = True

with h5py.File(scalar_dir / "scalars_s1.h5", mode='r') as file:
    # print(list(file['/scales']))
    # print(list(file['/tasks']))
    sim_time = np.array(file['/scales/sim_time'])  # or np.array(file['scales']['sim_time'])
    #nuT1 = np.array(file['/tasks/nusselt_T_1'])[:,0,0]
    nuT2 = np.array(file['/tasks/nusselt_T_2'])[:,0,0]
    #nuC1 = np.array(file['/tasks/nusselt_C_1'])[:,0,0]
    nuC2 = np.array(file['/tasks/nusselt_C_2'])[:,0,0]
    TD = np.array(file['/tasks/Temp_dis_1'])[:,0,0]
    CD = np.array(file['/tasks/Comp_dis_1'])[:,0,0]
    KE = np.array(file['/tasks/total kinetic energy'])[:,0,0]
    #rho = np.array(file['/tasks/Density'])[:,0,0]

#print(np.shape(nu2))
#print(np.shape(ILT))
#print(np.shape(sim_time))


#nu1_len=len(nu1)
#cut_index1 = int(nu1_len*0.7)
#nu1_ave = nu1[cut_index1:]
#Nu_com = sum(nu1_ave) / len(nu1_ave)
#print('Nusselt:', Nu_com)

#plot1 = plt.figure(1)
#plt.plot(sim_time, nuT1, c='C0', ls='-', label=r'$nusselt$')  
#plt.legend()
#plt.xlabel(r'$t$')
#plt.ylabel(r'nusselt_T')
#plt.savefig('nusselt_T_1.png')
#plt.show()


#plot1 = plt.figure(2)
#plt.semilogy(sim_time, nuT1, c='C0', ls='-', label=r'$nusselt$')
#plt.legend()
#plt.xlabel(r'$t$')
#plt.ylabel(r'nusselt')

#plt.savefig('nusselt_T_1_log.png')

#plt.show()


plot1 = plt.figure(3)
plt.plot(sim_time, nuT2, c='C0', ls='-', label=r'$nusselt_T$')  
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'nusselt_T')
plt.savefig('nusselt_T_2.png')
plt.show()


plot1 = plt.figure(4)
plt.semilogy(sim_time, nuT2-1.0 , c='C0', ls='-', label=r'$nusselt_T$')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'nusselt_T')
plt.savefig('nusselt_T_2_log.png')
plt.show()


#plot1 = plt.figure(5)
#plt.plot(sim_time, nuC1, c='C0', ls='-', label=r'$nusselt$')  
#plt.legend()
#plt.xlabel(r'$t$')
#plt.ylabel(r'nusselt_C')
#plt.savefig('nusselt_C_1.png')
#plt.show()


#plot1 = plt.figure(6)
#plt.semilogy(sim_time, nuC1, c='C0', ls='-', label=r'$nusselt$')
#plt.legend()
#plt.xlabel(r'$t$')
#plt.ylabel(r'nusselt_C')
#plt.savefig('nusselt_C_1_log.png')
#plt.show()


plot1 = plt.figure(7)
plt.plot(sim_time, nuC2, c='C0', ls='-', label=r'$nusselt_C$')  
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'nusselt_C')
plt.savefig('nusselt_C_2.png')
plt.show()


plot1 = plt.figure(8)
plt.semilogy(sim_time, nuC2-1.0, c='C0', ls='-', label=r'$nusselt_C$')
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'nusselt_C')
plt.savefig('nusselt_C_2_log.png')
plt.show()


#plot1 = plt.figure(3)
#plt.semilogy(sim_time, wT, c='C0', ls='-', label=r'$wT$')  
#plt.legend()
#plt.xlabel(r'$t$')
#plt.ylabel(r'wT')
#plt.savefig('wT_log.png')
#plt.show()


plot1 = plt.figure(9)
plt.plot(sim_time, TD, c='C0', ls='-', label=r'$T_dissipation$')  
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'Temp_diss')
plt.savefig('T_dissipation.png')
plt.show()

plot1 = plt.figure(10)
plt.plot(sim_time, CD, c='C0', ls='-', label=r'$C_dissipation$')  
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'C_diss')
plt.savefig('C_dissipation.png')
plt.show()

plot1 = plt.figure(11)
plt.plot(sim_time,KE , c='C0', ls='-', label=r'$energy$')  
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'energy')
plt.savefig('energy.png')
plt.show()

plot1 = plt.figure(12)
plt.semilogy(sim_time,KE , c='C0', ls='-', label=r'$energy$')  
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'energy')
plt.savefig('energy_log.png')
plt.show()



#plot1 = plt.figure(12)
#plt.plot(sim_time, rho, c='C0', ls='-', label=r'$density$')  
#plt.legend()
#plt.xlabel(r'$t$')
#plt.ylabel(r'density')
#plt.savefig('density.png')
#plt.show()




#plot1 = plt.figure(8)
#plt.plot(sim_time, bzz, c='C0', ls='-', label=r'$bz$')
#plt.plot(sim_time, bdiff, c='C0', ls='-', label=r'$bdif$')
#plt.legend()
#plt.xlabel(r'$t$')
#plt.ylabel(r'bz')
#plt.savefig('bzbdiff.png')
#plt.show()

#plot1 = plt.figure(8)
#plt.semilogy(sim_time, nu1, c='C0', ls='-', label=r'$nu_wb$')
#plt.semilogy(sim_time, nu2, c='C0', ls='-', label=r'$nu_dissipation$')
#plt.legend()
#plt.xlabel(r'$t$')
#plt.ylabel(r'bz')
#plt.savefig('nu-nu.png')
#plt.show()
