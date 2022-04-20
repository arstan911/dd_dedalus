"""
Dedalus script for 2D Rayleigh-Benard convection.
This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).
This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.
To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5
This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.
To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py
The simulations should take a few process-minutes to run.
"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters
Lx, Lz = (100., 100.)
Prandtl = 0.3
DiffRatio = 0.3
invDensRatio= 1.5


# Create bases and domain
x_basis = de.Fourier('x', 2048, interval=(0, Lx), dealias=3/2)
z_basis = de.Chebyshev('z', 128, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','Temp','Comp','u','w','Tempz','Compz','uz','wz'])
problem.meta['p','Temp','Comp','u','w']['z']['dirichlet'] = True
problem.parameters['Lx'] = Lx
problem.parameters['Lz'] = Lz
problem.parameters['R0m1'] = invDensRatio
problem.parameters['tau'] = DiffRatio 
problem.parameters['Pr'] = Prandtl
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(Temp) - (dx(dx(Temp)) + dz(Tempz)) - w       = -(u*dx(Temp) + w*Tempz)")
problem.add_equation("dt(Comp) - (dx(dx(Comp)) + dz(Compz))*tau - R0m1*w       = -(u*dx(Comp) + w*Compz)")
problem.add_equation("dt(u) - Pr*(dx(dx(u)) + dz(uz)) + dx(p)     = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) - Pr*(dx(dx(w)) + dz(wz)) + dz(p) - (Temp-Comp)*Pr = -(u*dx(w) + w*wz)")
problem.add_equation("Tempz - dz(Temp) = 0")
problem.add_equation("Compz - dz(Comp) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_bc("left(Tempz) = 0")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(Tempz) = 0")
problem.add_bc("right(Compz) = 0")
problem.add_bc("left(Compz) = 0")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK443)
logger.info('Solver built')

# Initial conditions or restart
if not pathlib.Path('restart.h5').exists():

    # Initial conditions
    x = domain.grid(0)
    z = domain.grid(1)
    Comp = solver.state['Comp']
    Compz = solver.state['Compz']

    # Random perturbations, initialized globally for same results in parallel
    gshape = domain.dist.grid_layout.global_shape(scales=1)
    slices = domain.dist.grid_layout.slices(scales=1)
    rand = np.random.RandomState(seed=42)
    noise = rand.standard_normal(gshape)[slices]  #Check this in documentation

    # Linear background + perturbations damped at walls
    zb, zt = z_basis.interval
    pert =  1e-3 * noise * (zt - z) * (z - zb)
    Comp['g'] = pert - z*invDensRatio 
    Comp.differentiate('z', out=Compz)

    # Timestepping and output
    dt = 0.1
    stop_sim_time = 1000
    fh_mode = 'overwrite'

else:
    # Restart
    write, last_dt = solver.load_state('restart.h5', -1)

    # Timestepping and output
    dt = last_dt
    stop_sim_time = 2000
    fh_mode = 'append'

# Integration parameters
solver.stop_sim_time = stop_sim_time

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10.0, max_writes=1200, mode=fh_mode)
snapshots.add_system(solver.state)

#snapshots2 = solver.evaluator.add_file_handler('snapshots2', sim_dt=10.0 , max_writes=1200)
#snapshots2.add_task('Temp', layout='g', name='Temp')
#snapshots2.add_task('Comp', layout='g', name='Comp')
#snapshots2.add_task('u', layout='g', name='u')
#snapshots2.add_task('w', layout='g', name='w')

scalars = solver.evaluator.add_file_handler('scalars', sim_dt=10.0, max_writes=800)
scalars.add_task("(Lx*Lz)**(-1)*integ((dx(Temp)**2 + Tempz*Tempz),'x','z')", name = 'Temp_dis_1')
scalars.add_task("(Lx*Lz)**(-1)*integ((dx(Comp)**2 + Compz*Compz),'x','z')", name = 'Comp_dis_1')
scalars.add_task("integ(0.5 * (u*u + w*w))", name = "total kinetic energy")

#scalars.add_task("(1+integ((dx(Temp)**2 + Tempz*Tempz),'x','z')/(Lx*Lz))", name = "nusselt_T_1")
scalars.add_task("1/(1-integ((dx(Temp)**2 + Tempz*Tempz),'x','z')/(Lx*Lz))", name = "nusselt_T_2")
#scalars.add_task("(1+integ((dx(Comp)**2 + Compz*Compz),'x','z')/(Lx*Lz))", name = "nusselt_C_1")
scalars.add_task("1/(1-R0m1**(-2)*integ((dx(Comp)**2 + Compz*Compz),'x','z')/(Lx*Lz))", name = "nusselt_C_2")

profiles = solver.evaluator.add_file_handler('vertical_profiles', sim_dt = 10.0, max_writes=800)
profiles.add_task("Lx**(-1)*integ(Temp,'x')",layout='g', name = 'Temp_HA_VP')
profiles.add_task("Lx**(-1)*integ(Comp,'x')",layout='g', name = 'Comp_HA_VP')
profiles.add_task("z*(1-R0m1)-Lx**(-1)*integ(Temp,'x')+Lx**(-1)*integ(Comp,'x')", name = 'Density')

#scalars.add_task("integ((w*b),'x','z')/(Lx*Lz)", name='wT')
#scalars.add_task("1+(R*Pr)**(1/2)*integ((w*b),'x','z')/(Lx*Lz)", name='nusselt_ft')
#scalars.add_task("1/(1-(R*Pr)**(1/2)*integ((w*b),'x','z')/(Lx*Lz))", name = "nusselt_ff")
#scalars.add_task("1/(1-integ((dx(b)**2 + bz*bz),'x','z')/(Lx*Lz))", name = "nusselt_ff_2")
#scalars.add_task("integ((b.differentiate('x')**2 + b.differentiate('z')**2),'x','z')", name = 'Int_dT')
#scalars.add_task("(Lx*Lz)**(-1)*integ((dx(b)**2 + bz*bz),'x','z')", name = 'Int_dT')
#scalars.add_task("(Lx*Lz)**(-1)*integ((bz*bz),'x','z')", name = 'bz')
#scalars.add_task("(Lx*Lz)**(-1)*integ((b.differentiate('z')*b.differentiate('z')),'x','z')", name = 'bdif')
#scalars.add_task("integ( - left(b) - right(b),'x')", name = 'Int_Tt-Tb')




# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
#flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
#flow.add_property("sqrt(u*u + w*w) / D", name='Re')

 #Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
           logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            #logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))