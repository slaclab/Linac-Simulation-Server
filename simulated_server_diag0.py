from cheetah.particles import ParticleBeam
from beamdriver import SimDriver, SimServer
from cheetah.accelerator import Segment 
import torch
from utils.load_yaml import load_relevant_controls
from utils.pvdb import create_pvdb
import pprint

incoming_beam = ParticleBeam.from_twiss(
    beta_x=torch.tensor(9.34),
    alpha_x=torch.tensor(-1.6946),
    emittance_x=torch.tensor(1e-7),
    beta_y=torch.tensor(9.34),
    alpha_y=torch.tensor(-1.6946),
    emittance_y=torch.tensor(1e-7),
    energy=torch.tensor(90e6),
    num_particles=100000,
    total_charge=torch.tensor(1e-9)
)

#diag0_lattice = Segment.from_lattice_json("lattices/diag0_reconstruction.json")
#print(diag0_lattice)
devices = load_relevant_controls('yaml_configs/DIAG0.yaml')
screen_name = 'OTRS:DIAG0:420'
#TODO: fix some type of bug were defaults are not getting set from passable dictionary.... 
screen_defaults = {'n_row': 1944, 'n_col': 1472, 'resolution': 23.33 }
tcav_defaults = {}
PVDB = create_pvdb(devices, **screen_defaults)
custom_pvs = {'VIRT:BEAM:EMITTANCES': {'type':'float', 'count': 2},
            'VIRT:BEAM:MU:XY': {'type':'float', 'count': 2},
            'VIRT:BEAM:SIGMA:XY': {'type':'float', 'count': 2},      
            'VIRT:BEAM:RESET_SIM': {'value': 0},
}
PVDB.update(custom_pvs)
pprint.pprint(PVDB)

server = SimServer(PVDB)
driver = SimDriver(
    server=server,
    screen=screen_name,
    devices=devices,
    particle_beam=incoming_beam,
    lattice_file="lattices/diag0.json" # check that lattice file is actually real..
)

print('Starting simulated server')
server.run()