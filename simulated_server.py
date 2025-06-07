from cheetah.particles import ParticleBeam
from beamdriver import SimDriver, SimServer
from cheetah.accelerator import Segment 
import torch
from utils.load_yaml import load_relevant_controls
from utils.pvdb import create_pvdb
import pprint 
#design_incoming = ParticleBeam.from_openpmd_file(path='impact_inj_output_YAG03.h5', energy = torch.tensor(125e6),dtype=torch.float32)
#lcls_lattice = Segment.from_lattice_json("lcls_cu_segment_otr2.json")
design_incoming_beam = {'path': 'h5/impact_inj_output_YAG03.h5',
                         'energy': torch.tensor(125e6),
                         'dtype':torch.float32}
lcls_lattice = 'lattices/lcls_cu_segment_otr2.json'
devices = load_relevant_controls('yaml_configs/DL1.yaml')
screen_name = 'OTRS:IN20:571'
screen_defaults = {'n_row': 1392, 'n_col': 1040, 'resolution': 4.65, 'pneumatic': 'OUT' }
PVDB = create_pvdb(devices, **screen_defaults)
custom_pvs = {'VIRT:BEAM:EMITTANCES': {'type':'float', 'count': 2},
              'VIRT:BEAM:RESET_SIM': {'value': 0}   
}
PVDB.update(custom_pvs)
pprint.pprint(PVDB)
server = SimServer(PVDB)
driver = SimDriver(
    server=server,
    screen=screen_name,
    devices=devices,
    design_incoming_beam=design_incoming_beam,
    lattice_file=lcls_lattice
)

print('Starting simulated server')
server.run()
