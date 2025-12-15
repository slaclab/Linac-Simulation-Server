import torch
import os
import pathlib

from cheetah.particles import ParticleBeam

from simulation_server.virtual_accelerator import VirtualAccelerator

FILEPATH = pathlib.Path(__file__).parent.resolve()
LCLS_LATTICE = pathlib.Path(os.environ.get("LCLS_LATTICE", "/sdf/group/ad/sw/scm/repos/optics/lcls-lattice/cheetah"))


def get_virtual_accelerator(name, monitor_overview=False, measurement_noise_level=None):
    """
    Create an instance of VirtualAccelerator for a given beamline.

    Parameters
    ----------
    name: str
        The name of the virtual accelerator. Current options are:
            - diag0
            - cu_injector
    monitor_overview: bool, optional
        If True, print out an overview plot of the accelerator
        simulation each time a PV is changed.
    measurement_noise_level: float, optional
        If provided, adds realistic noise to measurements.
        See `simulation_server.virtual_accelerator.utils.add_noise` for details.

    Returns
    -------
    VirtualAccelerator
        An instance of the VirtualAccelerator class.

    """
    if name == "diag0":
        incoming_beam = ParticleBeam.from_twiss(
            beta_x=torch.tensor(9.34),
            alpha_x=torch.tensor(-1.6946),
            emittance_x=torch.tensor(1e-7),
            beta_y=torch.tensor(9.34),
            alpha_y=torch.tensor(-1.6946),
            emittance_y=torch.tensor(1e-7),
            energy=torch.tensor(90e6),
            num_particles=100000,
            total_charge=torch.tensor(1.0),
        )

        mapping_file = os.path.join(FILEPATH, "mappings", "lcls_elements.csv")
        lattice_file = os.path.join(LCLS_LATTICE,"cheetah", "sc_diag0.json")
        subcell_dest = None
        
    elif name in ("nc_injector", 'nc_hxr'):
        incoming_beam = ParticleBeam.from_openpmd_file(
            path=os.path.join(FILEPATH, "beams", "impact_inj_output_YAG03.h5"),
            energy=torch.tensor(64e6),
            dtype=torch.float32,
        )
        incoming_beam.particle_charges = torch.tensor(1.0)

        mapping_file = os.path.join(FILEPATH, "mappings", "lcls_elements.csv") 
        lattice_file = os.path.join(LCLS_LATTICE,"cheetah","nc_hxr.json")

        if name == "nc_injector":
            subcell_dest = 'otr2'
        else:
            subcell_dest = None

    return VirtualAccelerator(
        lattice_file=lattice_file,
        initial_beam_distribution=incoming_beam,
        mapping_file=mapping_file,
        monitor_overview=monitor_overview,
        measurement_noise_level=measurement_noise_level,
        subcell_dest=subcell_dest
    )
