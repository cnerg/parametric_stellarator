import parastell.invessel_build as ivb
from parastell.utils import ribs_from_kisslinger_format
import numpy as np
import pystell.read_vmec as read_vmec

(
    toroidal_angles,
    num_toroidal_angles,
    num_poloidal_angles,
    periods,
    custom_ribs,
) = ribs_from_kisslinger_format("wistd_1m_alpha10.txt", scale=1)
poloidal_angles = np.linspace(0, 360, num_poloidal_angles)
ks = ivb.RibBasedSurface(custom_ribs, toroidal_angles, poloidal_angles)
ks.build_analytic_surface()
vmec_obj = read_vmec.VMECData("plasma_wistelld.nc")
vs = ivb.VMECSurface(vmec_obj)

toroidal_angles = np.linspace(0, 90, 64)
poloidal_angles = np.linspace(0, 360, 64)

uniform_unit_thickness = np.ones((len(toroidal_angles), len(poloidal_angles)))

radial_build = ivb.RadialBuild(
    toroidal_angles,
    poloidal_angles,
    1,
    {
        "first_wall": {"thickness_matrix": uniform_unit_thickness * 50},
    },
)
invessel_build = ivb.InVesselBuild(
    ks, radial_build, num_ribs=64, num_rib_pts=64
)
invessel_build.use_pydagmc = True
invessel_build.populate_surfaces()
invessel_build.calculate_loci()
invessel_build.generate_components()
invessel_build.dag_model.write_file("ivb.h5m")
invessel_build.dag_model.write_file("ivb.vtk")
