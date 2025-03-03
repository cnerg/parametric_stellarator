import parastell.invessel_build as ivb
import parastell.parastell as ps
import numpy as np

# Get a predefined set of points representing the first wall.
custom_surface_ribs = np.load(
    "../tests/files_for_tests/custom_surface_ribs.npy"
)

# For this example, the ribs and points on the ribs are evenly spaced, which
# is not required.
num_toroidal_angles, num_poloidal_angles, _ = custom_surface_ribs.shape
toroidal_angles = np.linspace(0, 90, num_toroidal_angles)
poloidal_angles = np.linspace(0, 360, num_poloidal_angles)

# Create a ReferenceSurface object from the known points and corresponding
# toroidal and poloidal angles.
ks = ivb.RibBasedSurface(custom_surface_ribs, toroidal_angles, poloidal_angles)

# A VMEC file is required for source mesh generation
vmec_file = "wout_vmec.nc"

# Instantiate ParaStell build. The PyDAGMC workflow is shown here, however
# the Cad-to-Dagmc and Cubit workflows also support the use of custom first
# wall profiles.
stellarator = ps.Stellarator(vmec_file, ref_surf=ks)

# Define desired toroidal and poloidal angles for building the stellarator
toroidal_angles = [0.0, 11.25, 22.5, 33.75, 45.0, 56.25, 67.5, 78.75, 90.0]
poloidal_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
wall_s = 1.08
# Use more points in the point cloud to smooth the ivb components
num_ribs = 100
num_rib_pts = 100

# Define a matrix of uniform unit thickness
uniform_unit_thickness = np.ones((len(toroidal_angles), len(poloidal_angles)))

radial_build_dict = {
    "first_wall": {"thickness_matrix": uniform_unit_thickness * 5},
    "breeder": {
        "thickness_matrix": (
            [
                [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0, 75.0, 75.0],
                [65.0, 25.0, 25.0, 65.0, 75.0, 75.0, 75.0, 75.0, 65.0],
                [45.0, 45.0, 75.0, 75.0, 75.0, 75.0, 75.0, 45.0, 45.0],
                [65.0, 75.0, 75.0, 75.0, 75.0, 65.0, 25.0, 25.0, 65.0],
                [75.0, 75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 75.0, 25.0, 25.0, 75.0, 75.0, 75.0],
                [75.0, 75.0, 75.0, 25.0, 25.0, 25.0, 75.0, 75.0, 75.0],
            ]
        )
    },
    "back_wall": {"thickness_matrix": uniform_unit_thickness * 5},
    "shield": {"thickness_matrix": uniform_unit_thickness * 40},
    "vacuum_vessel": {
        "thickness_matrix": uniform_unit_thickness * 10,
        "mat_tag": "vac_vessel",
    },
}
# Construct in-vessel components
stellarator.construct_invessel_build(
    toroidal_angles,
    poloidal_angles,
    wall_s,
    radial_build_dict,
    use_pydagmc=True,
    num_ribs=num_ribs,
    num_rib_pts=num_rib_pts,
    scale=1,
)

# Define build parameters for magnet coils
coils_file = "coils.example"
width = 40.0
thickness = 50.0
toroidal_extent = 90.0

stellarator.construct_magnets_from_filaments(
    coils_file, width, thickness, toroidal_extent, sample_mod=6
)
stellarator.export_magnets(
    step_filename="magnets",
    export_mesh=False,
)

stellarator.build_pydagmc_model(
    magnet_exporter="cad_to_dagmc", max_mesh_size=60
)
stellarator.export_pydagmc_model(filename="dagmc")
