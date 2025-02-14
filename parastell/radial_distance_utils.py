import numpy as np
import cubit
import pystell.read_vmec as read_vmec

from . import magnet_coils
from . import invessel_build as ivb
from .cubit_utils import init_cubit
from .utils import downsample_loop


def reorder_filament(coil):
    """Reorders the filament coordinates of a MagnetCoil class object such that
    they begin near the outboard midplane, and initially progress positively in
    the z-direction.

    Arguments:
        coil (object): MagnetCoil class object.

    Returns:
        reordered_coords (2-D iterable of float): reordered list of Cartesian
            coordinates defining a MagnetCoil filament.
    """
    # Start the filament at the outboard midplane
    outboard_index = coil.filament.get_ob_mp_index()
    if outboard_index != 0:
        coil.filament.reorder_coords(outboard_index)
    # Ensure points initially progress in positive z-direction
    coil.filament.orient_coords()


def reorder_coils(magnet_set):
    """Reorders a set of magnetic coils toroidally and reorders their filament
    coordinates such that they begin near the outboard midplane, and initially
    progress positively in the z-direction.

    Arguments:
        magnet_set (object): MagnetSet class object.

    Returns:
        magnet_coils (list of object): reordered list of MagnetCoil class
            objects.
    """
    magnet_set.populate_magnet_coils()
    magnet_coils = magnet_set.magnet_coils

    [reorder_filament(coil) for coil in magnet_coils]

    return magnet_coils


def build_line(point_1, point_2):
    """Builds a line between two points in Coreform Cubit.

    Arguments:
        point_1 (1-D iterable of float): Cartesian coordinates of first point.
        point_2 (1-D iterable of float): Cartesian coordinates of second point.

    Returns:
        curve_id (int): index of curve created in Coreform Cubit.
    """
    point_1 = " ".join(str(val) for val in point_1)
    point_2 = " ".join(str(val) for val in point_2)
    cubit.cmd(f"create curve location {point_1} location {point_2}")
    curve_id = cubit.get_last_id("curve")

    return curve_id


def build_magnet_surface(filaments, sample_mod=1):
    """Builds a surface in Coreform Cubit spanning a list of coil filaments.

    Arguments:
        filaments (list of object): list of Filament class objects,
            ordered toroidally. Each Filament object must also have its
            filament coordinates ordered poloidally (see reorder_coils
            function).
        sample_mod (int): sampling modifier for filament points (defaults to
            1). For a user-defined value n, every nth point will be sampled.
    """
    init_cubit()

    surface_lines = [
        [
            build_line(coord, next_coord)
            for coord, next_coord in zip(
                downsample_loop(filament.coords, sample_mod),
                downsample_loop(next_filament.coords, sample_mod),
            )
        ]
        for filament, next_filament in zip(filaments[:-1], filaments[1:])
    ]
    surface_lines = np.array(surface_lines)

    surface_sections = np.reshape(
        surface_lines,
        (
            len(filaments) - 1,
            len(downsample_loop(filaments[0].coords, sample_mod)),
        ),
    )

    [
        [
            cubit.cmd(f"create surface skin curve {line} {next_line}")
            for line, next_line in zip(lines[:-1], lines[1:])
        ]
        for lines in surface_sections
    ]


def fire_ray(point, direction):
    """Performs a ray-firing operation in Coreform Cubit from a reference point
    and along a reference direction to determine the distance from that point
    to the magnet surface.

    Arguments:
        point (iterable of float): Cartesian coordinates of reference point.
        direction (iterable of float): reference direction in
            Cartesian-coordinate system.

    Returns:
        distance (float): distance between reference point and magnet surface,
            along reference direction.
    """
    cubit.cmd(f"create vertex {point[0]} {point[1]} {point[2]}")
    vertex_id = cubit.get_last_id("vertex")

    cubit.cmd(
        f"create curve location at vertex {vertex_id} "
        f"location fire ray location at vertex {vertex_id} "
        f"direction {direction[0]} {direction[1]} {direction[2]} at "
        "surface all maximum hits 1"
    )

    curve_id = cubit.get_last_id("curve")
    distance = cubit.get_curve_length(curve_id)

    return distance


def measure_surface_coils_separation(surface):
    """Determines the distance between a given Surface class object and a
    surface spanning a set of MagnetCoil class objects via a ray-firing
    operation in Coreform Cubit.

    Arguments:
        surface (object): Surface class object.

    Returns:
        distance_matrix (2-D np.array of float): matrix of distances between
            points defining surface.Ribs and magnet surface, along
            Rib._normals.
    """
    distance_matrix = np.array(
        [
            [
                fire_ray(point, direction)
                for point, direction in zip(rib.rib_loci, rib._normals())
            ]
            for rib in surface.Ribs
        ]
    )

    return distance_matrix


def measure_fw_coils_separation(
    vmec_file,
    toroidal_angles,
    poloidal_angles,
    wall_s,
    coils_file,
    width,
    thickness,
    sample_mod=1,
    custom_fw_profile=None,
):
    """Measures the distance between a given first wall definition and a set of
    magnet filaments, at specified angular locations and along the profile
    normal at those angular locations, using ray-firing in Coreform Cubit.

    Arguments:
        vmec_file (str): path to plasma equilibrium VMEC file.
        toroidal_angles (array of float): toroidal angles at which distances
            should be computed [deg].
        poloidal_angles (array of float): poloidal angles at which distances
            should be computed [deg].
        wall_s (float): closed flux surface label extrapolation at wall.
        coils_file (str): path to coil filament data file.
        width (float): width of coil cross-section in toroidal direction [cm].
        thickness (float): thickness of coil cross-section in radial direction
            [cm].
        sample_mod (int): sampling modifier for filament points (defaults to
            1). For a user-defined value n, every nth point will be sampled.
        custom_fw_profile (2-D iterable of float): thickness matrix defining
            first wall profile (defaults to None).

    Returns:
        radial_distance_matrix (2-D np.array of float):
    """
    if custom_fw_profile is None:
        custom_fw_profile = np.zeros(
            (len(toroidal_angles), len(poloidal_angles))
        )

    vmec = read_vmec.VMECData(vmec_file)
    radial_build_dict = {"chamber": {"thickness_matrix": custom_fw_profile}}

    radial_build = ivb.RadialBuild(
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build_dict,
        split_chamber=False,
    )
    invessel_build = ivb.InVesselBuild(
        vmec,
        radial_build,
        # Set num_ribs and num_rib_pts to be less than length of corresponding
        # array to ensure that only defined angular locations are used
        num_ribs=len(toroidal_angles) - 1,
        num_rib_pts=len(poloidal_angles) - 1,
    )
    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    surface = invessel_build.Surfaces["chamber"]

    magnet_set = magnet_coils.MagnetSetFromFilaments(
        coils_file, width, thickness, toroidal_angles[-1] - toroidal_angles[0]
    )

    reordered_coils = reorder_coils(magnet_set)
    reordered_filaments = [coil.filament for coil in reordered_coils]
    build_magnet_surface(reordered_filaments, sample_mod=sample_mod)

    radial_distance_matrix = measure_surface_coils_separation(surface)

    return radial_distance_matrix
