from pathlib import Path
import subprocess

initialized = False


def check_cubit_installation():
    """Checks if Cubit is present on the Python module search path.

    Returns:
        (bool): flag indicating the presence of Cubit on the module search
            path.
    """
    try:
        import cubit

        return True
    except ImportError:
        return False


def init_cubit():
    """Initializes Coreform Cubit with the DAGMC plugin."""
    global cubit
    import cubit

    global initialized

    if not initialized:
        cubit.init(
            [
                "cubit",
                "-nojournal",
                "-nographics",
                "-information",
                "off",
                "-warning",
                "off",
            ]
        )
        initialized = True


def create_new_cubit_instance():
    """Creates new cubit instance checking if Cubit has been already
    initialized.

    """
    if initialized:
        cubit.cmd("new")
    else:
        init_cubit()


def import_step_cubit(filename, import_dir):
    """Imports STEP file into Coreform Cubit.

    Arguments:
        filename (str): name of STEP input file, excluding '.step' extension.
        import_dir (str): directory from which to import STEP file.

    Returns:
        vol_id (int): Cubit volume ID of imported CAD solid.
    """
    init_cubit()

    import_path = Path(import_dir) / Path(filename).with_suffix(".step")
    cubit.cmd(f'import step "{import_path}" heal')
    vol_id = cubit.get_last_id("volume")

    return vol_id


def export_step_cubit(filename, export_dir=""):
    """Export CAD solid as a STEP file via Coreform Cubit.

    Arguments:
        filename (str): name of STEP output file, excluding '.step' extension.
        export_dir (str): directory to which to export the STEP output file
            (defaults to empty string).
    """
    export_path = Path(export_dir) / Path(filename).with_suffix(".step")
    cubit.cmd(f'export step "{export_path}" overwrite')


def import_cub5_cubit(filename, import_dir):
    """Imports cub5 file with Coreform Cubit with default import settings.
    Arguments:
        filename (str): name of cub5 input file.
        import_dir (str): directory from which to import cub5 file.
    Returns:
        vol_id (int): Cubit volume ID of imported CAD solid.
    """
    init_cubit()
    import_path = Path(import_dir) / Path(filename).with_suffix(".cub5")
    cubit.cmd(
        f'import cubit "{import_path}" nofreesurfaces attributes_on separate_bodies'
    )
    vol_id = cubit.get_last_id("volume")
    return vol_id


def export_cub5(filename, export_dir=""):
    """Export cub5 representation of model (native Cubit format).

    Arguments:
        filename (str): name of cub5 output file, excluding '.cub5' extension.
        export_dir (str): directory to which to export the cub5 output file
            (defaults to empty string).
    """
    export_path = Path(export_dir) / Path(filename).with_suffix(".cub5")
    cubit.cmd(f'save cub5 "{export_path}" overwrite')


def export_mesh_cubit(filename, export_dir="", delete_upon_export=True):
    """Exports Cubit mesh to H5M file format, first exporting to Exodus format
    via Coreform Cubit and converting to H5M via MOAB.

    Arguments:
        filename (str): name of H5M output file, excluding '.h5m' extension.
        export_dir (str): directory to which to export the H5M output file
            (defaults to empty string).
        delete_upon_export (bool): delete the mesh from the Cubit instance
            after exporting. Prevents inclusion of mesh in future exports.
    """
    exo_path = Path(export_dir) / Path(filename).with_suffix(".exo")
    h5m_path = Path(export_dir) / Path(filename).with_suffix(".h5m")

    cubit.cmd(f'export mesh "{exo_path}" overwrite')
    subprocess.run(f"mbconvert {exo_path} {h5m_path}", shell=True)
    Path.unlink(exo_path)

    # Delete any meshes present to prevent inclusion in future Cubit mesh
    # exports
    if delete_upon_export:
        cubit.cmd(f"delete mesh volume all propagate")


def tag_surface(surface_id, tag):
    """Applies a boundary condition to a surface in cubit following the
    Coreform syntax.

    Arguments:
        surface_id (int): Surface to tag
        tag (str): boundary type
    """
    cubit.cmd(f"create sideset {surface_id}")
    cubit.cmd(f"sideset {surface_id} name 'boundary:{tag}'")
    cubit.cmd(f"sideset {surface_id} add surface {surface_id}")


def export_dagmc_cubit(
    filename="dagmc",
    export_dir="",
    anisotropic_ratio=100.0,
    deviation_angle=5.0,
    delete_upon_export=True,
):
    """Exports DAGMC neutronics H5M file of ParaStell components via Coreform
    Cubit.

    Arguments:
        anisotropic_ratio (float): controls edge length ratio of elements
            (defaults to 100.0).
        deviation_angle (float): controls deviation angle of facet from surface
            (i.e., lesser deviation angle results in more elements in areas
            with higher curvature) (defaults to 5.0).
        filename (str): name of DAGMC output file, excluding '.h5m' extension
            (defaults to 'dagmc').
        export_dir (str): directory to which to export the DAGMC output file
            (defaults to empty string).
        delete_upon_export (bool): delete the mesh from the Cubit instance
            after exporting. Prevents inclusion of mesh in future exports.
    """
    cubit.cmd(
        f"set trimesher coarse on ratio {anisotropic_ratio} "
        f"angle {deviation_angle}"
    )
    cubit.cmd("surface all scheme trimesh")
    cubit.cmd("mesh surface all")

    export_path = Path(export_dir) / Path(filename).with_suffix(".h5m")
    cubit.cmd(f'export dagmc "{export_path}" overwrite')

    # Delete any meshes present to prevent inclusion in future Cubit mesh
    # exports
    if delete_upon_export:
        cubit.cmd(f"delete mesh volume all propagate")


def import_geom_to_cubit(filename, import_dir=""):
    """Attempts to open a geometry file with the appropriate cubit_io function,
        based on file extension
    Arguments:
        filename (path): name of the file to import, including the suffix
        import_dir (str): directory from which to import the file.
    Returns:
        vol_id (int): Cubit volume ID of imported CAD solid.
    """
    importers = {
        ".step": import_step_cubit,
        ".cub5": import_cub5_cubit,
    }
    filename = Path(filename)
    vol_id = importers[filename.suffix](filename, import_dir)
    return vol_id


def make_material_block(mat_tag, block_id, vol_id_str):
    """Issue commands to make a material block in Cubit.

    Arguments:
       mat_tag (str) : name of material block
       block_id (int) : block number
       vol_id_str (str) : space-separated list of volume ids
    """
    cubit.cmd("set duplicate block elements off")
    cubit.cmd(f'create material "{mat_tag}" property_group ' '"CUBIT-ABAQUS"')
    cubit.cmd(f"block {block_id} add volume {vol_id_str}")
    cubit.cmd(f'block {block_id} material "{mat_tag}"')


def imprint_and_merge():
    """Imprints and merges all volumes in Cubit."""
    cubit.cmd("imprint volume all")
    cubit.cmd("merge volume all")


def orient_spline_surfaces(volume_id):
    """Extracts the inner and outer surface IDs for a given ParaStell in-vessel
    component volume in Coreform Cubit.

    Arguments:
        volume_id (int): Cubit volume ID.

    Returns:
        inner_surface_id (int): Cubit ID of in-vessel component inner surface.
        outer_surface_id (int): Cubit ID of in-vessel component outer surface.
    """

    surfaces = cubit.get_relatives("volume", volume_id, "surface")

    spline_surfaces = []
    for surface in surfaces:
        if cubit.get_surface_type(surface) == "spline surface":
            spline_surfaces.append(surface)

    if len(spline_surfaces) == 1:
        outer_surface_id = spline_surfaces[0]
        inner_surface_id = None
    else:
        # The outer surface bounding box will have the larger maximum XY value
        if (
            cubit.get_bounding_box("surface", spline_surfaces[1])[4]
            > cubit.get_bounding_box("surface", spline_surfaces[0])[4]
        ):
            outer_surface_id = spline_surfaces[1]
            inner_surface_id = spline_surfaces[0]
        else:
            outer_surface_id = spline_surfaces[0]
            inner_surface_id = spline_surfaces[1]

    return inner_surface_id, outer_surface_id


def merge_surfaces(surface_1, surface_2):
    """Merges two surfaces in Cubit.

    Arguments:
        surface_1 (int): Cubit ID of one surface to be merged.
        surface_2 (int): Cubit ID of the other surface to be merged.
    """
    cubit.cmd(f"merge surface {surface_1} {surface_2}")


def mesh_volume_auto_factor(volume_ids, mesh_size=5.0):
    """Meshes a volume in Cubit using automatically calculated interval sizes.

    Arguments:
        volume_ids (iterable of int): Cubit IDs of volumes to be meshed.
        mesh_size (float): controls the size of the mesh. Takes values between
            1.0 (finer) and 10.0 (coarser) (optional, defaults to 5.0).
    """
    volume_ids_str = " ".join(str(id) for id in volume_ids)

    cubit.cmd(f"volume {volume_ids_str} scheme tetmesh")
    cubit.cmd(f"volume {volume_ids_str} size auto factor {mesh_size}")
    cubit.cmd(f"mesh volume {volume_ids_str}")


def mesh_volume_skeleton(
    volume_ids, min_size=20.0, max_size=50.0, max_gradient=1.5
):
    """Meshes a volume in Cubit using the skeleton sizing function.

    Arguments:
        volume_ids (iterable of int): Cubit IDs of volumes to be meshed.
        min_size (float): minimum size of mesh elements (defaults to 20.0).
        max_size (float): maximum size of mesh elements (defaults to 50.0).
        max_gradient (float): maximum transition in mesh element size
            (defaults to 1.5).
    """
    volume_ids_str = " ".join(str(id) for id in volume_ids)

    cubit.cmd(f"volume {volume_ids_str} scheme tetmesh")
    cubit.cmd(
        f"volume {volume_ids_str} sizing function type skeleton min_size "
        f"{min_size} max_size {max_size} max_gradient {max_gradient} "
        "min_num_layers_3d 1 min_num_layers_2d 1 min_num_layers_1d 1"
    )
    cubit.cmd(f"mesh volume {volume_ids_str}")


def get_last_id(entity):
    """Returns the ID of the most recently created entity of a given type.

    Arguments:
        entity (str): the type of entity for which the ID should be retrieved.
            Valid arguments include "vertex", "curve", "surface", and "volume".

    Returns:
        (int): the ID of the most recently created entity of the given type.
    """
    return cubit.get_last_id(entity)
