import argparse
from pathlib import Path
from abc import ABC

import numpy as np

import cadquery as cq

from . import log
from .cubit_utils import (
    import_geom_to_cubit,
    export_mesh_cubit,
    mesh_volume_skeleton,
    get_last_id,
)

# Import cubit_utils separately for its initialized variable. If initialized is
# imported into this namespace, changes to the variable do not persist when
# modified by calls to the imported functions
from . import cubit_utils
from .utils import read_yaml_config, filter_kwargs, reorder_loop, m2cm

export_allowed_kwargs = ["step_filename", "export_mesh", "mesh_filename"]


class MagnetSet(ABC):
    """An object representing a set of modular stellarator magnet coils.

    Arguments:
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        mat_tag (str): DAGMC material tag to use for magnets in DAGMC
            neutronics model (defaults to 'magnets').
    """

    def __init__(
        self,
        logger=None,
        **kwargs,
    ):

        self.logger = logger

        self.mat_tag = "magnets"

        for name in kwargs.keys() & ("mat_tag",):
            self.__setattr__(name, kwargs[name])

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    def import_geom_cubit(self):
        """Import geometry file for magnet set into Coreform Cubit."""
        first_vol_id = 1
        if cubit_utils.initialized:
            first_vol_id += get_last_id("volume")

        last_vol_id = import_geom_to_cubit(
            self.geometry_file, self.working_dir
        )
        self.volume_ids = list(range(first_vol_id, last_vol_id + 1))

    def mesh_magnets(self, min_size=20.0, max_size=50.0, max_gradient=1.5):
        """Creates tetrahedral mesh of magnet volumes via Coreform Cubit.

        Arguments:
            min_size (float): minimum size of mesh elements (defaults to 20.0).
            max_size (float): maximum size of mesh elements (defaults to 50.0).
            max_gradient (float): maximum transition in mesh element size
                (defaults to 1.5).
        """
        self._logger.info("Generating tetrahedral mesh of magnet coils...")

        if not hasattr(self, "volume_ids"):
            self.import_geom_cubit()

        mesh_volume_skeleton(
            self.volume_ids,
            min_size=min_size,
            max_size=max_size,
            max_gradient=max_gradient,
        )

    def export_mesh(self, mesh_filename="magnet_mesh", export_dir=""):
        """Creates tetrahedral mesh of magnet volumes and exports H5M format
        via Coreform Cubit and  MOAB.

        Arguments:
            mesh_filename (str): name of H5M output file, excluding '.h5m'
                extension (optional, defaults to 'magnet_mesh').
            export_dir (str): directory to which to export the H5M output file
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting mesh H5M file for magnet coils...")

        export_mesh_cubit(filename=mesh_filename, export_dir=export_dir)


class MagnetSetFromFilaments(MagnetSet):
    """Inherits from MagnetSet. This subclass enables the construction of
    CAD solids using the filament data and CADquery.

    Arguments:
        coils_file (str): path to coil filament data file.
        width (float): width of coil cross-section in toroidal direction [cm].
        thickness (float): thickness of coil cross-section in radial direction
            [cm].
        toroidal_extent (float): toroidal extent to model [deg].
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        start_line (int): starting line index for data in filament data file
            (defaults to 3).
        sample_mod (int): sampling modifier for filament points (defaults to
            1). For a user-defined value n, every nth point will be sampled.
        scale (float): a scaling factor between the units of the point-locus
            data and [cm] (defaults to m2cm = 100).
        mat_tag (str): DAGMC material tag to use for magnets in DAGMC
            neutronics model (defaults to 'magnets').
    """

    def __init__(
        self,
        coils_file,
        width,
        thickness,
        toroidal_extent,
        logger=None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)

        self.coils_file = coils_file

        self.start_line = 3
        self.sample_mod = 1
        self.scale = m2cm

        self.width = width
        self.thickness = thickness
        self.toroidal_extent = toroidal_extent

        # Define maximum length of coil cross-section
        self.max_cs_len = max(self._width, self._thickness)

        for name in kwargs.keys() & (
            "start_line",
            "sample_mod",
            "scale",
            "mat_tag",
        ):
            self.__setattr__(name, kwargs[name])

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        if self._width < 0.0:
            e = ValueError("Coil cross-section width cannot be negative.")
            self._logger.error(e.args[0])
            raise e

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
        if self._thickness < 0.0:
            e = ValueError("Coil cross-section thickness cannot be negative.")
            self._logger.error(e.args[0])
            raise e

    @property
    def toroidal_extent(self):
        return self._toroidal_extent

    @toroidal_extent.setter
    def toroidal_extent(self, angle):
        self._toroidal_extent = np.deg2rad(angle)
        if self._toroidal_extent > 360.0:
            e = ValueError("Toroidal extent cannot exceed 360.0 degrees.")
            self._logger.error(e.args[0])
            raise e

    def _instantiate_filaments(self):
        """Extracts filament coordinate data from input data file and
        instantiates Filament class objects.
        (Internal function not intended to be called externally)
        """
        with open(self.coils_file, "r") as file:
            data = file.readlines()[self.start_line :]

        coords = []
        self.filaments = []

        for line in data:
            columns = line.strip().split()

            if columns[0] == "end":
                break

            # Coil current
            s = float(columns[3])

            # s = 0 signals end of filament
            if s != 0:
                coords.append(
                    [float(ord) * self.scale for ord in columns[0:3]]
                )

            else:
                coords.append(coords[0])
                self.filaments.append(Filament(np.array(coords)))
                coords.clear()

    def sort_filaments_toroidally(self):
        """Reorders list of filaments by toroidal angle on range [-pi, pi].

        Returns:
            (list of object): sorted list of Filament class objects.
        """
        return sorted(self.filaments, key=lambda x: x.com_toroidal_angle)

    def _filter_filaments(self, tol=0):
        """Filters list of Filament objects such that only those within the
        toroidal extent of the model are included and filaments are sorted by
        center-of-mass toroidal angle.
        (Internal function not intended to be called externally)
        """

        # Compute lower and upper bounds of toroidal extent within tolerance
        lower_bound = 2 * np.pi - tol
        upper_bound = self._toroidal_extent + tol

        # Create filter determining whether each coil lies within model's
        # toroidal extent
        filtered_filaments = [
            filament
            for filament in self.filaments
            if filament.in_toroidal_extent(lower_bound, upper_bound)
        ]
        self.filaments = filtered_filaments

        # Sort coils by center-of-mass toroidal angle and overwrite stored list
        self.filaments = self.sort_filaments_toroidally()

    def _instantiate_coils(self):
        """Instantiates MagnetCoil class objects using filament data.
        (Internal function not intended to be called externally)
        """
        self.magnet_coils = []
        for filament in self.filaments:
            self.magnet_coils.append(
                MagnetCoil(
                    filament, self.width, self.thickness, self.sample_mod
                )
            )

    def _compute_radial_distance_data(self):
        """Computes average and maximum radial distance of filament points.
        (Internal function not intended to be called externally)
        """
        radii_count = 0
        self.average_radial_distance = 0
        self.max_radial_distance = -1

        for filament in self.filaments:
            radii = np.linalg.norm(filament.coords[:-1, :2], axis=1)
            radii_count += len(radii)
            self.average_radial_distance += np.sum(radii)
            self.max_radial_distance = max(
                self.max_radial_distance, np.max(radii)
            )

        self.average_radial_distance /= radii_count

    def _cut_magnets(self):
        """Cuts the magnets at the planes defining the toriodal extent.
        (Internal function not intended to be called externally)
        """
        side_length = 1.25 * self.max_radial_distance

        toroidal_region = cq.Workplane("XZ")
        toroidal_region = toroidal_region.transformed(
            offset=(side_length / 2, 0)
        )
        toroidal_region = toroidal_region.rect(side_length, side_length)
        toroidal_region = toroidal_region.revolve(
            np.rad2deg(self._toroidal_extent),
            (-side_length / 2, 0),
            (-side_length / 2, 1),
        )
        toroidal_region = toroidal_region.val()

        for coil in self.magnet_coils:
            cut_coil = coil.solid.intersect(toroidal_region)
            coil.solid = cut_coil

    def export_step(self, step_filename="magnet_set.step", export_dir=""):
        """Export CAD solids as a STEP file via CadQuery.

        Arguments:
            step_filename (str): name of STEP output file (optional, defaults
                to 'magnet_set.step').
            export_dir (str): directory to which to export the STEP output file
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting STEP file for magnet coils...")

        self.working_dir = export_dir
        self.geometry_file = Path(step_filename).with_suffix(".step")

        export_path = Path(self.working_dir) / self.geometry_file

        coil_set = cq.Compound.makeCompound(
            [coil.solid for coil in self.magnet_coils]
        )
        cq.exporters.export(coil_set, str(export_path))

    def populate_magnet_coils(self):
        """Populates MagnetCoil class objects representing each of the magnetic
        coils that lie within the specified toroidal extent.
        """
        self._logger.info("Populating magnet coils...")

        self._instantiate_filaments()
        self._compute_radial_distance_data()
        # Define tolerance of toroidal extent to account for dimensionality of
        # coil cross-section
        # Multiply by factor of 2 to be conservative
        tol = 2 * np.arctan2(self.max_cs_len, self.average_radial_distance)
        self._filter_filaments(tol=tol)
        self._instantiate_coils()

    def build_magnet_coils(self):
        """Builds each filament in self.filtered_filaments in cubit, then cuts
        to the toroidal extent using self._cut_magnets().
        """
        self._logger.info("Constructing magnet coils...")

        [magnet_coil.create_magnet() for magnet_coil in self.magnet_coils]

        self._cut_magnets()

        self.coil_solids = [coil.solid for coil in self.magnet_coils]


class MagnetSetFromGeometry(MagnetSet):
    """An object representing a set of modular stellarator magnet coils
    with previously defined geometry files.

    Arguments:
        geometry_file (str): path to the existing coil geometry. Can be of
            the types supported by cubit_io.import_geom_to_cubit(). For
            cad_to_dagmc, only step files are supported.
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        mat_tag (str): DAGMC material tag to use for magnets in DAGMC
            neutronics model (defaults to 'magnets').
    """

    def __init__(
        self,
        geometry_file,
        logger=None,
        **kwargs,
    ):
        super().__init__(logger, **kwargs)
        self.geometry_file = Path(geometry_file).resolve()
        self.working_dir = self.geometry_file.parent
        self._coil_solids = None

        for name in kwargs.keys() & (
            "start_line",
            "sample_mod",
            "scale",
            "mat_tag",
        ):
            self.__setattr__(name, kwargs[name])

    @property
    def coil_solids(self):
        if self._coil_solids is None:
            self._coil_solids = (
                cq.importers.importStep(str(self.geometry_file)).val().Solids()
            )
        return self._coil_solids


class Filament(object):
    """Object containing basic data defining a Filament, and necessary
    functions for working with that data.

    Arguments:
        coords (2-D array of float): set of Cartesian coordinates defining
            magnet filament location.
    """

    def __init__(self, coords):
        self.coords = coords

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, data):
        self._coords = data

        tangents = np.subtract(
            np.append(data[1:], [data[1]], axis=0),
            np.append([data[-2]], data[0:-1], axis=0),
        )
        self.tangents = (
            tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        )

        self.com = np.average(data[:-1], axis=0)
        self.com_toroidal_angle = np.arctan2(self.com[1], self.com[0])

    def get_ob_mp_index(self):
        """Finds the index of the outboard midplane coordinate on a coil
        filament.

        Returns:
            outboard_index (int): index of the outboard midplane point.
        """
        # Compute radial distance of coordinates from z-axis
        radii = np.linalg.norm(self.coords[:, :2], axis=1)
        # Determine whether adjacent points cross the midplane (if so, they will
        # have opposite signs)
        shifted_coords = np.append(self.coords[1:], [self.coords[1]], axis=0)
        midplane_flags = -np.sign(self.coords[:, 2] * shifted_coords[:, 2])
        # Find index of outboard midplane point
        outboard_index = np.argmax(midplane_flags * radii)

        return outboard_index

    def reorder_coords(self, index):
        """Reorders coil filament coordinate loop about a given index.

        Arguments:
            index (int): index about which to reorder coordinate loop.
        """
        self.coords = reorder_loop(self.coords, index)

    def orient_coords(self, positive=True):
        """Orients coil filament coordinate loop such that they initially
        progress positively or negatively.

        Arguments:
            positive (bool): progress coordinates in positive direciton
                (defaults to True). If negative, coordinates will progress in
                negative direction.
        """
        if positive == (self.coords[0, 2] > self.coords[1, 2]):
            self.coords = np.flip(self.coords, axis=0)

    def in_toroidal_extent(self, lower_bound, upper_bound):
        """Determines if the coil lies within a given toroidal angular extent,
        based on filament coordinates.

        Arguments:
            lower_bound (float): lower bound of toroidal extent [rad].
            upper_bound (float): upper bound of toroidal extent [rad].

        Returns:
            in_toroidal_extent (bool): flag to indicate whether coil lies
                within toroidal bounds.
        """
        # Compute toroidal angle of each point in filament
        toroidal_angles = np.arctan2(self.coords[:, 1], self.coords[:, 0])
        # Ensure angles are positive
        toroidal_angles = (toroidal_angles + 2 * np.pi) % (2 * np.pi)
        # Compute bounds of toroidal extent of filament
        min_tor_ang = np.min(toroidal_angles)
        max_tor_ang = np.max(toroidal_angles)

        # Determine if filament toroidal extent overlaps with that of model
        if (min_tor_ang >= lower_bound or min_tor_ang <= upper_bound) or (
            max_tor_ang >= lower_bound or max_tor_ang <= upper_bound
        ):
            in_toroidal_extent = True
        else:
            in_toroidal_extent = False

        return in_toroidal_extent


class MagnetCoil(object):
    """An object representing a single modular stellarator magnet coil.

    Arguments:
        filament (Filament object): filament definining the location of the
            coil
        width (float): width of coil cross-section in toroidal direction [cm].
        thickness (float): thickness of coil cross-section in radial direction
            [cm].
        sample_mod (int): Length of stride when sampling from coordinate data.
    """

    def __init__(self, filament, width, thickness, sample_mod):

        self.filament = filament
        self.sample_mod = sample_mod
        self.coords = filament.coords
        self.center_of_mass = filament.com
        self.tangents = filament.tangents
        self.width = width
        self.thickness = thickness

    def create_magnet(self):
        """Creates a single magnet coil CAD solid in CadQuery.

        Returns:
            coil (object): cq.Solid object representing a single magnet coil.
        """
        # Sample filament coordinates and tangents by modifier
        coords = self.coords[0 : -1 : self.sample_mod]
        coords = np.append(coords, [self.coords[0]], axis=0)
        tangents = self.tangents[0 : -1 : self.sample_mod]
        tangents = np.append(tangents, [self.tangents[0]], axis=0)

        tangent_vectors = [cq.Vector(tuple(tangent)) for tangent in tangents]

        # Define coil filament path normals such that they face the filament
        # center of mass
        # Compute "outward" direction as difference between filament positions
        # and filament center of mass
        outward_dirs = coords - self.center_of_mass
        outward_dirs = (
            outward_dirs / np.linalg.norm(outward_dirs, axis=1)[:, np.newaxis]
        )

        # Project outward directions onto desired coil cross-section (CS) plane
        # at each filament position to define filament path normals
        parallel_parts = np.diagonal(
            np.matmul(outward_dirs, tangents.transpose())
        )

        normals = outward_dirs - parallel_parts[:, np.newaxis] * tangents
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        # Compute binormals projected onto CS plane at each position
        binormals = np.cross(tangents, normals)

        # Compute coordinates of edges of rectangular coils
        edge_offsets = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])

        coil_edge_coords = []
        for edge_offset in edge_offsets:
            coil_edge = (
                coords
                + edge_offset[0] * binormals * (self.width / 2)
                + edge_offset[1] * normals * (self.thickness / 2)
            )

            coil_edge_coords.append(
                [cq.Vector(tuple(pos)) for pos in coil_edge]
            )

        # Append first edge once again
        coil_edge_coords.append(coil_edge_coords[0])

        coil_edges = [
            cq.Edge.makeSpline(coord_vectors, tangents=tangent_vectors).close()
            for coord_vectors in coil_edge_coords
        ]

        face_list = [
            cq.Face.makeRuledSurface(edge1, edge2)
            for edge1, edge2 in zip(coil_edges[:-1], coil_edges[1:])
        ]

        shell = cq.Shell.makeShell(face_list)
        self.solid = cq.Solid.makeSolid(shell)


def parse_args():
    """Parser for running as a script"""
    parser = argparse.ArgumentParser(prog="magnet_coils")

    parser.add_argument(
        "filename", help="YAML file defining ParaStell magnet configuration"
    )
    parser.add_argument(
        "-e",
        "--export_dir",
        default="",
        help=(
            "Directory to which output files are exported (default: working "
            "directory)"
        ),
        metavar="",
    )
    parser.add_argument(
        "-l",
        "--logger",
        default=False,
        help=(
            "Flag to indicate whether to instantiate a logger object (default: "
            "False)"
        ),
        metavar="",
    )

    return parser.parse_args()


def generate_magnet_set():
    """Main method when run as command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger == True:
        logger = log.init()
    else:
        logger = log.NullLogger()

    magnet_coils_dict = all_data["magnet_coils"]

    magnet_set = MagnetSet(
        magnet_coils_dict["coils_file"],
        magnet_coils_dict["cross_section"],
        magnet_coils_dict["toroidal_extent"],
        logger=logger**magnet_coils_dict,
    )

    magnet_set.build_magnet_coils()

    magnet_set.export_step(
        export_dir=args.export_dir,
        **(filter_kwargs(magnet_coils_dict, ["step_filename"])),
    )

    if magnet_coils_dict["export_mesh"]:
        magnet_set.export_mesh(
            export_dir=args.export_dir,
            **(filter_kwargs(magnet_coils_dict, ["mesh_filename"])),
        )


if __name__ == "__main__":
    generate_magnet_set()
