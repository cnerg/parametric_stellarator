import argparse
from pathlib import Path

import numpy as np
from pymoab import core, types
import pystell.read_vmec as read_vmec

from . import log as log
from .utils import read_yaml_config, filter_kwargs, m2cm, m3tocm3

export_allowed_kwargs = ["filename"]


def default_reaction_rate(n_i, T_i):
    """Default reaction rate formula for DT fusion assumes an equal mixture of
    D and T in a hot plasma. From A. Bader et al 2021 Nucl. Fusion 61 116060
    DOI 10.1088/1741-4326/ac2991


    Arguments:
        n_i (float) : ion density (ions per m3)
        T_i (float) : ion temperature (KeV)

    Returns:
        rr (float) : reaction rate in reactions/cm3/s. Equates to neutron source
            density.
    """
    if T_i == 0 or n_i == 0:
        return 0

    rr = (
        3.68e-18
        * (n_i**2)
        / 4
        * T_i ** (-2 / 3)
        * np.exp(-19.94 * T_i ** (-1 / 3))
    )

    return rr / m3tocm3


def default_plasma_conditions(s):
    """Calculates ion density and temperature as a function of the
    plasma paramter s using profiles found in A. Bader et al 2021 Nucl. Fusion
    61 116060 DOI 10.1088/1741-4326/ac2991

    Arguments:
        s (float): closed magnetic flux surface index in range of 0 (magnetic
            axis) to 1 (plasma edge).

    Returns:
        n_i (float) : ion density in ions/m3
        T_i (float) : ion temperature in KeV
    """

    # Temperature
    T_i = 11.5 * (1 - s)
    # Ion density
    n_i = 4.8e20 * (1 - s**5)

    return n_i, T_i


class SourceMesh(object):
    """Generates a source mesh that describes the relative source intensity of
    neutrons in a magnetically confined plasma described by a VMEC plasma
    equilibrium.

    The mesh will be defined on a regular grid in the plasma coordinates of s,
    theta, phi.  Mesh vertices will be defined on circular grid at each toroidal
    plane, and connected between toroidal planes. This results in wedge elements
    along the magnetic axis and hexagonal elements throughout the remainder of
    the mesh.  Each of these elements will be subdivided into tetrahedra (4 for
    the wedges and 5 for the hexahedra) to result in a mesh that is simpler to
    use.

    Each tetrahedron will be tagged with the volumetric neutron source intensity
    in n/cm3/s, using on a finite-element based quadrature of the source
    intensity evaluated at each vertex.

    Arguments:
        vmec_obj (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
        cfs_grid (iterable of float): mesh grid points along closed flux
            surface (CFS) dimension of flux space.
        poloidal_grid (iterable of float): mesh grid points along poloidal
            angle dimension of flux space [deg].
        toroidal_grid (iterable of float): mesh grid points along toroidal
            angle dimension of flux space [deg].
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        scale (float): a scaling factor between the units of VMEC and [cm]
            (defaults to m2cm = 100).
        plasma_conditions (function): function that takes the plasma parameter
            s, and returns temperature and ion density with suitable units for
            the reaction_rate() function. Defaults to
            default_plasma_conditions()
        reaction_rate (function): function that takes the values returned by
            plasma_conditions() and returns a reaction rate in reactions/cm3/s
    """

    def __init__(
        self,
        vmec_obj,
        cfs_grid,
        poloidal_grid,
        toroidal_grid,
        logger=None,
        **kwargs
    ):

        self.logger = logger
        self.vmec_obj = vmec_obj
        self.cfs_grid = cfs_grid
        self.poloidal_grid = poloidal_grid
        self.toroidal_grid = toroidal_grid

        self.scale = m2cm
        self.plasma_conditions = default_plasma_conditions
        self.reaction_rate = default_reaction_rate

        for name in kwargs.keys() & (
            "scale",
            "plasma_conditions",
            "reaction_rate",
        ):
            self.__setattr__(name, kwargs[name])

        self.strengths = []
        self.volumes = []

        self._create_mbc()

    @property
    def cfs_grid(self):
        return self._cfs_grid

    @cfs_grid.setter
    def cfs_grid(self, array):
        self._cfs_grid = array
        if self._cfs_grid[0] != 0 or self._cfs_grid[-1] != 1:
            e = ValueError("CFS grid values must span the range [0, 1].")
            self._logger.error(e.args[0])
            raise e

    @property
    def poloidal_grid(self):
        return self._poloidal_grid

    @poloidal_grid.setter
    def poloidal_grid(self, array):
        self._poloidal_grid = np.deg2rad(array)
        if self._poloidal_grid[-1] - self._poloidal_grid[0] != 360.0:
            e = ValueError(
                "Poloidal extent spanned by poloidal_grid must be exactly 360 "
                "degrees."
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def toroidal_grid(self):
        return self._toroidal_grid

    @toroidal_grid.setter
    def toroidal_grid(self, array):
        self._toroidal_grid = np.deg2rad(array)
        if self._toroidal_grid[-1] - self._toroidal_grid[0] > 360.0:
            e = ValueError(
                "Toroidal extent spanned by toroidal_grid cannot exceed 360 "
                "degrees."
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    def _create_mbc(self):
        """Creates PyMOAB core instance with source strength tag.
        (Internal function not intended to be called externally)
        """
        self.mbc = core.Core()

        tag_type = types.MB_TYPE_DOUBLE
        tag_size = 1
        storage_type = types.MB_TAG_DENSE

        ss_tag_name = "Source Strength"
        self.source_strength_tag = self.mbc.tag_get_handle(
            ss_tag_name,
            tag_size,
            tag_type,
            storage_type,
            create_if_missing=True,
        )

        vol_tag_name = "Volume"
        self.volume_tag = self.mbc.tag_get_handle(
            vol_tag_name,
            tag_size,
            tag_type,
            storage_type,
            create_if_missing=True,
        )

    def create_vertices(self):
        """Creates mesh vertices and adds them to PyMOAB core.

        The grid of mesh vertices is generated from the user input
        defining the number of meshes in each of the plasma
        coordinate directions. Care is taken to manage the
        mesh at the 0 == 2 * pi wrap so that everything
        is closed and consistent.
        """
        self._logger.info("Computing source mesh point cloud...")

        phi_list = np.linspace(0, self._toroidal_extent, num=self.num_phi)
        # don't include magnetic axis in list of s values
        s_list = np.linspace(0.0, 1.0, num=self.num_s)[1:]
        # don't include repeated entry at 0 == 2*pi
        theta_list = np.linspace(0, 2 * np.pi, num=self.num_theta)[:-1]

        # don't include repeated entry at 0 == 2*pi
        if self._toroidal_extent == 2 * np.pi:
            phi_list = phi_list[:-1]

        self.verts_per_ring = theta_list.shape[0]
        # add one vertex per plane for magenetic axis
        self.verts_per_plane = s_list.shape[0] * self.verts_per_ring + 1

        num_verts = phi_list.shape[0] * self.verts_per_plane
        self.coords = np.zeros((num_verts, 3))
        self.coords_s = np.zeros(num_verts)

        # Initialize vertex index
        vert_idx = 0

        for phi in phi_list:
            # vertex coordinates on magnetic axis
            self.coords[vert_idx, :] = (
                np.array(self.vmec_obj.vmec2xyz(0, 0, phi)) * self.scale
            )
            self.coords_s[vert_idx] = 0

            vert_idx += 1

            # vertex coordinate away from magnetic axis
            for s in s_list:
                for theta in theta_list:
                    self.coords[vert_idx, :] = (
                        np.array(self.vmec_obj.vmec2xyz(s, theta, phi))
                        * self.scale
                    )
                    self.coords_s[vert_idx] = s

                    vert_idx += 1

        self.verts = self.mbc.create_vertices(self.coords)

    def _source_strength(self, tet_ids):
        """Computes neutron source strength for a tetrahedron using five-node
        Gaussian quadrature.
        (Internal function not intended to be called externally)

        Arguments:
            ids (list of int): tetrahedron vertex indices.

        Returns:
            ss (float): integrated source strength for tetrahedron.
        """

        # Initialize list of vertex coordinates for each tetrahedron vertex
        tet_coords = [self.coords[id] for id in tet_ids]

        # Initialize list of source strengths for each tetrahedron vertex
        vertex_strengths = [
            self.reaction_rate(*self.plasma_conditions(self.coords_s[id]))
            for id in tet_ids
        ]

        # Define barycentric coordinates for integration points
        bary_coords = np.array(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.5, 1 / 6, 1 / 6, 1 / 6],
                [1 / 6, 0.5, 1 / 6, 1 / 6],
                [1 / 6, 1 / 6, 0.5, 1 / 6],
                [1 / 6, 1 / 6, 1 / 6, 0.5],
            ]
        )

        # Define weights for integration points
        int_w = np.array([-0.8, 0.45, 0.45, 0.45, 0.45])

        # Interpolate source strength at integration points
        ss_int_pts = np.dot(bary_coords, vertex_strengths)

        # Compute edge vectors between tetrahedron vertices
        edge_vectors = np.subtract(tet_coords[:3], tet_coords[3]).T

        tet_vol = -np.linalg.det(edge_vectors) / 6

        ss = np.abs(tet_vol) * np.dot(int_w, ss_int_pts)

        return ss, tet_vol

    def _create_tet(self, tet_ids):
        """Creates tetrahedron and adds to pyMOAB core.
        (Internal function not intended to be called externally)

        Arguments:
            tet_ids (list of int): tetrahedron vertex indices.
        """

        tet_verts = [self.verts[int(id)] for id in tet_ids]
        tet = self.mbc.create_element(types.MBTET, tet_verts)
        self.mbc.add_entity(self.mesh_set, tet)

        # Compute source strength for tetrahedron
        ss, vol = self._source_strength(tet_ids)
        self.strengths.append(ss)
        self.volumes.append(vol)

        # Tag tetrahedra with data
        self.mbc.tag_set_data(self.source_strength_tag, tet, [ss])
        self.mbc.tag_set_data(self.volume_tag, tet, [vol])

    def _get_vertex_id(self, vertex_idx):
        """Computes vertex index in row-major order as stored by MOAB from
        three-dimensional n x 3 matrix indices.
        (Internal function not intended to be called externally)

        Arguments:
            vert_idx (list of int): list of vertex
                [flux surface index, poloidal angle index, toroidal angle index]

        Returns:
            id (int): vertex index in row-major order as stored by MOAB
        """

        s_idx, theta_idx, phi_idx = vertex_idx

        ma_offset = phi_idx * self.verts_per_plane

        # Wrap around if final plane and it is 2*pi
        if self._toroidal_extent == 2 * np.pi and phi_idx == self.num_phi - 1:
            ma_offset = 0

        # Compute index offset from closed flux surface
        s_offset = s_idx * self.verts_per_ring

        theta_offset = theta_idx

        # Wrap around if theta is 2*pi
        if theta_idx == self.num_theta:
            theta_offset = 1

        id = ma_offset + s_offset + theta_offset

        return id

    def _create_tets_from_hex(self, s_idx, theta_idx, phi_idx):
        """Creates five tetrahedra from defined hexahedron.
        (Internal function not intended to be called externally)

        Arguments:
            idx_list (list of int): list of hexahedron vertex indices.
        """

        # relative offsets of vertices in a 3-D index space
        hex_vertex_stencil = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )

        # Ids of hex vertices applying offset stencil to current point
        hex_idx_data = (
            np.array([s_idx, theta_idx, phi_idx]) + hex_vertex_stencil
        )

        idx_list = [
            self._get_vertex_id(vertex_idx) for vertex_idx in hex_idx_data
        ]

        # Define MOAB canonical ordering of hexahedron vertex indices
        # Ordering follows right hand rule such that the fingers curl around
        # one side of the tetrahedron and the thumb points to the remaining
        # vertex. The vertices are ordered such that those on the side are
        # first, ordered clockwise relative to the thumb, followed by the
        # remaining vertex at the end of the thumb.
        # See Moreno, Bader, Wilson 2024 for hexahedron splitting
        # Conditionally alternate ordering of vertices defining hexahedron
        # splitting to avoid gaps and overlaps between non-planar hexahedron
        # faces
        if self.alt_flag:
            hex_canon_ids = [
                [idx_list[0], idx_list[2], idx_list[1], idx_list[5]],
                [idx_list[0], idx_list[3], idx_list[2], idx_list[7]],
                [idx_list[0], idx_list[7], idx_list[5], idx_list[4]],
                [idx_list[7], idx_list[2], idx_list[5], idx_list[6]],
                [idx_list[0], idx_list[2], idx_list[5], idx_list[7]],
            ]
        else:
            hex_canon_ids = [
                [idx_list[0], idx_list[3], idx_list[1], idx_list[4]],
                [idx_list[1], idx_list[3], idx_list[2], idx_list[6]],
                [idx_list[1], idx_list[4], idx_list[6], idx_list[5]],
                [idx_list[3], idx_list[6], idx_list[4], idx_list[7]],
                [idx_list[1], idx_list[3], idx_list[6], idx_list[4]],
            ]

        for vertex_ids in hex_canon_ids:
            self._create_tet(vertex_ids)

    def _create_tets_from_wedge(self, theta_idx, phi_idx):
        """Creates three tetrahedra from defined wedge.
        (Internal function not intended to be called externally)

        Arguments:
            idx_list (list of int): list of wedge vertex indices.
        """

        # relative offsets of wedge vertices in a 3-D index space
        wedge_vertex_stencil = np.array(
            [
                [0, 0, 0],
                [0, theta_idx, 0],
                [0, theta_idx + 1, 0],
                [0, 0, 1],
                [0, theta_idx, 1],
                [0, theta_idx + 1, 1],
            ]
        )

        # Ids of wedge vertices applying offset stencil to current point
        wedge_idx_data = np.array([0, 0, phi_idx]) + wedge_vertex_stencil

        idx_list = [
            self._get_vertex_id(vertex_idx) for vertex_idx in wedge_idx_data
        ]

        # Define MOAB canonical ordering of wedge vertex indices
        # Ordering follows right hand rule such that the fingers curl around
        # one side of the tetrahedron and the thumb points to the remaining
        # vertex. The vertices are ordered such that those on the side are
        # first, ordered clockwise relative to the thumb, followed by the
        # remaining vertex at the end of the thumb.
        # See Moreno, Bader, Wilson 2024 for wedge splitting
        # Conditionally alternate ordering of vertices defining wedge splitting
        # to avoid gaps and overlaps between non-planar wedge faces
        if self.alt_flag:
            wedge_canon_ids = [
                [idx_list[0], idx_list[2], idx_list[1], idx_list[3]],
                [idx_list[1], idx_list[3], idx_list[5], idx_list[4]],
                [idx_list[1], idx_list[3], idx_list[2], idx_list[5]],
            ]
        else:
            wedge_canon_ids = [
                [idx_list[0], idx_list[2], idx_list[1], idx_list[3]],
                [idx_list[3], idx_list[2], idx_list[4], idx_list[5]],
                [idx_list[3], idx_list[2], idx_list[1], idx_list[4]],
            ]

        for vertex_ids in wedge_canon_ids:
            self._create_tet(vertex_ids)

    def create_mesh(self):
        """Creates volumetric source mesh in real space."""
        self._logger.info("Constructing source mesh...")

        self.mesh_set = self.mbc.create_meshset()
        self.mbc.add_entity(self.mesh_set, self.verts)

        for phi_idx in range(self.num_phi - 1):
            # Set alternation flag to true at beginning of each toroidal block
            self.alt_flag = True
            # Create tetrahedra for wedges at center of plasma
            for theta_idx in range(1, self.num_theta):
                self._create_tets_from_wedge(theta_idx, phi_idx)
                self.alt_flag = not self.alt_flag

            # Create tetrahedra for hexahedra beyond center of plasma
            for s_idx in range(self.num_s - 2):
                for theta_idx in range(1, self.num_theta):
                    self._create_tets_from_hex(s_idx, theta_idx, phi_idx)
                    self.alt_flag = not self.alt_flag

    def export_mesh(self, filename="source_mesh", export_dir=""):
        """Use PyMOAB interface to write source mesh with source strengths
        tagged.

        Arguments:
            filename: name of H5M output file, excluding '.h5m' extension
                (optional, defaults to 'source_mesh').
            export_dir (str): directory to which to export the H5M output file
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting source mesh H5M file...")

        export_path = Path(export_dir) / Path(filename).with_suffix(".h5m")
        self.mbc.write_file(str(export_path))


def parse_args():
    """Parser for running as a script"""
    parser = argparse.ArgumentParser(prog="source_mesh")

    parser.add_argument(
        "filename",
        help="YAML file defining ParaStell source mesh configuration",
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


def generate_source_mesh():
    """Main method when run as a command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger == True:
        logger = log.init()
    else:
        logger = log.NullLogger()

    vmec_file = all_data["vmec_file"]
    vmec_obj = read_vmec.VMECData(vmec_file)

    source_mesh_dict = all_data["source_mesh"]

    source_mesh = SourceMesh(
        vmec_obj,
        source_mesh_dict["mesh_size"],
        source_mesh_dict["toroidal_extent"],
        logger=logger**source_mesh_dict,
    )

    source_mesh.create_vertices()
    source_mesh.create_mesh()

    source_mesh.export_mesh(
        export_dir=args.export_dir,
        **(filter_kwargs(source_mesh_dict, ["filename"]))
    )


if __name__ == "__main__":
    generate_source_mesh()
