import argparse
from pathlib import Path
from abc import ABC
from itertools import cycle

import numpy as np
from scipy.interpolate import (
    RegularGridInterpolator,
    CloughTocher2DInterpolator,
)


import cadquery as cq
import pystell.read_vmec as read_vmec
import dagmc
from pymoab import core, types

from . import log
from .cubit_utils import (
    import_step_cubit,
    export_mesh_cubit,
    orient_spline_surfaces,
    merge_surfaces,
    mesh_volume_auto_factor,
)
from .utils import (
    normalize,
    expand_list,
    read_yaml_config,
    rotate_ribs,
    m2cm,
)


def create_moab_tris_from_verts(corners, mbc, reverse=False):
    """Create 2 moab triangle elements from a list of 4 pymoab verts.

    Arguments:
        corners (4x3 numpy array): list of 4 (x,y,z) points. Connecting the
            points in the order given should result in a polygon
        mbc (pymoab core): pymoab core instance to create elements with.

    Returns:
        list of two pymoab MBTRI elements
    """
    if reverse:
        tri_1 = mbc.create_element(
            types.MBTRI, [corners[0], corners[1], corners[2]]
        )
        tri_2 = mbc.create_element(
            types.MBTRI, [corners[0], corners[2], corners[3]]
        )
    else:
        tri_1 = mbc.create_element(
            types.MBTRI, [corners[2], corners[1], corners[0]]
        )
        tri_2 = mbc.create_element(
            types.MBTRI, [corners[3], corners[2], corners[0]]
        )

    return [tri_1, tri_2]


class ReferenceSurface(ABC):
    """An object representing the innermost surface from which subsequent
    layers are built.
    """

    def __init__():
        pass

    def angles_to_xyz(self, toroidal_angles, poloidal_angles, s, scale):
        """Method to go from a location defined by two angles and some
        constant to x, y, z coordinates.

        Arguments:
            toroidal_angles (iterable of float): Toroidal angles at which to
                evaluate cartesian coordinates. Measured in radians. Must be
                of the same length as poloidal_angles.
            poloidal_angles (iterable of float): Poloidal angles at which to
                evaluate cartesian coordinates. Measured in radians. Must be
                of the same length as toroidal_angles].
            s (float): Generic parameter which may affect the evaluation of
                the cartesian coordinate at a given angle pair.
            scale (float): Amount to scale resulting coordinates by.

        Returns:
            coords (numpy array): Nx3 array of Cartesian coordinates at each
                angle pair specified.
        """


class VMECSurface(ReferenceSurface):
    """An object that uses VMEC data to represent the innermost surface
    of an in vessel build

    Arguments:
        vmec_obj (object): plasma equilibrium VMEC object as defined by the
            PyStell-UW VMEC reader. Must have a method
            'vmec2xyz(s, theta, phi)' that returns an (x,y,z) coordinate for
            any closed flux surface label, s, poloidal angle, theta, and
            toroidal angle, phi.
    """

    def __init__(self, vmec_obj):
        self.vmec_obj = vmec_obj

    def angles_to_xyz(self, toroidal_angles, poloidal_angles, s, scale):
        """Evaluate the cartesian coordinates for a set of toroidal and
        poloidal angles and flux surface label.

        Arguments:
            toroidal_angles (iterable of float): Toroidal angles at which to
                    evaluate cartesian coordinates. Measured in radians. Must
                    be of the same length as poloidal_angles.
            poloidal_angles (iterable of float): Poloidal angles at which to
                    evaluate cartesian coordinates. Measured in radians. Must
                    be of the same length as toroidal_angles].
            s (float): the normalized closed flux surface label defining the
                point of reference for offset.
            scale (float): Amount to scale resulting coordinates by.

        Returns:
            coords (numpy array): Nx3 array of Cartesian coordinates at each
                angle pair specified.
        """
        coords = []
        for toroidal_angle, poloidal_angle in zip(
            toroidal_angles, poloidal_angles
        ):
            x, y, z = self.vmec_obj.vmec2xyz(s, poloidal_angle, toroidal_angle)
            coords.append([x, y, z])
        return np.array(coords) * scale


class RibBasedSurface(ReferenceSurface):
    """An object that uses closed loops of cartesian points (ribs) on planes of
    constant toroidal angle to approximate the innermost surface of an in
    vessel build

    Arguments:
        rib_data (numpy array): NxMx3 array of of cartesian points. The first
            dimension corresponds to the plane of constant toroidal angle on
            which the closed loop of points lies. The second dimension is the
            location on the closed loop at which the point lies, and the third
            dimension is the x,y,z value of that point.
        toroidal_angles (iterable of float): List of toroidal angles
            corresponding to the first dimension of rib_data. Measured in
            degrees.
        poloidal_angles (iterable of float): List of poloidal angles
            corresponding to the second dimension of rib_data. Measured in
            degrees. Should start at 0 degrees and end 360 degrees.
        neighbors (int): Number of neighbors to use when constructing the
            Radial Basis Function interpolator. If set to None, all points in
            rib_data will be used. This may require a large amount of memory.
            Defaults to 400. More neighbors results in a better fit.
    """

    def __init__(
        self, rib_data, toroidal_angles, poloidal_angles, neighbors=800
    ):
        self.rib_data = rib_data
        self.toroidal_angles = toroidal_angles
        self.poloidal_angles = poloidal_angles
        self.build_analytic_surface(neighbors=neighbors)

    def _extract_rib_data(self, ribs, toroidal_angles, poloidal_angles):
        for phi, rib in zip(toroidal_angles, ribs):
            for theta, rib_locus in zip(poloidal_angles, rib):
                self.x_data.append(rib_locus[0])
                self.y_data.append(rib_locus[1])
                self.z_data.append(rib_locus[2])
                self.grid_points.append([phi, theta])

    def build_analytic_surface(self, neighbors=400):
        """Build RBF interpolators for x,y,z coordinates using provided
        rib_data, toroidal_angles, and poloidal_angles.

        Arguments:
            neighbors (int): Number of neighbors to use when constructing the
                Radial Basis Function interpolator. If set to None, all points
                in rib_data will be used. This may require a large amount of
                memory. Defaults to 400.
        """
        self.x_data = []
        self.y_data = []
        self.z_data = []
        self.grid_points = []

        # add mock region before region to be modeled so the interpolator
        # knows about the periodicity

        # Toroidal Periodicity
        rotated_ribs = rotate_ribs(self.rib_data, -max(self.toroidal_angles))[
            0:-1
        ]
        shifted_toroidal_angles = self.toroidal_angles[0:-1] - max(
            self.toroidal_angles
        )
        self._extract_rib_data(
            rotated_ribs, shifted_toroidal_angles, self.poloidal_angles
        )

        # Poloidal Periodicity
        shifted_poloidal_angles = self.poloidal_angles[0:-1] - max(
            self.poloidal_angles
        )
        rib_subset = self.rib_data[:, 0:-1, :]
        self._extract_rib_data(
            rib_subset, self.toroidal_angles, shifted_poloidal_angles
        )

        # add data for the region to be modeled
        self._extract_rib_data(
            self.rib_data,
            self.toroidal_angles,
            self.poloidal_angles,
        )

        # add mock region after region to be modeled
        # Toroidal Periodicity
        rotated_ribs = rotate_ribs(self.rib_data, max(self.toroidal_angles))[
            1:
        ]
        shifted_toroidal_angles = self.toroidal_angles[1:] + max(
            self.toroidal_angles
        )

        self._extract_rib_data(
            rotated_ribs,
            shifted_toroidal_angles,
            self.poloidal_angles,
        )

        # Poloidal Periodicity
        shifted_poloidal_angles = self.poloidal_angles[1:] + max(
            self.poloidal_angles
        )
        rib_subset = self.rib_data[:, 1:, :]
        self._extract_rib_data(
            rib_subset, self.toroidal_angles, shifted_poloidal_angles
        )
        self.rbf_x = CloughTocher2DInterpolator(self.grid_points, self.x_data)
        self.rbf_y = CloughTocher2DInterpolator(self.grid_points, self.y_data)
        self.rbf_z = CloughTocher2DInterpolator(self.grid_points, self.z_data)

    def angles_to_xyz(self, toroidal_angles, poloidal_angles, s, scale):
        """ "Return the cartesian coordinates from the Radial Basis Function
        interpolators for a set of toroidal and poloidal angle pairs. Takes
        s as a argument for compatibility, but does nothing with it.

        Arguments:
            toroidal_angles (iterable of float): Toroidal angles at which to
                    evaluate cartesian coordinates. Measured in radians. Must
                    be of the same length as poloidal_angles.
            poloidal_angles (iterable of float): Poloidal angles at which to
                    evaluate cartesian coordinates. Measured in radians. Must
                    be of the same length as toroidal_angles].
            s (float): Not used.
            scale (float): Amount to scale resulting coordinates by.

        Returns:
            coords (numpy array): Nx3 array of Cartesian coordinates at each
                angle pair specified.
        """
        coords = []
        toroidal_angles = np.rad2deg(toroidal_angles)
        poloidal_angles = np.rad2deg(poloidal_angles)
        for toroidal_angle, poloidal_angle in zip(
            toroidal_angles, poloidal_angles
        ):
            x = self.rbf_x(toroidal_angle, poloidal_angle)
            y = self.rbf_y(toroidal_angle, poloidal_angle)
            z = self.rbf_z(toroidal_angle, poloidal_angle)
            coords.append([x, y, z])
        return np.array(coords) * scale


class InVesselBuild(object):
    """Parametrically models fusion stellarator in-vessel components using
    plasma equilibrium VMEC data and a user-defined radial build.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s)' that returns
            an Nx3 numpy array of cartesian coordinates for any closed flux
            surface label, s, poloidal angle (theta), and toroidal angle (phi).
        radial_build (object): RadialBuild class object with all attributes
            defined.
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        repeat (int): number of times to repeat build segment for full model
            (defaults to 0).
        num_ribs (int): total number of ribs over which to loft for each build
            segment (defaults to 61). Ribs are set at toroidal angles
            interpolated between those specified in 'toroidal_angles' if this
            value is greater than the number of entries in 'toroidal_angles'.
        num_rib_pts (int): total number of points defining each rib spline
            (defaults to 67). Points are set at poloidal angles interpolated
            between those specified in 'poloidal_angles' if this value is
            greater than the number of entries in 'poloidal_angles'.
        scale (float): a scaling factor between the units of VMEC and [cm]
            (defaults to m2cm = 100).
        use_pydagmc (bool): If True, generate components with pydagmc, rather
            than CADQuery. Defaults to False.
    """

    def __init__(self, ref_surf, radial_build, logger=None, **kwargs):

        self.logger = logger
        self.ref_surf = ref_surf
        self.radial_build = radial_build

        self.repeat = 0
        self.num_ribs = 61
        self.num_rib_pts = 67
        self.scale = m2cm
        self.use_pydagmc = False

        for name in kwargs.keys() & (
            "repeat",
            "num_ribs",
            "num_rib_pts",
            "scale",
            "use_pydagmc",
        ):
            self.__setattr__(name, kwargs[name])

        self.Surfaces = {}
        self.Components = {}

    @property
    def ref_surf(self):
        return self._ref_surf

    @ref_surf.setter
    def ref_surf(self, ref_surf):
        self._ref_surf = ref_surf

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    @property
    def repeat(self):
        return self._repeat

    @repeat.setter
    def repeat(self, num):
        self._repeat = num
        if (self._repeat + 1) * self.radial_build.toroidal_angles[-1] > 360.0:
            e = AssertionError(
                "Total toroidal extent requested with repeated geometry "
                'exceeds 360 degrees. Please examine the "repeat" parameter '
                'and the "toroidal_angles" parameter of "radial_build".'
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def use_pydagmc(self):
        return self._use_pydagmc

    @use_pydagmc.setter
    def use_pydagmc(self, value):
        self._use_pydagmc = value
        if self._use_pydagmc:
            self.mbc = core.Core()
            self.dag_model = dagmc.DAGModel(self.mbc)

    def _interpolate_offset_matrix(self, offset_mat):
        """Interpolates total offset for expanded angle lists using cubic spline
        interpolation.
        (Internal function not intended to be called externally)

        Returns:
            interpolated_offset_mat (np.ndarray(double)): expanded matrix
                including interpolated offset values at additional rows and
                columns [cm].
        """
        interpolator = RegularGridInterpolator(
            (
                self.radial_build.toroidal_angles,
                self.radial_build.poloidal_angles,
            ),
            offset_mat,
            method="linear" if self.use_pydagmc else "pchip",
        )

        interpolated_offset_mat = np.array(
            [
                [
                    interpolator([np.rad2deg(phi), np.rad2deg(theta)])[0]
                    for theta in self._poloidal_angles_exp
                ]
                for phi in self._toroidal_angles_exp
            ]
        )

        return interpolated_offset_mat

    def populate_surfaces(self):
        """Populates Surface class objects representing the outer surface of
        each component specified in the radial build.
        """
        self._logger.info(
            "Populating surface objects for in-vessel components..."
        )

        self._toroidal_angles_exp = np.deg2rad(
            expand_list(self.radial_build.toroidal_angles, self.num_ribs)
        )
        self._poloidal_angles_exp = np.deg2rad(
            expand_list(self.radial_build.poloidal_angles, self.num_rib_pts)
        )

        offset_mat = np.zeros(
            (
                len(self.radial_build.toroidal_angles),
                len(self.radial_build.poloidal_angles),
            )
        )

        for name, layer_data in self.radial_build.radial_build.items():
            if name == "plasma":
                s = 1.0
            else:
                s = self.radial_build.wall_s

            offset_mat += np.array(layer_data["thickness_matrix"])
            interpolated_offset_mat = self._interpolate_offset_matrix(
                offset_mat
            )

            self.Surfaces[name] = Surface(
                self._ref_surf,
                s,
                self._poloidal_angles_exp,
                self._toroidal_angles_exp,
                interpolated_offset_mat,
                self.scale,
            )

        [surface.populate_ribs() for surface in self.Surfaces.values()]

    def calculate_loci(self):
        """Calls calculate_loci method in Surface class for each component
        specified in the radial build.
        """
        self._logger.info("Computing point cloud for in-vessel components...")

        [surface.calculate_loci() for surface in self.Surfaces.values()]

    def generate_components(self):
        if self.use_pydagmc:
            self.generate_components_pydagmc()
        else:
            self.generate_components_cadquery()

    def generate_components_cadquery(self):
        """Constructs a CAD solid for each component specified in the radial
        build by cutting the interior surface solid from the outer surface
        solid for a given component.
        """
        self._logger.info(
            "Constructing CadQuery objects for in-vessel components..."
        )

        interior_surface = None

        segment_angles = np.linspace(
            self.radial_build.toroidal_angles[-1],
            self._repeat * self.radial_build.toroidal_angles[-1],
            num=self._repeat,
        )

        for name, surface in self.Surfaces.items():
            outer_surface = surface.generate_surface()

            if interior_surface is not None:
                segment = outer_surface.cut(interior_surface)
            else:
                segment = outer_surface

            component = segment

            for angle in segment_angles:
                rot_segment = segment.rotate((0, 0, 0), (0, 0, 1), angle)
                component = component.fuse(rot_segment)

            self.Components[name] = component
            interior_surface = outer_surface

    def _connect_ribs_with_tris_moab(self, rib1, rib2, reverse=False):
        """Creat MBTRI elements add add them to a surface between two ribs.

        Arguments:
            rib1 (Rib object): First of two ribs to be connected.
            rib2 (Rib object): Second of two ribs to be connected.
            reverse (bool): Optional. Whether to reverse the connectivity of
                the MBTRIs being generated. Defaults to False.

        Returns:
            mb_tris (list of Entity Handle): List of the entity handles of the
                MBTRIs connecting the two ribs.
        """
        mb_tris = []
        for rib_loci_index, _ in enumerate(rib1.rib_loci[0:-1]):
            corner1 = rib1.mb_verts[rib_loci_index]
            corner2 = rib1.mb_verts[rib_loci_index + 1]
            corner3 = rib2.mb_verts[rib_loci_index + 1]
            corner4 = rib2.mb_verts[rib_loci_index]
            corners = [corner1, corner2, corner3, corner4]
            mb_tris += create_moab_tris_from_verts(
                corners, self.mbc, reverse=reverse
            )
        return mb_tris

    def _generate_pymoab_verts(self):
        """Generate MBVERTEX entities from rib loci in all surfaces"""
        [
            surface._generate_pymoab_verts(self.mbc)
            for surface in self.Surfaces.values()
        ]

    def _generate_curved_surfaces_pydagmc(self):
        """Generate the faceted representation of each curved surface and
        add it to the PyDAGMC model, remembering the surface ids. The sense
        of the triangles should point outward (increasing radial direction),
        with the exception of the first surface, which should point inward
        since the implicit complement is being used for the plasma chamber.
        """
        self.curved_surface_ids = []
        surfaces = list(self.Surfaces.values())
        first_surface = surfaces[0]
        for surface in surfaces:
            mb_tris = []
            for rib, next_rib in zip(surface.Ribs[0:-1], surface.Ribs[1:]):
                mb_tris += self._connect_ribs_with_tris_moab(
                    rib,
                    next_rib,
                    reverse=(surface == first_surface),
                )
            dagmc_surface = self.dag_model.create_surface()
            self.dag_model.mb.add_entities(dagmc_surface.handle, mb_tris)
            self.curved_surface_ids.append(dagmc_surface.id)

    def _generate_end_cap_surfaces_pydagmc(self):
        """Generate the faceted representation of the planar end cap surfaces
        and add them to the PyDAGMC model, remembering the surface ids.
        The sense of the triangles should point toward the implicit complement.
        """
        self.end_cap_surface_ids = []
        for surface, next_surface in zip(
            list(self.Surfaces.values())[0:-1],
            list(self.Surfaces.values())[1:],
        ):
            end_cap_pair = []
            for index in (0, -1):
                mb_tris = self._connect_ribs_with_tris_moab(
                    surface.Ribs[index],
                    next_surface.Ribs[index],
                    reverse=(index == -1),
                )
                end_cap = self.dag_model.create_surface()
                self.mbc.add_entities(end_cap.handle, mb_tris)
                end_cap_pair.append(end_cap.id)

            self.end_cap_surface_ids.append(end_cap_pair)

    def _generate_volumes_pydagmc(self):
        """Use the curved surface and end cap surface IDs to build the
        the volumes by applying the correct surface sense to each surface.
        The convention here is to point the surface sense toward the implicit
        complement, or if the surface is between two volumes then the surface
        sense should point in the increasing radial direction."""

        [self.dag_model.create_volume() for _ in list(self.Surfaces)[:-1]]

        # First surface goes to the implicit complement (plasma chamber)
        first_surface = self.dag_model.surfaces_by_id[
            self.curved_surface_ids[0]
        ]
        first_surface.surf_sense = [
            self.dag_model.volumes_by_id[first_surface.id],
            None,
        ]

        for surface_id in self.curved_surface_ids[1:-1]:
            self.dag_model.surfaces_by_id[surface_id].surf_sense = [
                self.dag_model.volumes_by_id[surface_id - 1],
                self.dag_model.volumes_by_id[surface_id],
            ]

        # if it the last surface it goes to the implicit complement
        last_surface = self.dag_model.surfaces_by_id[
            self.curved_surface_ids[-1]
        ]
        last_surface.surf_sense = [
            self.dag_model.volumes_by_id[last_surface.id - 1],
            None,
        ]

        # all end caps go to the implicit complement.
        for vol_id, end_cap_ids in enumerate(
            self.end_cap_surface_ids, start=1
        ):
            for end_cap_id in end_cap_ids:
                self.dag_model.surfaces_by_id[end_cap_id].surf_sense = [
                    self.dag_model.volumes_by_id[vol_id],
                    None,
                ]

    def _tag_volumes_with_materials_pydagmc(self):
        """Tag each volume with the appropriate material name"""
        for vol, (layer_name, layer_data) in zip(
            self.dag_model.volumes,
            list(self.radial_build.radial_build.items())[1:],
        ):

            mat = layer_data.get("mat_tag", layer_name)
            group = dagmc.Group.create(self.dag_model, name="mat:" + mat)
            group.add_set(vol)
            layer_data["vol_id"] = vol.id

    def generate_components_pydagmc(self):
        """Use PyDAGMC to build a DAGMC model of the invessel components"""
        if np.isclose(
            (self._repeat + 1) * self.radial_build.toroidal_angles[-1], 360
        ):
            e = AssertionError(
                "The PyDAGMC workflow does not support modeling 360-degree "
                "geometries. For configurations with more than one period, "
                "please consider modeling only one period. i.e. set "
                "'repeat = 0'."
            )
            self._logger.error(e.args[0])
            raise e
        self._logger.info(
            "Generating DAGMC model of in-vessel components with PyDAGMC..."
        )
        self._generate_pymoab_verts()
        self._generate_curved_surfaces_pydagmc()
        self._generate_end_cap_surfaces_pydagmc()
        self._generate_volumes_pydagmc()
        self._tag_volumes_with_materials_pydagmc()

    def get_loci(self):
        """Returns the set of point-loci defining the outer surfaces of the
        components specified in the radial build.
        """
        return np.array(
            [surface.get_loci() for surface in self.Surfaces.values()]
        )

    def merge_layer_surfaces(self):
        """Merges ParaStell in-vessel component surfaces in Coreform Cubit
        based on surface IDs rather than imprinting and merging all. Assumes
        that the radial_build dictionary is ordered radially outward. Note that
        overlaps between magnet volumes and in-vessel components will not be
        merged in this workflow.
        """
        # Tracks the surface id of the outer surface of the previous layer
        prev_outer_surface_id = None

        for data in self.radial_build.radial_build.values():

            inner_surface_id, outer_surface_id = orient_spline_surfaces(
                data["vol_id"]
            )

            # Conditionally skip merging (first iteration only)
            if prev_outer_surface_id is None:
                prev_outer_surface_id = outer_surface_id
            else:
                merge_surfaces(inner_surface_id, prev_outer_surface_id)
                prev_outer_surface_id = outer_surface_id

    def import_step_cubit(self):
        """Imports STEP files from in-vessel build into Coreform Cubit."""
        for name, data in self.radial_build.radial_build.items():
            vol_id = import_step_cubit(name, self.export_dir)
            data["vol_id"] = vol_id

    def export_step(self, export_dir=""):
        """Export CAD solids as STEP files via CadQuery.

        Arguments:
            export_dir (str): directory to which to export the STEP output files
                (optional, defaults to empty string).
        """
        self._logger.info("Exporting STEP files for in-vessel components...")

        self.export_dir = export_dir

        for name, component in self.Components.items():
            export_path = Path(self.export_dir) / Path(name).with_suffix(
                ".step"
            )
            cq.exporters.export(component, str(export_path))

    def extract_solids_and_mat_tags(self):
        """Get a list of all cadquery solids, and a corresponding list of
        the respective material tags.

        Returns:
            solids (list): list of in-vessel component CadQuery solid objects.
            mat_tags (list): list of in-vessel component material tags.
        """
        solids = []
        mat_tags = []

        for name, solid in self.Components.items():
            solids.append(solid)
            mat_tags.append(self.radial_build.radial_build[name]["mat_tag"])

        return solids, mat_tags

    def export_component_mesh(
        self, components, mesh_size=5, import_dir="", export_dir=""
    ):
        """Creates a tetrahedral mesh of an in-vessel component volume
        via Coreform Cubit and exports it as H5M file.

        Arguments:
            components (array of strings): array containing the name
                of the in-vessel components to be meshed.
            mesh_size (float): controls the size of the mesh. Takes values
                between 1.0 (finer) and 10.0 (coarser) (optional, defaults to
                5.0).
            import_dir (str): directory containing the STEP file of
                the in-vessel component (optional, defaults to empty string).
            export_dir (str): directory to which to export the h5m
                output file (optional, defaults to empty string).
        """
        for component in components:
            vol_id = import_step_cubit(component, import_dir)
            mesh_volume_auto_factor([vol_id], mesh_size=mesh_size)
            export_mesh_cubit(
                filename=component,
                export_dir=export_dir,
                delete_upon_export=True,
            )


class Surface(object):
    """An object representing a surface formed by lofting across a set of
    "ribs" located at different toroidal planes and offset from a reference
    surface.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s)' that returns
            an Nx3 numpy array of cartesian coordinates for any closed flux
            surface label, s, poloidal angle (theta), and toroidal angle (phi).
        s (float): the normalized closed flux surface label defining the point
            of reference for offset.
        theta_list (np.array(double)): the set of poloidal angles specified for
            each rib [rad].
        phi_list (np.array(double)): the set of toroidal angles defining the
            plane in which each rib is located [rad].
        offset_mat (np.array(double)): the set of offsets from the surface
            defined by s for each toroidal angle, poloidal angle pair on the
            surface [cm].
        scale (float): a scaling factor between the units of VMEC and [cm].
    """

    def __init__(self, ref_surf, s, theta_list, phi_list, offset_mat, scale):

        self.ref_surf = ref_surf
        self.s = s
        self.theta_list = theta_list
        self.phi_list = phi_list
        self.offset_mat = offset_mat
        self.scale = scale

        self.surface = None

    def populate_ribs(self):
        """Populates Rib class objects for each toroidal angle specified in
        the surface.
        """
        self.Ribs = [
            Rib(
                self.ref_surf,
                self.s,
                self.theta_list,
                phi,
                self.offset_mat[i, :],
                self.scale,
                i,
            )
            for i, phi in enumerate(self.phi_list)
        ]

    def calculate_loci(self):
        """Calls calculate_loci method in Rib class for each rib in the surface."""
        [rib.calculate_loci() for rib in self.Ribs]

    def _generate_pymoab_verts(self, mbc):
        """Generate MBTVERTEX entities from rib loci in all ribs.

        Arguments:
            mbc (PyMOAB Core): PyMOAB Core instance to add the MBVERTEX
                entities to.
        """
        [rib._generate_pymoab_verts(mbc) for rib in self.Ribs]

    def generate_surface(self):
        """Constructs a surface by lofting across a set of rib splines."""
        if not self.surface:
            self.surface = cq.Solid.makeLoft(
                [rib.generate_rib() for rib in self.Ribs]
            )

        return self.surface

    def get_loci(self):
        """Returns the set of point-loci defining the ribs in the surface."""
        return np.array([rib.rib_loci for rib in self.Ribs])


class Rib(object):
    """An object representing a curve formed by interpolating a spline through
    a set of points located in the same toroidal plane but differing poloidal
    angles and offset from a reference curve.

    Arguments:
        ref_surf (object): ReferenceSurface object. Must have a method
            'angles_to_xyz(toroidal_angles, poloidal_angles, s)' that returns
            an Nx3 numpy array of cartesian coordinates for any closed flux
            surface label, s, poloidal angle (theta), and toroidal angle (phi).
        s (float): the normalized closed flux surface label defining the point
            of reference for offset.
        phi (np.array(double)): the toroidal angle defining the plane in which
            the rib is located [rad].
        theta_list (np.array(double)): the set of poloidal angles specified for
            the rib [rad].
        offset_list (np.array(double)): the set of offsets from the curve
            defined by s for each toroidal angle, poloidal angle pair in the rib
            [cm].
        scale (float): a scaling factor between the units of VMEC and [cm].
    """

    def __init__(
        self, ref_surf, s, theta_list, phi, offset_list, scale, rib_index
    ):

        self.ref_surf = ref_surf
        self.s = s
        self.theta_list = theta_list
        self.phi = phi
        self.offset_list = offset_list
        self.scale = scale
        self.rib_index = rib_index

    def _calculate_cartesian_coordinates(self, poloidal_offset=0):
        """Return an N x 3 NumPy array containing the Cartesian coordinates of
        the points at this toroidal angle and N different poloidal angles, each
        offset slightly.
        (Internal function not intended to be called externally)

        Arguments:
            poloidal_offset (float) : some offset to apply to the full set of
                poloidal angles for evaluating the location of the Cartesian
                points (optional, defaults to 0).
        """
        toroidal_angles = np.ones(len(self.theta_list)) * self.phi
        return self.ref_surf.angles_to_xyz(
            toroidal_angles,
            self.theta_list + poloidal_offset,
            self.s,
            self.scale,
        )

    def _normals(self):
        """Approximate the normal to the curve at each poloidal angle by first
        approximating the tangent to the curve and then taking the
        cross-product of that tangent with a vector defined as normal to the
        plane at this toroidal angle.
        (Internal function not intended to be called externally)

        Arguments:
            r_loci (np.array(double)): Cartesian point-loci of reference
                surface rib [cm].
        """
        eps = 1e-4
        next_pt_loci = self._calculate_cartesian_coordinates(eps)

        tangent = next_pt_loci - self.rib_loci

        plane_norm = np.array([-np.sin(self.phi), np.cos(self.phi), 0])

        norm = np.cross(plane_norm, tangent)

        return normalize(norm)

    def calculate_loci(self):
        """Generates Cartesian point-loci for stellarator rib."""
        self.rib_loci = self._calculate_cartesian_coordinates()
        if not np.all(self.offset_list == 0):
            self.rib_loci += self.offset_list[:, np.newaxis] * self._normals()

        self.rib_loci[-1] = self.rib_loci[0]

    def _generate_pymoab_verts(self, mbc):
        """Converts point-loci to MBVERTEX and adds them to a PyMOAB
        Core instance. The first and last rib loci are identical. To avoid
        having separate MBVERTEX entities which are coincident, the last
        element in rib_loci is not made into an MBVERTEX, and the entity
        handle corresponding to the first rib locus is appended to the array
        of MBVERTEX, closing the loop.

        Arguments:
            mbc (PyMOAB Core): PyMOAB Core instance to add the MBVERTEX
                entities to.
        """
        self.mb_verts = mbc.create_vertices(
            self.rib_loci[0:-1].flatten()
        ).to_array()
        self.mb_verts = np.append(self.mb_verts, self.mb_verts[0])

    def generate_rib(self):
        """Constructs component rib by constructing a spline connecting all
        specified Cartesian point-loci.
        """
        rib_loci = [cq.Vector(tuple(r)) for r in self.rib_loci]
        spline = cq.Edge.makeSpline(rib_loci).close()
        rib_spline = cq.Wire.assembleEdges([spline]).close()

        return rib_spline


class RadialBuild(object):
    """Parametrically defines ParaStell in-vessel component geometries.
    In-vessel component thicknesses are defined on a grid of toroidal and
    poloidal angles, and the first wall profile is defined by a closed flux
    surface extrapolation.

    Arguments:
        toroidal_angles (array of float): toroidal angles at which radial build
            is specified. This list should always begin at 0.0 and it is
            advised not to extend beyond one stellarator period. To build a
            geometry that extends beyond one period, make use of the 'repeat'
            parameter [deg].
        poloidal_angles (array of float): poloidal angles at which radial build
            is specified. This array should always span 360 degrees [deg].
        wall_s (float): closed flux surface label extrapolation at wall.
        radial_build (dict): dictionary representing the three-dimensional
            radial build of in-vessel components, including
            {
                'component': {
                    'thickness_matrix': 2-D matrix defining component
                        thickness at (toroidal angle, poloidal angle)
                        locations. Rows represent toroidal angles, columns
                        represent poloidal angles, and each must be in the same
                        order provided in toroidal_angles and poloidal_angles
                        [cm](ndarray(float)).
                    'mat_tag': DAGMC material tag for component in DAGMC
                        neutronics model (str, optional, defaults to None). If
                        none is supplied, the 'component' key will be used.
                }
            }.
        split_chamber (bool): if wall_s > 1.0, separate interior vacuum
            chamber into plasma and scrape-off layer components (optional,
            defaults to False). If an item with a 'sol' key is present in the
            radial_build dictionary, settting this to False will not combine
            the resultant 'chamber' with 'sol'. To include a custom scrape-off
            layer definition for 'chamber', add an item with a 'chamber' key
            and desired 'thickness_matrix' value to the radial_build dictionary.
        logger (object): logger object (optional, defaults to None). If no
            logger is supplied, a default logger will be instantiated.

    Optional attributes:
        plasma_mat_tag (str): DAGMC material tag to use for plasma if
            split_chamber is True (defaults to 'Vacuum').
        sol_mat_tag (str): DAGMC material tag to use for scrape-off layer if
            split_chamber is True (defaults to 'Vacuum').
        chamber_mat_tag (str): DAGMC material tag to use for interior vacuum
            chamber if split_chamber is False (defaults to 'Vacuum).
    """

    def __init__(
        self,
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build,
        split_chamber=False,
        logger=None,
        **kwargs,
    ):

        self.logger = logger
        self.toroidal_angles = toroidal_angles
        self.poloidal_angles = poloidal_angles
        self.wall_s = wall_s
        self.radial_build = radial_build
        self.split_chamber = split_chamber

        for name in kwargs.keys() & (
            "plasma_mat_tag",
            "sol_mat_tag",
            "chamber_mat_tag",
        ):
            self.__setattr__(name, kwargs[name])

        self._logger.info("Constructing radial build...")

    @property
    def toroidal_angles(self):
        return self._toroidal_angles

    @toroidal_angles.setter
    def toroidal_angles(self, angle_list):
        if hasattr(self, "toroidal_angles"):
            e = AttributeError(
                '"toroidal_angles" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._toroidal_angles = angle_list
        if self._toroidal_angles[0] != 0.0:
            e = ValueError("The first entry in toroidal_angles must be 0.0.")
            self._logger.error(e.args[0])
            raise e
        if self._toroidal_angles[-1] > 360.0:
            e = ValueError("Toroidal extent cannot exceed 360.0 degrees.")
            self._logger.error(e.args[0])
            raise e

    @property
    def poloidal_angles(self):
        return self._poloidal_angles

    @poloidal_angles.setter
    def poloidal_angles(self, angle_list):
        if hasattr(self, "poloidal_angles"):
            e = AttributeError(
                '"poloidal_angles" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._poloidal_angles = angle_list
        if self._poloidal_angles[-1] - self._poloidal_angles[0] > 360.0:
            e = AssertionError(
                "Poloidal extent must span exactly 360.0 degrees."
            )
            self._logger.error(e.args[0])
            raise e

    @property
    def wall_s(self):
        return self._wall_s

    @wall_s.setter
    def wall_s(self, s):
        if hasattr(self, "wall_s"):
            e = AttributeError(
                '"wall_s" cannot be set after class initialization. Please '
                "create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._wall_s = s
        if self._wall_s < 1.0:
            e = ValueError("wall_s must be greater than or equal to 1.0.")
            self._logger.error(e.args[0])
            raise e

    @property
    def radial_build(self):
        return self._radial_build

    @radial_build.setter
    def radial_build(self, build_dict):
        self._radial_build = build_dict

        for name, component in self._radial_build.items():
            component["thickness_matrix"] = np.array(
                component["thickness_matrix"]
            )
            if component["thickness_matrix"].shape != (
                len(self._toroidal_angles),
                len(self._poloidal_angles),
            ):
                e = AssertionError(
                    f"The dimensions of {name}'s thickness matrix "
                    f'{component["thickness_matrix"].shape} must match the '
                    "dimensions defined by the toroidal and poloidal angle "
                    "lists "
                    f"{len(self._toroidal_angles),len(self._poloidal_angles)}, "
                    "which define the rows and columns of the matrix, "
                    "respectively."
                )
                self._logger.error(e.args[0])
                raise e

            if np.any(component["thickness_matrix"] < 0):
                e = ValueError(
                    "Component thicknesses must be greater than or equal to 0. "
                    "Check thickness inputs for negative values."
                )
                self._logger.error(e.args[0])
                raise e

            if "mat_tag" not in component:
                self._set_mat_tag(name, name)

    @property
    def split_chamber(self):
        return self._split_chamber

    @split_chamber.setter
    def split_chamber(self, value):
        if hasattr(self, "split_chamber"):
            e = AttributeError(
                '"split_chamber" cannot be set after class initialization. '
                "Please create new class instance to alter this attribute."
            )
            self._logger.error(e.args[0])
            raise e

        self._split_chamber = value

        if self._split_chamber:
            if self._wall_s > 1.0 and "sol" not in self._radial_build:
                self.radial_build = {
                    "sol": {
                        "thickness_matrix": np.zeros(
                            (
                                len(self._toroidal_angles),
                                len(self._poloidal_angles),
                            )
                        )
                    },
                    **self.radial_build,
                }
                if not hasattr(self, "sol_mat_tag"):
                    self.sol_mat_tag = "Vacuum"

            inner_volume_name = "plasma"
            inner_volume_tag = "plasma_mat_tag"
        else:
            inner_volume_name = "chamber"
            inner_volume_tag = "chamber_mat_tag"

        self.radial_build = {
            inner_volume_name: {
                "thickness_matrix": np.zeros(
                    (len(self._toroidal_angles), len(self._poloidal_angles))
                )
            },
            **self.radial_build,
        }
        if not hasattr(self, inner_volume_tag):
            self.__setattr__(inner_volume_tag, "Vacuum")

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, logger_object):
        self._logger = log.check_init(logger_object)

    @property
    def plasma_mat_tag(self):
        return self._plasma_mat_tag

    @plasma_mat_tag.setter
    def plasma_mat_tag(self, mat_tag):
        self._plasma_mat_tag = mat_tag
        self._set_mat_tag("plasma", self._plasma_mat_tag)

    @property
    def sol_mat_tag(self):
        return self._sol_mat_tag

    @sol_mat_tag.setter
    def sol_mat_tag(self, mat_tag):
        self._sol_mat_tag = mat_tag
        self._set_mat_tag("sol", self._sol_mat_tag)

    @property
    def chamber_mat_tag(self):
        return self._chamber_mat_tag

    @chamber_mat_tag.setter
    def chamber_mat_tag(self, mat_tag):
        self._chamber_mat_tag = mat_tag
        self._set_mat_tag("chamber", self._chamber_mat_tag)

    def _set_mat_tag(self, name, mat_tag):
        """Sets DAGMC material tag for a given component.
        (Internal function not intended to be called externally)

        Arguments:
            name (str): name of component.
            mat_tag (str): DAGMC material tag.
        """
        self.radial_build[name]["mat_tag"] = mat_tag


def parse_args():
    """Parser for running as a script."""
    parser = argparse.ArgumentParser(prog="invessel_build")

    parser.add_argument(
        "filename",
        help="YAML file defining ParaStell in-vessel component configuration",
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


def generate_invessel_build():
    """Main method when run as a command line script."""
    args = parse_args()

    all_data = read_yaml_config(args.filename)

    if args.logger == True:
        logger = log.init()
    else:
        logger = log.NullLogger()

    vmec_file = all_data["vmec_file"]
    vmec_obj = read_vmec.VMECData(vmec_file)

    invessel_build_dict = all_data["invessel_build"]

    radial_build = RadialBuild(
        invessel_build_dict["toroidal_angles"],
        invessel_build_dict["poloidal_angles"],
        invessel_build_dict["wall_s"],
        invessel_build_dict["radial_build"],
        logger=logger,
        **invessel_build_dict,
    )

    invessel_build = InVesselBuild(
        vmec_obj, radial_build, logger=logger, **invessel_build_dict
    )

    invessel_build.populate_surfaces()
    invessel_build.calculate_loci()
    invessel_build.generate_components()

    invessel_build.export_step(export_dir=args.export_dir)


if __name__ == "__main__":
    generate_invessel_build()
