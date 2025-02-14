from pathlib import Path

import pytest
import numpy as np
import cadquery as cq

import parastell.magnet_coils as magnet_coils
from parastell.cubit_utils import (
    check_cubit_installation,
    create_new_cubit_instance,
)

files_to_remove = [
    "magnet_set.step",
    "magnet_mesh.exo",
    "magnet_mesh.h5m",
    "stellarator.log",
    "step_import.log",
    "single_coil.step",
]


def remove_files():
    for file in files_to_remove:
        if Path(file).exists():
            Path.unlink(file)


simple_filament_coords = np.array(
    [[0, 0, 0], [200, 300, 0], [0, 0, 100], [0, 0, 0]]
)


@pytest.fixture
def coil_set_from_filaments():
    coils_file = Path("files_for_tests") / "coils.example"
    width = 40.0
    thickness = 50.0
    toroidal_extent = 90.0
    sample_mod = 10

    coil_set_obj = magnet_coils.MagnetSetFromFilaments(
        coils_file, width, thickness, toroidal_extent, sample_mod=sample_mod
    )

    return coil_set_obj


@pytest.fixture
def coil_set_from_geometry():
    geom_file = Path("files_for_tests") / "magnet_geom.step"

    coil_set_obj = magnet_coils.MagnetSetFromGeometry(geom_file)

    return coil_set_obj


@pytest.fixture
def single_filament():
    return magnet_coils.Filament(simple_filament_coords)


@pytest.fixture
def single_coil(single_filament):
    return magnet_coils.MagnetCoil(single_filament, 10, 20, 1)


def test_single_filament(single_filament):
    """Tests whether the data for a Filament object is generated as expected."""
    tangents_exp = np.array(
        [
            [0.53452248, 0.80178373, -0.26726124],
            [0.0, 0.0, 1.0],
            [-0.5547002, -0.83205029, 0.0],
            [0.53452248, 0.80178373, -0.26726124],
        ]
    )
    com_exp = np.array([66.66666667, 100.0, 33.33333333])
    com_toroidal_angle_exp = 0.982793723247329

    assert np.allclose(tangents_exp, single_filament.tangents)
    assert np.allclose(com_exp, single_filament.com)
    assert np.isclose(
        com_toroidal_angle_exp, single_filament.com_toroidal_angle
    )


def test_single_coil(single_coil):
    """Tests whether a MagnetCoil object can be generated with valid CAD."""
    remove_files()

    single_coil.create_magnet()
    cq.exporters.export(single_coil.solid, "single_coil.step")
    assert Path("single_coil.step").exists()

    remove_files()


def test_magnet_construction(coil_set_from_filaments):
    """Tests whether the MagnetSetFromFilaments object is instantiated and
    constructed as expected, along with relevant data.
    """
    remove_files()

    width_exp = 40.0
    thickness_exp = 50.0
    toroidal_extent_exp = np.deg2rad(90.0)
    max_cs_len_exp = 50.0
    average_radial_distance_exp = 1023.7170384211436
    max_radial_distance_exp = 1646.3258131460148
    len_coords_exp = 129
    len_coils_exp = 1

    coil_set_from_filaments.populate_magnet_coils()
    coil_set_from_filaments.build_magnet_coils()

    assert len(coil_set_from_filaments.magnet_coils) == len_coils_exp
    assert coil_set_from_filaments.width == width_exp
    assert coil_set_from_filaments.thickness == thickness_exp
    assert coil_set_from_filaments.toroidal_extent == toroidal_extent_exp
    assert coil_set_from_filaments.max_cs_len == max_cs_len_exp
    assert (
        coil_set_from_filaments.average_radial_distance
        == average_radial_distance_exp
    )
    assert (
        coil_set_from_filaments.max_radial_distance == max_radial_distance_exp
    )

    test_coil = coil_set_from_filaments.magnet_coils[0]
    assert len(test_coil.coords) == len_coords_exp

    assert len(coil_set_from_filaments.coil_solids) == len_coils_exp

    remove_files()


def test_magnet_exports_from_filaments(coil_set_from_filaments):
    """Tests whether the MagnetSetFromFilaments' export functionality behaves
    as expected. The Cubit-enabled portion of this test is skipped if Cubit
    cannot be imported.
    """
    remove_files()

    volume_ids_exp = list(range(1, 2))

    coil_set_from_filaments.populate_magnet_coils()
    coil_set_from_filaments.build_magnet_coils()
    coil_set_from_filaments.export_step()
    assert Path("magnet_set.step").exists()

    if check_cubit_installation():
        create_new_cubit_instance()

        coil_set_from_filaments.mesh_magnets()
        assert coil_set_from_filaments.volume_ids == volume_ids_exp

        coil_set_from_filaments.export_mesh()
        assert Path("magnet_mesh.h5m").exists()

    remove_files()


def test_magnet_exports_from_geometry(coil_set_from_geometry):
    """Tests whether the MagnetSetFromGeometry's export functionality behaves
    as expected. This test is skipped if Cubit cannot be imported.
    """
    pytest.importorskip("cubit")

    remove_files()

    volume_ids_exp = list(range(1, 2))

    create_new_cubit_instance()

    coil_set_from_geometry.mesh_magnets()
    assert coil_set_from_geometry.volume_ids == volume_ids_exp

    coil_set_from_geometry.export_mesh()
    assert Path("magnet_mesh.h5m").exists()

    remove_files()
