from pathlib import Path

import numpy as np
import pytest
import pystell.read_vmec as read_vmec

import parastell.source_mesh as sm


def remove_files():

    if Path("source_mesh.h5m").exists():
        Path.unlink("source_mesh.h5m")
    if Path("stellarator.log").exists():
        Path.unlink("stellarator.log")


@pytest.fixture
def source_mesh():

    vmec_file = Path("files_for_tests") / "wout_vmec.nc"

    vmec_obj = read_vmec.VMECData(vmec_file)

    # Set mesh grids to minimum that maintains element aspect ratios that do not
    # result in negative volumes
    cfs_grid = np.linspace(0.0, 1.0, num=6)
    poloidal_grid = np.linspace(0.0, 360.0, num=41)
    toroidal_grid = np.linspace(0.0, 15.0, num=9)

    source_mesh_obj = sm.SourceMesh(
        vmec_obj, cfs_grid, poloidal_grid, toroidal_grid
    )

    return source_mesh_obj


def test_mesh_basics(source_mesh):

    cfs_grid_exp = np.linspace(0.0, 1.0, num=6)[1:]
    poloidal_grid_exp = np.linspace(0.0, 2 * np.pi, num=41)[:-1]
    toroidal_grid_exp = np.linspace(0.0, np.deg2rad(15.0), num=9)
    scale_exp = 100

    remove_files()

    assert np.allclose(source_mesh.cfs_grid, cfs_grid_exp)
    assert np.allclose(source_mesh.poloidal_grid, poloidal_grid_exp)
    assert np.allclose(source_mesh.toroidal_grid, toroidal_grid_exp)
    assert source_mesh.scale == scale_exp

    remove_files()


def test_vertices(source_mesh):

    num_s = 6
    num_theta = 41
    num_phi = 9

    num_verts_exp = num_phi * ((num_s - 1) * (num_theta - 1) + 1)

    remove_files()

    source_mesh.create_vertices()

    assert source_mesh.coords.shape == (num_verts_exp, 3)
    assert source_mesh.coords_s.shape == (num_verts_exp,)
    assert len(source_mesh.verts) == num_verts_exp

    remove_files()


def test_mesh_generation(source_mesh):

    num_s = 6
    num_theta = 41
    num_phi = 9

    tets_per_wedge = 3
    tets_per_hex = 5

    num_elements_exp = tets_per_wedge * (num_theta - 1) * (
        num_phi - 1
    ) + tets_per_hex * (num_s - 2) * (num_theta - 1) * (num_phi - 1)
    num_neg_vols_exp = 0

    remove_files()

    source_mesh.create_vertices()
    source_mesh.create_mesh()

    assert len(source_mesh.volumes) == num_elements_exp
    assert len([i for i in source_mesh.volumes if i < 0]) == num_neg_vols_exp

    remove_files()


def test_export(source_mesh):

    remove_files()

    source_mesh.create_vertices()
    source_mesh.create_mesh()
    source_mesh.export_mesh()

    assert Path("source_mesh.h5m").exists()

    remove_files()
