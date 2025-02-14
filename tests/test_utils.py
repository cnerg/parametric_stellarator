import pytest
from parastell.utils import *


def test_dagmc_renumbering():
    """Tests whether multiple DAGMC models are correctly combined, renumbering
    their IDs, using the example of three cubes. two_cubes.h5m is two cubes
    which share a face. one_cube.h5m has one cube. They do not overlap.
    """
    num_surfaces_exp = 17
    max_surf_id_exp = 17
    num_vol_exp = 3
    max_vol_id_exp = 3

    mats_exp = ["iron", "tungsten", "air"]

    core_1 = core.Core()
    core_2 = core.Core()
    core_1.load_file("files_for_tests/two_cubes.h5m")
    core_2.load_file("files_for_tests/one_cube.h5m")

    combined_model = combine_dagmc_models([core_1, core_2])

    mats = [vol.material for vol in combined_model.volumes]

    assert len(combined_model.surfaces) == num_surfaces_exp
    assert max(combined_model.surfaces_by_id.keys()) == max_surf_id_exp
    assert len(combined_model.volumes) == num_vol_exp
    assert max(combined_model.volumes_by_id.keys()) == max_vol_id_exp
    assert all(mat in mats_exp for mat in mats)
