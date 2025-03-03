import pytest
from parastell.utils import *


def test_dagmc_renumbering():
    """Tests whether multiple DAGMC models are correctly combined, renumbering
    their IDs, by testing if:
        * the expected number of surfaces is created
        * the resulting largest surface ID is the expected value
        * the expected number of volumes is created
        * the resulting largest volume ID is the expected value
        * the correct material tags, in the correct order, are applied

    An example of three cubes is used for this test. two_cubes.h5m is two cubes
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


def test_expand_list():
    """Tests utils.expand_list() to ensure returned arrays are the length
    expected, and contain the expected values, by testing if:
        * the expected entries are added to uniformly and non-uniformly spaced
          lists
        * entries are added when the requested size is less than or equal to
          that of the input list (no entries should be added)
    """
    # Make sure new entries are inserted as expected
    test_values = np.linspace(1, 10, 10)
    exp_expanded_list = np.linspace(1, 10, 19)
    expanded_list = expand_list(test_values, 19)
    assert np.allclose(exp_expanded_list, expanded_list)

    # Make sure no changes are made if list already has the requested number of
    # entries
    expanded_list = expand_list(test_values, 10)
    assert len(expanded_list) == len(test_values)
    assert np.allclose(expanded_list, test_values)

    # Make sure no changes are made if list has more than the requested number
    # of entries
    expanded_list = expand_list(test_values, 5)
    assert len(expanded_list) == len(test_values)
    assert np.allclose(expanded_list, test_values)

    # Make sure it works with unevenly spaced entries
    test_values = [1, 5, 6, 7, 10]
    exp_expanded_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    expanded_list = expand_list(test_values, 10)
    assert np.allclose(expanded_list, exp_expanded_list)

    # Make sure it works with unevenly spaced entries that are not
    # nicely divisible
    test_values = [1, 4.5, 6, 7, 10]
    expanded_list = expand_list(test_values, 5)
    assert len(expanded_list) == 5

    # int math makes this list one element longer than requested
    test_values = [1, 4.5, 6, 7, 10]
    expected_values = [1, 1.875, 2.75, 3.625, 4.5, 5.25, 6, 7, 8, 9, 10]
    expanded_list = expand_list(test_values, 10)
    assert len(expanded_list) == 11
    assert np.allclose(expected_values, expanded_list)


def test_rib_rotation():
    """Tests utils.rotate_ribs() to ensure rotations are as expected
    by testing if:
        * a set of points rotated 90 degrees about the z axis has the expected
          values
    """
    # fmt: off
    original_points = np.array(
        [
            [[1, 1, 1], [1, 2, 1], [3, 2, 0]],
            [[1, 1, 1], [1, 2, 1], [3, 2, 0]],
        ]
    )
    # fmt: on
    rotation_angle = 90
    rotated_points_exp = np.array(
        [
            [[-1, 1, 1], [-2, 1, 1], [-2, 3, 0]],
            [[-1, 1, 1], [-2, 1, 1], [-2, 3, 0]],
        ]
    )
    rotated_points = rotate_ribs(original_points, rotation_angle)
    assert np.allclose(rotated_points, rotated_points_exp)
