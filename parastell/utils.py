import yaml
import tempfile
from functools import cached_property

import numpy as np
import math
from scipy.ndimage import gaussian_filter
from pymoab import core, types
import dagmc


m2cm = 100
m3tocm3 = m2cm * m2cm * m2cm


def downsample_loop(list, sample_mod):
    """Downsamples a list representing a closed loop.

    Arguments:
        list (iterable): closed loop.
        sample_mod (int): sampling modifier.

    Returns:
        (iterable): downsampled closed loop
    """
    return np.append(list[:-1:sample_mod], [list[0]], axis=0)


def enforce_helical_symmetry(matrix):
    """Ensures that a matrix is helically symmetric according to stellarator
    geometry by overwriting certain matrix elements.

    Assumes several qualities about the input matrix:
        - Regular spacing between angles defining matrix
        - Matrix represents a full stellarator period
        - The first row corresponds to a toroidal angle at the beginning of a
          period (i.e., poloidal symmetry is expected)

    Arguments:
        matrix (2-D iterable of float): matrix to be made helically symmetric.

    Returns:
        matrix (2-D iterable of float): helically symmetric matrix.
    """
    num_rows, num_columns = matrix.shape

    # Ensure rows represent closed loops
    matrix[:, -1] = matrix[:, 0]

    # Ensure poloidal symmetry at beginning of period
    matrix[0] = np.concatenate(
        [
            # Ceil and floor ensure middle element of odd sized array is
            # included only once
            matrix[0, : math.ceil(num_columns / 2)],
            np.flip(matrix[0, : math.floor(num_columns / 2)]),
        ]
    )

    # Ensure helical symmetry toroidally and poloidally by mirroring the period
    # about both matrix axes
    flattened_matrix = matrix.flatten()
    flattened_length = len(flattened_matrix)

    first_half = flattened_matrix[: math.ceil(flattened_length / 2)]
    last_half = np.flip(flattened_matrix[: math.floor(flattened_length / 2)])
    flattened_matrix = np.concatenate([first_half, last_half])

    matrix = flattened_matrix.reshape((num_rows, num_columns))

    return matrix


def expand_list(list_to_expand, num):
    """Expands a list of ordered floats to a total number of entries by
    linearly interpolating between entries, inserting a proportional number of
    new entries between original entries. If num <= len(list), list_to_expand
    is not modified. It is possible that the result will have slightly more
    or fewer elements than num, due to round off approximations.

    Arguments:
        list_to_expand (iterable of float): list to be expanded.
        num (int): desired number of entries in expanded list.

    Returns:
        list_exp (iterable of float): expanded list.
    """
    if len(list_to_expand) >= num:
        return list_to_expand

    list_exp = []

    init_entry = list_to_expand[0]
    final_entry = list_to_expand[-1]
    extent = final_entry - init_entry

    avg_diff = extent / (num - 1)

    for entry, next_entry in zip(list_to_expand[:-1], list_to_expand[1:]):
        # Only add entries to current block if difference between entry and
        # next_entry is greater than desired average
        num_new_entries = 0

        if next_entry - entry > avg_diff:
            # Goal is to create bins of approximately avg_diff width between
            # entry and next_entry
            num_new_entries = int(round((next_entry - entry) / avg_diff)) - 1

        # Manually append first entry
        list_exp = np.append(list_exp, entry)

        # If num_new_entries == 0, don't add new entries
        # First and last elements of new_entries are entry and next_entry,
        # respectively
        new_entries = np.linspace(
            entry,
            next_entry,
            num=num_new_entries + 2,
        )[1:-1]

        list_exp = np.append(list_exp, new_entries)

    list_exp = np.append(list_exp, final_entry)

    return list_exp


def filter_kwargs(
    dict, allowed_kwargs, all_kwargs=False, fn_name=None, logger=None
):
    """Constructs a dictionary of keyword arguments with corresponding values
    for a class method from an input dictionary based on a list of allowable
    keyword argument argument names. Conditionally raises an exception if the
    user flags that the supplied list of allowable arguments should represent
    all keys present in the input dictionary of arguments.

    Arguments:
        dict (dict): dictionary of arguments and corresponding values.
        allowed_kwargs (list of str): list of allowed keyword argument names.
        all_kwargs (bool): flag to indicate whether 'allowed_kwargs' should
            represent all keys present in 'dict' (optional, defaults to False).
        fn_name (str): name of class method (optional, defaults to None). If
            'all_kwargs' is True, a method name should be supplied.
        logger (object): logger object (optional, defaults to None). If
            'all_kwargs' is True, a logger object should be supplied.

    Returns:
        kwarg_dict (dict): dictionary of keyword arguments and values.
    """
    allowed_keys = dict.keys() & allowed_kwargs
    extra_keys = dict.keys() - allowed_kwargs

    if all_kwargs and extra_keys:
        e = ValueError(
            f"{extra_keys} not supported keyword argument(s) of "
            f'"{fn_name}"'
        )
        logger.error(e.args[0])
        raise e

    return {name: dict[name] for name in allowed_keys}


def normalize(vec_list):
    """Normalizes a set of vectors.

    Arguments:
        vec_list (1 or 2D np array): single 1D vector or array of 1D vectors
            to be normalized
    Returns:
        vec_list (np array of same shape as input): single 1D normalized vector
            or array of normalized 1D vectors
    """
    if len(vec_list.shape) == 1:
        return vec_list / np.linalg.norm(vec_list)
    elif len(vec_list.shape) == 2:
        return vec_list / np.linalg.norm(vec_list, axis=1)[:, np.newaxis]
    else:
        print('Input "vec_list" must be 1-D or 2-D NumPy array')


def read_yaml_config(filename):
    """Read YAML file describing ParaStell configuration and extract all data."""
    with open(filename) as yaml_file:
        all_data = yaml.safe_load(yaml_file)

    return all_data


def reorder_loop(list, index):
    """Reorders a list representing a closed loop.

    Arguments:
        list (iterable): closed loop.
        index (int): list index about which to reorder loop.

    Returns:
        (iterable): reordered closed loop.
    """
    return np.concatenate([list[index:], list[1 : index + 1]])


def smooth_matrix(matrix, steps, sigma):
    """Smooths a matrix via Gaussian filtering, without allowing matrix
    elements to increase in value.

    Arguments:
        matrix (2-D iterable of float): matrix to be smoothed.
        steps (int): number of smoothing steps.
        sigma (float): standard deviation for Gaussian kernel.

    Returns:
        smoothed_matrix (2-D iterable of float): smoothed matrix.
    """
    previous_matrix = matrix

    for step in range(steps):
        smoothed_matrix = np.minimum(
            previous_matrix,
            gaussian_filter(
                previous_matrix,
                sigma=sigma,
                mode="wrap",
            ),
        )
        previous_matrix = smoothed_matrix

    return smoothed_matrix


class DAGMCRenumberizer(object):
    """Class to facilitate renumbering of entities to combine DAGMC models.

    Arguments:
        mb (PyMOAB Core): PyMOAB core with a DAGMC models to be renumbered
        loaded. Optional.
    """

    def __init__(self, mb=None):
        self.mb = mb
        if mb is None:
            self.mb = core.Core()

    @cached_property
    def global_id_tag(self):
        return self.mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE
        )

    @cached_property
    def category_tag(self):
        """Returns the category tag used to intidate the use of meshset. Values
        include "Group", "Volume", "Surface", "Curve" and "Vertex".
        """
        return self.mb.tag_get_handle(
            types.CATEGORY_TAG_NAME,
            types.CATEGORY_TAG_SIZE,
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )

    def load_file(self, filename):
        """Load DAGMC model from file.

        Arguments:
            filename (str): Path to DAGMC model to be loaded.
        """
        self.mb.load_file(filename)

    def renumber_ids(self):
        """Renumbers the ids from 1 to N where N is the total number of
        entities in that category.
        """
        categories = ["Vertex", "Curve", "Surface", "Volume", "Group"]
        root_set = self.mb.get_root_set()
        for category in categories:
            category_set = self.mb.get_entities_by_type_and_tag(
                root_set, types.MBENTITYSET, self.category_tag, [category]
            )
            num_ids = len(category_set)
            if num_ids != 0:
                set_ids = list(range(1, num_ids + 1))
                self.mb.tag_set_data(
                    self.global_id_tag,
                    category_set,
                    set_ids,
                )


def combine_dagmc_models(models_to_merge):
    """Takes a list of DAGMC models, and renumbers entity ids such that they
    will no longer clash, allowing the models to be loaded into the same
    PyMOAB core instance, and saved to a single model.

    Arguments:
        models_to_merge (list of PyMOAB core): List of DAGMC models to be
            merged.
    Returns:
        combined_model (dagmc.DAGModel): Single DAGMC model containing the
            combined individual models.
    """
    renumberizer = DAGMCRenumberizer()
    for model in models_to_merge:
        with tempfile.NamedTemporaryFile(
            delete=True, suffix=".h5m"
        ) as temp_file:
            temp_filename = temp_file.name
            model.write_file(temp_filename)
            renumberizer.load_file(temp_filename)
    renumberizer.renumber_ids()
    return dagmc.DAGModel(renumberizer.mb)


def rotate_ribs(ribs, angle):
    """Rotate a set of NxMx3 set of loci about the Z axis in the
    counter-clockwise direction.

    Arguments:
        ribs (numpy array): NxMx3 array of of cartesian points. The first
            dimension corresponds to the plane of constant toroidal angle on
            which the closed loop of points lies. The second dimension is the
            location on the closed loop at which the point lies, and the third
            dimension is the x,y,z value of that point.
        angle (float): Amount by which to rotate the points in ribs. Measured
            in degrees, positive in right hand direction about the Z axis.

    Return:
        ribs (numpy array): Array of the same shape as the ribs argument,
            with each point rotated by angle about the Z axis.
    """
    angle = np.deg2rad(angle)
    rotation_mat = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    ribs = np.dot(ribs, rotation_mat.T)
    return ribs


def ribs_from_kisslinger_format(filename, start_line=2, scale=m2cm):
    """Reads a Kisslinger format file and returns a list of toroidal angles,
    along with a list of lists of the rib loci (in x,y,z) at each toroidal
    angle. It is expected that toroidal angles are provided in degrees.

    Arguments:
        filename (str): Path to the file to be read.
        start_line (int): Line at which the data should start being read,
            Defaults to 2.
        scale (float): Amount to scale the r-z coordinates by. Defaults to 100.
    Returns:
        toroidal_angles (numpy array): Toroidal angles in the
            angles in the input file in degrees.
        num_toroidal_angles (int): Number of toroidal angles as specified in
            the file header.
        num_poloidal_angles (int): Number of points at each toroidal angle.
        periods (int): Number of periods as specified in the file header.
        profiles (numpy array): 3 dimensional numpy array where the first
            dimension corresponds to individual ribs, the second to the
            position on a rib, and the third to the actual x,y,z coordinate
            of the point.
    """

    with open(file=filename) as file:
        data = file.readlines()

    profiles = []
    profile = []
    toroidal_angles = []

    for line in data[start_line - 1 :]:
        if line.count("\t") == 2:
            num_toroidal_angles, num_poloidal_angles, periods = (
                line.rstrip().split("\t")
            )
        elif "\t" not in line and "#" not in line:
            toroidal_angle = float(line.rstrip())
            toroidal_angles.append(toroidal_angle)
            if len(profile) != 0:
                profiles.append(profile)
                profile = []
        else:
            line = line.rstrip()
            r_z_coords = [float(coord) * scale for coord in line.split("\t")]
            x_coord = r_z_coords[0] * np.cos(np.deg2rad(toroidal_angle))
            y_coord = r_z_coords[0] * np.sin(np.deg2rad(toroidal_angle))
            z_coord = r_z_coords[1]
            profile.append([x_coord, y_coord, z_coord])
    profiles.append(profile)

    return (
        np.array(toroidal_angles),
        int(num_toroidal_angles),
        int(num_poloidal_angles),
        int(periods),
        np.array(profiles),
    )
