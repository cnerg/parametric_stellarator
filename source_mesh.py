import log
from pymoab import core, types
import numpy as np


def rxn_rate(s):
    """Calculates fusion reaction rate in plasma.

    Arguments:
        s (float): closed magnetic flux surface index in range of 0 (magnetic
            axis) to 1 (plasma edge).

    Returns:
        rr (float): fusion reaction rate (1/cm^3/s). Equates to neutron source
            density.
    """
    if s == 1:
        rr = 0
    else:
        # Temperature
        T = 11.5*(1 - s)
        # Ion density
        n = 4.8e20*(1 - s**5)
        # Reaction rate in 1/m^3/s
        rr = 3.68e-18*(n**2)/4*T**(-2/3)*np.exp(-19.94*T**(-1/3))
        # Define m^3 to cm^3 constant
        m3tocm3 = 1e6
        
        rr = rr/m3tocm3

    return rr


def source_strength(verts, verts_s, ids):
    """Computes neutron source strength for a tetrahedron using five-node
    Gaussian quadrature.

    Arguments:
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface labels for each
            vertex.
        ids (list of int): tetrahedron vertex indices.

    Returns:
        ss (float): integrated source strength for tetrahedron.
    """
    # Initialize list of coordinates for each tetrahedron vertex
    tet_verts = []
    # Initialize list of source strengths for each tetrahedron vertex
    ss_verts = []

    # Define vertices for tetrahedron
    for id in ids:
        tet_verts += [verts[id]]
        ss_verts += [rxn_rate(verts_s[id])]

    # Define barycentric coordinates for integration points
    bary_coords = np.array([
        [0.25, 0.25, 0.25, 0.25],
        [0.5, 1/6, 1/6, 1/6],
        [1/6, 0.5, 1/6, 1/6],
        [1/6, 1/6, 0.5, 1/6],
        [1/6, 1/6, 1/6, 0.5]
    ])

    # Define weights for integration points
    int_w = np.array([-0.8, 0.45, 0.45, 0.45, 0.45])
    
    # Interpolate source strength at integration points
    ss_int_pts = np.dot(bary_coords, ss_verts)
    
    # Compute graph of tetrahedral vertices
    T = np.subtract(tet_verts[:3], tet_verts[3]).transpose()
    
    tet_vol = np.abs(np.linalg.det(T))/6
    
    ss = tet_vol * np.dot(int_w, ss_int_pts)

    return ss


def create_tet(
    mbc, tag_handle, mesh_set, mbc_verts, verts, verts_s, ids, strengths):
    """Creates tetrahedron and adds to moab core.

    Arguments:
        mbc (object): PyMOAB core instance.
        tag_handle (TagHandle): PyMOAB source strength tag.
        mesh_set (EntityHandle): PyMOAB mesh set.
        mbc_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface labels for each
            vertex.
        ids (list of int): tetrahedron vertex indices.
        strengths (list of float): list of source strengths for each
            tetrahedron (1/s).
    """
    # Define vertices for tetrahedron
    tet_verts = [
            mbc_verts[ids[0]],
            mbc_verts[ids[1]],
            mbc_verts[ids[2]],
            mbc_verts[ids[3]]
        ]
    # Create tetrahedron in PyMOAB
    tet = mbc.create_element(types.MBTET, tet_verts)
    mbc.add_entity(mesh_set, tet)
    
    # Compute source strength for tetrahedron
    ss = source_strength(verts, verts_s, ids)
    strengths.append(ss)

    # Tag tetrahedra with source strength data
    mbc.tag_set_data(tag_handle, tet, [ss])
    

def create_tets_from_hex(
    mbc, tag_handle, mesh_set, mbc_verts, verts, verts_s, ids_hex, strengths):
    """Creates five tetrahedra from defined hexahedron.

    Arguments:
        mbc (object): PyMOAB core instance.
        tag_handle (TagHandle): PyMOAB source strength tag.
        mesh_set (EntityHandle): PyMOAB mesh set.
        mbc_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface labels for each
            vertex.
        ids_hex (list of int): list of hexahedron vertex indices.
        strengths (list of float): list of source strengths for each
            tetrahedron (1/s).
    """
    # Define MOAB canonical ordering of hexahedron vertex indices
    hex_canon_ids = [
        [ids_hex[0], ids_hex[3], ids_hex[1], ids_hex[4]],
        [ids_hex[7], ids_hex[4], ids_hex[6], ids_hex[3]],
        [ids_hex[2], ids_hex[1], ids_hex[3], ids_hex[6]],
        [ids_hex[5], ids_hex[6], ids_hex[4], ids_hex[1]],
        [ids_hex[3], ids_hex[1], ids_hex[4], ids_hex[6]]
    ]
    
    for vertex_ids in hex_canon_ids:
        create_tet(
            mbc, tag_handle, mesh_set, mbc_verts, verts, verts_s, vertex_ids,
            strengths
        )


def create_tets_from_wedge(
    mbc, tag_handle, mesh_set, mbc_verts, verts, verts_s, ids_wedge, strengths):
    """Creates three tetrahedra from defined wedge.

    Arguments:
        mbc (object): PyMOAB core instance.
        tag_handle (TagHandle): PyMOAB source strength tag.
        mesh_set (EntityHandle): PyMOAB mesh set.
        mbc_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface labels for each
            vertex.
        ids_wedge (list of int): list of wedge vertex indices.
        strengths (list of float): list of source strengths for each
            tetrahedron (1/s).
    """
    # Define MOAB canonical ordering of wedge vertex indices
    wedge_canon_ids = [
        [ids_wedge[1], ids_wedge[2], ids_wedge[4], ids_wedge[0]],
        [ids_wedge[5], ids_wedge[4], ids_wedge[2], ids_wedge[3]],
        [ids_wedge[0], ids_wedge[2], ids_wedge[4], ids_wedge[3]]
    ]
    
    for vertex_ids in wedge_canon_ids:
        create_tet(
            mbc, tag_handle, mesh_set, mbc_verts, verts, verts_s, vertex_ids,
            strengths
        )


def get_vertex_id(vert_ids, num_phi, num_s, num_theta, tor_ext):
    """Computes vertex index in row-major order as stored by MOAB from
    three-dimensional n x 3 matrix indices.

    Arguments:
        vert_ids (list of int): list of vertex indices expressed in n x 3
            matrix order. The list format is:
            [toroidal angle index, flux surface index, poloidal angle index]
        num_phi (int): number of toroidal angles defining mesh.
        num_s (int): number of closed magnetic flux surfaces defining mesh.
        num_theta (int): number of poloidal angles defining mesh.
        tor_ext (float): toroidal extent to model (deg).
    
    Returns:
        id (int): vertex index in row-major order as stored by MOAB
    """
    # Compute index offset from magnetic axis
    if tor_ext == 2*np.pi and vert_ids[0] == num_phi - 1:
        # Ensure logical connectivity at final toroidal angle
        ma_offset = 0
    else:
        ma_offset = vert_ids[0] * (1 + ((num_s - 1) * (num_theta - 1)))
    
    # Compute index offset from closed flux surface
    s_offset = vert_ids[1] * (num_theta - 1)
    
    # Compute index offset from poloidal angle
    if vert_ids[2] == num_theta:
        # Ensure logical connectivity at final poloidal angle
        theta_offset = 1
    else:
        theta_offset = vert_ids[2]
    
    id = ma_offset + s_offset + theta_offset

    return id


def create_mesh(
    mbc, tag_handle, num_s, num_theta, num_phi, tor_ext, mbc_verts, verts,
    verts_s):
    """Creates volumetric source mesh in real space.

    Arguments:
        mbc (object): PyMOAB core instance.
        tag_handle (TagHandle): PyMOAB source strength tag.
        num_s (int): number of closed magnetic flux surfaces defining mesh.
        num_theta (int): number of poloidal angles defining mesh.
        num_phi (int): number of toroidal angles defining mesh.
        tor_ext (float): toroidal extent to model (deg).
        mbc_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface labels for each
            vertex.
            
    Returns:
        strengths (list of float): list of source strengths for each
            tetrahedron (1/s).
    """
    # Instantiate tetrahedra mesh set in PyMOAB
    tet_set = mbc.create_meshset()
    # Add vertices to mesh set
    mbc.add_entity(tet_set, mbc_verts)

    strengths = []
    
    for i in range(num_phi - 1):
        # Create tetrahedra for wedges at center of plasma
        for k in range(1, num_theta):
            # Define six wedge vertex indices in matrix format as well as index
            # offset accounting for magnetic axis vertex in the form
            # [toroidal index, flux surface index, poloidal index]
            wedge_id_data = [
                [i,     0, 0,   ],
                [i,     0, k,   ],
                [i,     0, k + 1],
                [i + 1, 0, 0,   ],
                [i + 1, 0, k,   ],
                [i + 1, 0, k + 1]
            ]
            
            ids_wedge = []
            
            # Define vertex indices in row-major order for wedges
            for vertex in wedge_id_data:
                ids_wedge += [
                    get_vertex_id(vertex, num_phi, num_s, num_theta, tor_ext)
                ]
            
            create_tets_from_wedge(
                mbc, tag_handle, tet_set, mbc_verts, verts, verts_s, ids_wedge,
                strengths
            )

        # Create tetrahedra for hexahedra beyond center of plasma
        for j in range(num_s - 2):
            for k in range(1, num_theta):
                # Define eight hexahedron vertex indices in matrix format as
                # well as index offset accounting for magnetic axis vertex in
                # the form
                # [toroidal index, flux surface index, poloidal index]
                hex_id_data = [
                    [i,     j,     k    ],
                    [i,     j + 1, k,   ],
                    [i,     j + 1, k + 1],
                    [i,     j,     k + 1],
                    [i + 1, j,     k,   ],
                    [i + 1, j + 1, k,   ],
                    [i + 1, j + 1, k + 1],
                    [i + 1, j,     k + 1]
                ]
                
                ids_hex = []
                
                # Define vertex indices in row-major order for hexahedra
                for vertex in hex_id_data:
                    ids_hex += [
                        get_vertex_id(
                            vertex, num_phi, num_s, num_theta, tor_ext
                        )
                    ]
                
                create_tets_from_hex(
                    mbc, tag_handle, tet_set, mbc_verts, verts, verts_s,
                    ids_hex, strengths
                )

    return strengths


def create_vertices(vmec, mbc, num_s, num_theta, num_phi, tor_ext):
    """Creates mesh vertices and adds them to PyMOAB core.

    Arguments:
        vmec (object): plasma equilibrium VMEC object.
        mbc (object): PyMOAB core instance.
        num_s (int): number of closed magnetic flux surfaces defining mesh.
        num_theta (int): number of poloidal angles defining mesh.
        num_phi (int): number of toroidal angles defining mesh.
        tor_ext (float): toroidal extent to model (deg).

    Returns:
        mbc_verts (list of EntityHandle): list of mesh vertices.
        verts (list of list of float): list of 3D Cartesian coordinates of each
            vertex in form [x (cm), y (cm), z (cm)].
        verts_s (list of float): list of closed flux surface labels for each
            vertex.
    """
    # Compute total number of vertices
    # Compute regular spacing between toroidal angles in confinement space to
    # be included in mesh
    if tor_ext == 2*np.pi:
        num_verts = (num_phi - 1) * (1 + ((num_s - 1) * (num_theta - 1)))
        phi_list = np.linspace(0, tor_ext, num = num_phi)[:-1]
    else:
        num_verts = num_phi * (1 + ((num_s - 1) * (num_theta - 1)))
        phi_list = np.linspace(0, tor_ext, num = num_phi)
    
    # Compute regular spacing between closed flux surface labels in confinement
    # space to be included in mesh
    s_list = np.linspace(0.0, 1.0, num = num_s)[1:]
    
    # Compute regular spacing between poloidal angles in confinement space to
    # be included in mesh
    theta_list = np.linspace(0, 2*np.pi, num = num_theta)[:-1]

    # Initialize n x 3 array of vertex coordinates
    verts = np.zeros((num_verts, 3))
    # Initialize array of vertex closed flux surface labels
    verts_s = np.zeros(num_verts)

    # Define conversion from m to cm
    m2cm = 100
    
    # Initialize vertex index
    i_vert = 0
    
    for phi in phi_list:
        # Determine vertex at magnetic axis, converting to cm
        verts[i_vert,:] = np.array(vmec.vmec2xyz(0, 0, phi)) * m2cm
        verts_s[i_vert] = 0
        
        i_vert += 1

        # Compute off-axis vertices at specified toroidal angle
        for s in s_list:
            for theta in theta_list:
                # Determine vertex at specified toroidal angle, closed flux
                # surface, poloidal angle
                verts[i_vert,:] = np.array(vmec.vmec2xyz(s, theta, phi)) * m2cm
                verts_s[i_vert] = s
                
                i_vert += 1

    # Create vertices in PyMOAB
    mbc_verts = mbc.create_vertices(verts)

    return mbc_verts, verts, verts_s


def create_mbc():
    """Creates PyMOAB core instance with source strength tag.

    Returns:
        mbc (object): PyMOAB core instance.
        tag_handle (TagHandle): PyMOAB source strength tag.
    """
    # Create PyMOAB core instance
    mbc = core.Core()

    # Define data type for source strength tag
    tag_type = types.MB_TYPE_DOUBLE
    # Define tag size for source strength tag (1 double value)
    tag_size = 1
    # Define storage type for source strength
    storage_type = types.MB_TAG_DENSE
    # Define tag handle for source strength
    tag_handle = mbc.tag_get_handle(
        "SourceStrength", tag_size, tag_type, storage_type,
        create_if_missing = True
    )

    return mbc, tag_handle


def source_mesh(vmec, source, logger = None):
    """Creates H5M volumetric mesh defining fusion source using PyMOAB and
    user-supplied plasma equilibrium VMEC data.

    Arguments:
        vmec (object): plasma equilibrium VMEC object.
        source (dict): dictionary of source mesh parameters including
            {
                'num_s': number of closed magnetic flux surfaces defining mesh
                    (int),
                'num_theta': number of poloidal angles defining mesh (int),
                'num_phi': number of toroidal angles defining mesh (int),
                'tor_ext': toroidal extent to model (float, deg)
            }
        logger (object): logger object (defaults to None). If no logger is
            supplied, a default logger will be instantiated.

    Returns:
        strengths (list of float): list of source strengths for each
            tetrahedron (1/s).
    """
    if logger == None or not logger.hasHandlers():
        logger = log.init()

    logger.info(f'Building SourceMesh.h5m...')

    num_s = source['num_s']
    num_theta = source['num_theta']
    num_phi = source['num_phi']
    tor_ext = source['tor_ext']
    tor_ext = np.deg2rad(tor_ext)

    # Instantiate PyMOAB core instance and define source strength tag
    mbc, tag_handle = create_mbc()

    # Define mesh vertices in real and confinement space and add to PyMOAB core
    mbc_verts, verts, verts_s = create_vertices(
        vmec, mbc, num_s, num_theta, num_phi, tor_ext
    )

    # Create source mesh
    strengths = create_mesh(
        mbc, tag_handle, num_s, num_theta, num_phi, tor_ext, mbc_verts, verts,
        verts_s
    )

    mbc.write_file("SourceMesh.h5m")

    return strengths
