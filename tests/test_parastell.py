from pathlib import Path

import numpy as np
import pytest
import dagmc

import parastell.parastell as ps
from parastell.cubit_utils import (
    check_cubit_installation,
    create_new_cubit_instance,
)

files_to_remove = [
    "chamber.step",
    "component.step",
    "magnet_set.step",
    "magnet_mesh.exo",
    "magnet_mesh.h5m",
    "dagmc.h5m",
    "dagmc.cub5",
    "source_mesh.h5m",
    "stellarator.log",
    "step_import.log",
    "step_export.log",
    "magnet_model.h5m",
]


def remove_files():
    for file in files_to_remove:
        if Path(file).exists():
            Path.unlink(file)


def check_surfaces_and_volumes(filename, num_surfaces_exp, num_volumes_exp):
    """Checks whether a DAGMC model has the expected number of surfaces and
    volumes.

    Arguments:
        filename (str): filepath to DAGMC model.
        num_surfaces_exp (int): expected number of surfaces.
        num_volumes_exp (int): expected number of volumes.
    """
    dagmc_model = dagmc.DAGModel(str(Path(filename).with_suffix(".h5m")))

    assert len(dagmc_model.surfaces) == num_surfaces_exp
    assert len(dagmc_model.volumes) == num_volumes_exp


def construct_invessel_build(stellarator_obj, use_pydagmc=False):
    """Constructs the in-vessel build of a Stellarator class object.

    Arguments:
        stellarator_obj (object): parastell.Stellarator class object.
        use_pydagmc (bool): flag to indicate whether ParaStell's PyDAGMC
            workflow should be used to construct the in-vessel build (defaults
            to False).
    """
    toroidal_angles = [0.0, 5.0, 10.0, 15.0]
    poloidal_angles = [0.0, 120.0, 240.0, 360.0]
    wall_s = 1.08
    component_name = "component"
    radial_build_dict = {
        component_name: {
            "thickness_matrix": np.ones(
                (len(toroidal_angles), len(poloidal_angles))
            )
            * 10
        }
    }
    num_ribs = 11

    stellarator_obj.construct_invessel_build(
        toroidal_angles,
        poloidal_angles,
        wall_s,
        radial_build_dict,
        num_ribs=num_ribs,
        use_pydagmc=use_pydagmc,
    )


def create_ivb_cad_magnets_from_filaments(stellarator_obj):
    """Constructs the in-vessel build (CadQuery workflow) and magnet set (from
    filaments) of a Stellarator class object.

    Arguments:
        stellarator_obj (object): parastell.Stellarator class object.
    """
    construct_invessel_build(stellarator_obj)
    stellarator_obj.export_invessel_build()

    coils_file = Path("files_for_tests") / "coils.example"
    width = 40.0
    thickness = 50.0
    toroidal_extent = 90.0
    sample_mod = 6

    stellarator_obj.construct_magnets_from_filaments(
        coils_file, width, thickness, toroidal_extent, sample_mod=sample_mod
    )

    step_filename_exp = "magnet_set.step"

    stellarator_obj.export_magnets(step_filename=step_filename_exp)


def create_ivb_cad_magnets_from_cad(stellarator_obj):
    """Constructs the in-vessel build (CadQuery workflow) and magnet set
    (imported CAD) of a Stellarator class object.

    Arguments:
        stellarator_obj (object): parastell.Stellarator class object.
    """
    construct_invessel_build(stellarator_obj)
    stellarator_obj.export_invessel_build()

    geometry_file = Path("files_for_tests") / "magnet_geom.step"

    stellarator_obj.add_magnets_from_geometry(geometry_file)


def create_ivb_pydagmc_magnets_from_filaments(stellarator_obj):
    """Constructs the in-vessel build (PyDAGMC workflow) and magnet set (from
    filaments) of a Stellarator class object.

    Arguments:
        stellarator_obj (object): parastell.Stellarator class object.
    """
    construct_invessel_build(stellarator_obj, use_pydagmc=True)

    coils_file = Path("files_for_tests") / "coils.example"
    width = 40.0
    thickness = 50.0
    toroidal_extent = 90.0
    sample_mod = 6

    stellarator_obj.construct_magnets_from_filaments(
        coils_file, width, thickness, toroidal_extent, sample_mod=sample_mod
    )

    step_filename_exp = "magnet_set.step"

    stellarator_obj.export_magnets(step_filename=step_filename_exp)


def create_ivb_pydagmc_magnets_from_cad(stellarator_obj):
    """Constructs the in-vessel build (PyDAGMC workflow) and magnet set
    (imported CAD) of a Stellarator class object.

    Arguments:
        stellarator_obj (object): parastell.Stellarator class object.
    """
    construct_invessel_build(stellarator_obj, use_pydagmc=True)

    geometry_file = Path("files_for_tests") / "magnet_geom.step"

    stellarator_obj.add_magnets_from_geometry(geometry_file)


@pytest.fixture
def stellarator():
    vmec_file = Path("files_for_tests") / "wout_vmec.nc"

    stellarator_obj = ps.Stellarator(vmec_file)

    return stellarator_obj


def test_invessel_build(stellarator):
    """Tests whether in-vessel build functionality can be called via the
    Stellarator class.
    """
    remove_files()

    component_name_exp = "component"
    construct_invessel_build(stellarator)

    chamber_filename_exp = Path("chamber").with_suffix(".step")
    component_filename_exp = Path(component_name_exp).with_suffix(".step")

    stellarator.export_invessel_build()

    assert chamber_filename_exp.exists()
    assert component_filename_exp.exists()

    remove_files()


def test_magnet_set(stellarator):
    """Tests whether magnet set functionality can be called via the Stellarator
    class.
    """
    remove_files()

    coils_file = Path("files_for_tests") / "coils.example"
    width = 40.0
    thickness = 50.0
    toroidal_extent = 90.0
    sample_mod = 6

    stellarator.construct_magnets_from_filaments(
        coils_file, width, thickness, toroidal_extent, sample_mod=sample_mod
    )

    step_filename_exp = "magnet_set.step"
    export_mesh = check_cubit_installation()
    mesh_filename_exp = "magnet_mesh"

    stellarator.export_magnets(
        step_filename=step_filename_exp,
        export_mesh=export_mesh,
        mesh_filename=mesh_filename_exp,
    )

    assert Path(step_filename_exp).with_suffix(".step").exists()
    if export_mesh:
        assert Path(mesh_filename_exp).with_suffix(".h5m").exists()

    remove_files()


def test_source_mesh(stellarator):
    """Tests whether source mesh functionality can be called via the
    Stellarator class.
    """
    remove_files()

    mesh_size = (6, 41, 9)
    toroidal_extent = 15.0

    stellarator.construct_source_mesh(mesh_size, toroidal_extent)

    filename_exp = "source_mesh"

    stellarator.export_source_mesh(filename=filename_exp)

    assert Path(filename_exp).with_suffix(".h5m").exists()

    remove_files()


def test_cubit_ps_geom(stellarator):
    """Tests whether the Cubit-DAGMC workflow produces the expected model,
    using magnets constructed from filaments. This test is skipped if Cubit
    cannot be imported.
    """
    pytest.importorskip("cubit")

    remove_files()
    create_new_cubit_instance()

    create_ivb_cad_magnets_from_filaments(stellarator)

    chamber_volume_id_exp = 1
    component_volume_id_exp = 2
    component_name_exp = "component"
    magnet_volume_ids_exp = list(range(3, 4))
    filename_exp = "dagmc"

    # Each in-vessel component (2 present) gives 3 unique surfaces; each magnet
    # (1 present) gives 4 surfaces
    num_surfaces_exp = 10
    num_volumes_exp = 3

    stellarator.build_cubit_model()

    assert (
        stellarator.invessel_build.radial_build.radial_build["chamber"][
            "vol_id"
        ]
        == chamber_volume_id_exp
    )
    assert (
        stellarator.invessel_build.radial_build.radial_build[
            component_name_exp
        ]["vol_id"]
        == component_volume_id_exp
    )
    assert stellarator.magnet_set.volume_ids == magnet_volume_ids_exp

    stellarator.export_cubit_dagmc(filename=filename_exp)
    stellarator.export_cub5(filename=filename_exp)

    assert Path(filename_exp).with_suffix(".h5m").exists()
    assert Path(filename_exp).with_suffix(".cub5").exists()

    check_surfaces_and_volumes(filename_exp, num_surfaces_exp, num_volumes_exp)

    remove_files()


def test_cubit_cad_magnets(stellarator):
    """Tests whether the Cubit-DAGMC workflow produces the expected model,
    using an imported CAD magnet set. This test is skipped if Cubit cannot be
    imported.
    """
    pytest.importorskip("cubit")

    remove_files()
    create_new_cubit_instance()

    create_ivb_cad_magnets_from_cad(stellarator)

    chamber_volume_id_exp = 1
    component_volume_id_exp = 2
    component_name_exp = "component"
    magnet_volume_ids_exp = list(range(3, 4))
    filename_exp = "dagmc"

    # Each in-vessel component (2 present) gives 3 unique surfaces; each magnet
    # (1 present) gives 4 surfaces
    num_surfaces_exp = 10
    num_volumes_exp = 3

    stellarator.build_cubit_model()

    assert (
        stellarator.invessel_build.radial_build.radial_build["chamber"][
            "vol_id"
        ]
        == chamber_volume_id_exp
    )
    assert (
        stellarator.invessel_build.radial_build.radial_build[
            component_name_exp
        ]["vol_id"]
        == component_volume_id_exp
    )
    assert stellarator.magnet_set.volume_ids == magnet_volume_ids_exp

    stellarator.export_cubit_dagmc(filename=filename_exp)
    stellarator.export_cub5(filename=filename_exp)

    assert Path(filename_exp).with_suffix(".h5m").exists()
    assert Path(filename_exp).with_suffix(".cub5").exists()

    check_surfaces_and_volumes(filename_exp, num_surfaces_exp, num_volumes_exp)

    remove_files()


def test_cad_to_dagmc_ps_geom(stellarator):
    """Tests whether the CAD-to-DAGMC workflow produces the expected model,
    using magnets constructed from filaments.
    """
    remove_files()

    create_ivb_cad_magnets_from_filaments(stellarator)

    filename_exp = "dagmc"

    # Each in-vessel component (2 present) gives 3 unique surfaces; each magnet
    # (1 present) gives 4 surfaces
    num_surfaces_exp = 10
    num_volumes_exp = 3

    stellarator.build_cad_to_dagmc_model()
    stellarator.export_cad_to_dagmc(min_mesh_size=50, max_mesh_size=100)

    assert Path(filename_exp).with_suffix(".h5m").exists()

    check_surfaces_and_volumes(filename_exp, num_surfaces_exp, num_volumes_exp)

    remove_files()


def test_cad_to_dagmc_cad_magnets(stellarator):
    """Tests whether the CAD-to-DAGMC workflow produces the expected model,
    using an imported CAD magnet set.
    """
    remove_files()

    create_ivb_cad_magnets_from_cad(stellarator)

    filename_exp = "dagmc"

    # Each in-vessel component (2 present) gives 3 unique surfaces; each magnet
    # (1 present) gives 4 surfaces
    num_surfaces_exp = 10
    num_volumes_exp = 3

    stellarator.build_cad_to_dagmc_model()
    stellarator.export_cad_to_dagmc(min_mesh_size=50, max_mesh_size=100)

    assert Path(filename_exp).with_suffix(".h5m").exists()

    check_surfaces_and_volumes(filename_exp, num_surfaces_exp, num_volumes_exp)

    remove_files()


def test_pydagmc_ps_geom_cubit(stellarator):
    """Tests whether the PyDAGMC workflow produces the expected model, using
    constructed magnets faceted via Cubit. This test is skipped if Cubit cannot
    be imported.
    """
    pytest.importorskip("cubit")

    remove_files()
    create_new_cubit_instance()

    create_ivb_pydagmc_magnets_from_filaments(stellarator)

    # Intentionally pass a kwarg for 'cad_to_dagmc' export to verify that
    # kwargs are filtered appropriately
    stellarator.build_pydagmc_model(
        magnet_exporter="cubit", deviation_angle=6, max_mesh_size=40
    )
    stellarator.export_pydagmc_model("dagmc.h5m")

    # No plasma chamber. 4 surfaces from single magnet, 4 surfaces from single
    # component.
    num_surfaces_exp = 8

    # One magnet and one component
    num_volumes_exp = 2

    check_surfaces_and_volumes("dagmc.h5m", num_surfaces_exp, num_volumes_exp)

    assert Path("dagmc.h5m").exists()

    remove_files()


def test_pydagmc_ps_geom_cad_to_dagmc(stellarator):
    """Tests whether the PyDAGMC workflow produces the expected model, using
    constructed magnets faceted via CAD-to-DAGMC.
    """
    remove_files()

    create_ivb_pydagmc_magnets_from_filaments(stellarator)

    # Intentionally pass a kwarg for 'cubit' export to verify that
    # kwargs are filtered appropriately
    stellarator.build_pydagmc_model(
        magnet_exporter="cad_to_dagmc", deviation_angle=6, max_mesh_size=40
    )
    stellarator.export_pydagmc_model("dagmc.h5m")

    # No plasma chamber. 4 surfaces from single magnet, 4 surfaces from single
    # component.
    num_surfaces_exp = 8

    # One magnet and one component
    num_volumes_exp = 2

    check_surfaces_and_volumes("dagmc.h5m", num_surfaces_exp, num_volumes_exp)

    assert Path("dagmc.h5m").exists()

    remove_files()


def test_pydagmc_cad_magnets_cubit(stellarator):
    """Tests whether the PyDAGMC workflow produces the expected model, using
    imported magnets faceted via Cubit. This test is skipped if Cubit cannot be
    imported.
    """
    pytest.importorskip("cubit")

    remove_files()
    create_new_cubit_instance()

    create_ivb_pydagmc_magnets_from_cad(stellarator)

    # Intentionally pass a kwarg for 'cad_to_dagmc' export to verify that
    # kwargs are filtered appropriately
    stellarator.build_pydagmc_model(
        magnet_exporter="cubit", deviation_angle=6, max_mesh_size=40
    )
    stellarator.export_pydagmc_model("dagmc.h5m")

    # No plasma chamber. 4 surfaces from single magnet, 4 surfaces from single
    # component.
    num_surfaces_exp = 8

    # One magnet and one component
    num_volumes_exp = 2

    check_surfaces_and_volumes("dagmc.h5m", num_surfaces_exp, num_volumes_exp)

    assert Path("dagmc.h5m").exists()

    remove_files()


def test_pydagmc_cad_magnets_cad_to_dagmc(stellarator):
    """Tests whether the PyDAGMC workflow produces the expected model, using
    imported magnets faceted via CAD-to-DAGMC.
    """
    remove_files()

    create_ivb_pydagmc_magnets_from_cad(stellarator)

    # Intentionally pass a kwarg for 'cubit' export to verify that
    # kwargs are filtered appropriately
    stellarator.build_pydagmc_model(
        magnet_exporter="cad_to_dagmc", deviation_angle=6, max_mesh_size=40
    )
    stellarator.export_pydagmc_model("dagmc.h5m")

    # No plasma chamber. 4 surfaces from single magnet, 4 surfaces from single
    # component.
    num_surfaces_exp = 8

    # One magnet and one component
    num_volumes_exp = 2

    check_surfaces_and_volumes("dagmc.h5m", num_surfaces_exp, num_volumes_exp)

    assert Path("dagmc.h5m").exists()

    remove_files()
