![Logo](ParaStell-Logo.png)

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/svalinn/parastell?tab=MIT-1-ov-file#readme)
[![CI testing](https://github.com/svalinn/parastell/actions/workflows/ci.yml/badge.svg)](https://github.com/svalinn/parastell/actions/workflows/ci.yml)
[![Build status](https://github.com/svalinn/parastell/actions/workflows/docker_publish.yml/badge.svg)](https://github.com/svalinn/parastell/actions/workflows/docker_publish.yml)
---

Open-source Python package featuring a parametric 3-D CAD modeling toolset for stellarator fusion devices with additional neutronics support. ParaStell uses plasma equilibrium VMEC data and a user-defined radial build to model in-vessel components of varying thickness in low-fidelity. Furthermore, coil filament point-locus data and a user-defined cross-section are used to model magnet coils. Additional neutronics support includes the generation of tetrahedral neutron source definitions and DAGMC geometries for use in Monte Carlo radiation transport software. In addition, an option is included to generate tetrahedral meshes of in-vessel components and magnets using Coreform Cubit for use in Monte Carlo mesh tallies. A neutron wall-loading utility is included that uses OpenMC to fire rays from a ParaStell neutron source mesh onto a ParaStell first wall CAD geometry.

![Example model](ParaStell-Example.png)

## Dependencies
ParaStell depends on:

- [PyStell](https://github.com/aaroncbader/pystell_uw)
- [CadQuery](https://cadquery.readthedocs.io/en/latest/installation.html)
- [PyDAGMC](https://github.com/svalinn/pydagmc)
- [MOAB](https://bitbucket.org/fathomteam/moab/src/master/)
- [CAD-to-DAGMC](https://github.com/fusion-energy/cad_to_dagmc)
- [OpenMC](https://github.com/openmc-dev/openmc)
- [NumPy](https://numpy.org/install/)
- [SciPy](https://scipy.org/install/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Coreform Cubit](https://coreform.com/products/downloads/) (optional)

## Install ParaStell
Download and extract the ParaStell repository:

```bash
git clone git@github.com:svalinn/parastell.git
```

or download the ZIP file from the repository home page. Once extracted, add the repository directory to your `PYTHONPATH`.

### Install Python Dependencies

This guide will use the conda package manager to install Python dependencies. Conda provides straight-forward installation of Python packages and switching between different collections of Python packages through the use of [environments](https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html).

If you have not already installed conda, you can use one of the following installers to do so:
- [Miniforge](https://github.com/conda-forge/miniforge)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda](https://www.anaconda.com/)

A working conda environment with all ParaStell Python dependencies can be found in this repository's `environment.yml` file. To create the corresponding `parastell_env` conda environment on your machine, create the environment from the `environment.yml` file and activate the new environment:

```bash
conda env create -f environment.yml
conda activate parastell_env
```

Alternatively, view `INSTALL.md` for instructions on manually installing these Python dependencies using mamba.

### Install Coreform Cubit
To make use of ParaStell's Cubit functionality, download and install the latest version from [Coreform's Website](https://coreform.com/products/downloads/), then add the `/Coreform-Cubit-[version]/bin/` directory to your `PYTHONPATH` by adding a line similar to the following to your `.bashrc` file:

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/Coreform-Cubit-[version]/bin/
```

Replace `$HOME` with the path to the Coreform Cubit directory on your system. Additional information about adding modules to your `PYTHONPATH` can be found [here](https://www.tutorialspoint.com/How-to-set-python-environment-variable-PYTHONPATH-on-Linux).
While it is possible to use ParaStell with older versions of Cubit, additional steps not in this guide may be required.

If you do not have a Coreform Cubit license, you may be able to get one through [Cubit Learn](https://coreform.com/products/coreform-cubit/free-meshing-software/) at no cost.

## Executing ParaStell Scripts with YAML Input
While ParaStell can imported as a module to make use of its Python API, ParaStell also has an executable to alternatively call functionality via command line. This executable uses a YAML configuration file as a command-line argument to define input parameters.

To make use of this feature, add the ParaStell `executables` directory to your `PATH` by adding a line similar to the following to your `.bashrc` file:

```bash
export PATH = $PATH:$HOME/parastell/executables/
```

Replace `$HOME` with the path to the ParaStell repository directory on your system. Information about adding directories to your `PATH` can be found [here](https://phoenixnap.com/kb/linux-add-to-path).

Next, give any files in the `executables` directory file execution permission:

```bash
chmod -R u+x $HOME/parastell/executables/
```

Now, the executable can be run from command line with a corresponding YAML file argument. For example:

```bash
parastell config.yaml
```

See the executable's help message for more details.

## Citing
If referencing ParaStell in a document or presentation, please cite the following publication:

- Connor A. Moreno, Aaron Bader, and Paul P.H. Wilson, "ParaStell: parametric modeling and neutronics support for stellarator fusion power plants," *Frontiers in Nuclear Engineering*, **3**:1384788 (2024). DOI: [10.3389/fnuen.2024.1384788](https://www.frontiersin.org/journals/nuclear-engineering/articles/10.3389/fnuen.2024.1384788/full)
