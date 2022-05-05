#!/bin/env python

import numpy as np
import unyt
from halo_properties import HaloProperty

from dataset_names import mass_dataset


class SOProperties(HaloProperty):

    # Arrays which must be read in for this calculation.
    # Note that if there are no particles of a given type in the
    # snapshot, that type will not be read in and will not have
    # an entry in the data argument to calculate(), below.
    # (E.g. gas, star or BH particles in DMO runs)
    particle_properties = {
        "PartType0": [
            "Coordinates",
            "Masses",
            "Velocities",
            "Temperatures",
            "XrayLuminosities",
            "XrayPhotonLuminosities",
            "ComptonYParameters",
        ],
        "PartType1": ["Coordinates", "Masses", "Velocities"],
        "PartType4": ["Coordinates", "Masses", "InitialMasses", "Velocities"],
        "PartType5": ["Coordinates", "DynamicalMasses", "SubgridMasses", "Velocities"],
    }

    # Minimum physical radius to read in (pMpc)
    physical_radius_mpc = 0.0

    def __init__(self, cellgrid, SOval, type="mean"):
        super().__init__(cellgrid)

        if not type in ["mean", "crit"]:
            raise AttributeError(f"Unknown SO type: {type}!")
        self.type = type

        # This specifies how large a sphere is read in:
        self.mean_density_multiple = SOval
        self.critical_density_multiple = SOval

        # Give this calculation a name so we can select it on the command line
        self.name = "SO_%d_%s" % (SOval, type)

    def calculate(self, input_halo, data, halo_result):
        """
        Compute spherical masses and overdensities for a halo

        input_halo       - dict with halo properties passed in from VR (see
                           halo_centres.py)
        data             - contains particle data. E.g. data["PartType1"]["Coordinates"]
                           has the particle coordinates for type 1
        halo_result      - dict with halo properties computed so far. Properties
                           computed here should be added to halo_result.

        Input particle data arrays are unyt_arrays.
        """

        # Find the halo centre of potential
        centre = input_halo["cofp"]

        # Make an array of particle masses, radii and positions
        mass = []
        radius = []
        position = []
        velocity = []
        types = []
        for ptype in data:
            mass.append(data[ptype][mass_dataset(ptype)])
            pos = data[ptype]["Coordinates"] - centre[None, :]
            position.append(pos)
            r = np.sqrt(np.sum(pos ** 2, axis=1))
            radius.append(r)
            velocity.append(data[ptype]["Velocities"])
            typearr = np.zeros(r.shape, dtype="U9")
            typearr[:] = ptype
            types.append(typearr)
        mass = unyt.array.uconcatenate(mass)
        radius = unyt.array.uconcatenate(radius)
        position = unyt.array.uconcatenate(position)
        velocity = unyt.array.uconcatenate(velocity)
        types = np.concatenate(types)
        nr_parts = mass.shape[0]

        # Sort by radius
        order = np.argsort(radius)
        ordered_radius = radius[order]
        cumulative_mass = np.cumsum(mass[order], dtype=np.float64)

        # Compute density within radius of each particle.
        # Will need to skip any at zero radius.
        nskip = 0
        while nskip < len(ordered_radius) and ordered_radius[nskip] == 0:
            nskip += 1
        ordered_radius = ordered_radius[nskip:]
        cumulative_mass = cumulative_mass[nskip:]
        nr_parts = len(ordered_radius)
        density = cumulative_mass / (4.0 / 3.0 * np.pi * ordered_radius ** 3)

        if self.type == "crit":
            reference_density = self.critical_density_multiple * self.critical_density
            name = f"{self.critical_density_multiple:.0f}_crit"
            label = f"within which the density is {self.critical_density_multiple:.0f} times the critical value"
        elif self.type == "mean":
            reference_density = self.mean_density_multiple * self.mean_density
            name = f"{self.mean_density_multiple:.0f}_mean"
            label = f"within which the density is {self.mean_density_multiple:.0f} times the mean value"

        # Check if we ever reach the density threshold
        if nr_parts > 0 and np.any(density > reference_density):
            # Find smallest radius where the density is below the threshold
            i = np.argmax(density < reference_density)
            mSO = cumulative_mass[i]
            rSO = ordered_radius[i]
        else:
            # Below threshold at all radii. Need to return zero with correct units attached.
            mSO = unyt.unyt_array(
                0, dtype=cumulative_mass.dtype, units=cumulative_mass.units
            )
            rSO = unyt.unyt_array(0, dtype=radius.dtype, units=radius.units)

        Tgas = data["PartType0"]["Temperatures"]
        Xray_lum = data["PartType0"]["XrayLuminosities"]
        Xray_phlum = data["PartType0"]["XrayPhotonLuminosities"]
        compY = data["PartType0"]["ComptonYParameters"]
        if rSO > 0.0 * radius.units:
            gas_selection = radius[types == "PartType0"] < rSO
            star_selection = radius[types == "PartType4"] < rSO
            bh_selection = radius[types == "PartType5"] < rSO

            all_selection = radius < rSO
            mass = mass[all_selection]
            position = position[all_selection]
            velocity = velocity[all_selection]
            types = types[all_selection]

            comSO = (mass[:, None] * position).sum(axis=0) / mSO
            comSO += centre
            vcomSO = (mass[:, None] * velocity).sum(axis=0) / mSO
            gas_masses = mass[types == "PartType0"]
            MgasSO = gas_masses.sum()
            MdmSO = mass[types == "PartType1"].sum()
            MstarSO = mass[types == "PartType4"].sum()
            MBHdynSO = mass[types == "PartType5"].sum()

            gas_temperatures = Tgas[gas_selection]
            Tgas_selection = gas_temperatures > 1.0e5 * unyt.K
            MhotgasSO = gas_masses[Tgas_selection].sum()

            if np.any(Tgas_selection):
                TgasSO = (
                    gas_temperatures[Tgas_selection] * gas_masses[Tgas_selection]
                ).sum() / MhotgasSO
            else:
                # Handle the case where there is no gas above 1e5K
                TgasSO = unyt.unyt_array(0.0, dtype=gas_temperatures.dtype, units=gas_temperatures.units)

            XraylumSO = Xray_lum[gas_selection].sum()
            XrayphlumSO = Xray_phlum[gas_selection].sum()

            compYSO = compY[gas_selection].sum()

            MstarinitSO = data["PartType4"]["InitialMasses"][star_selection].sum()
            MBHsubSO = data["PartType5"]["SubgridMasses"][bh_selection].sum()

        else:

            comSO = unyt.unyt_array(
                [0.0, 0.0, 0.0], dtype=radius.dtype, units=radius.units
            )
            vcomSO = unyt.unyt_array(
                [0.0, 0.0, 0.0], dtype=velocity.dtype, units=velocity.units
            )
            MgasSO = unyt.unyt_array(0.0, dtype=mSO.dtype, units=mSO.units)
            MdmSO = unyt.unyt_array(0.0, dtype=mSO.dtype, units=mSO.units)
            MstarSO = unyt.unyt_array(0.0, dtype=mSO.dtype, units=mSO.units)
            MBHdynSO = unyt.unyt_array(0.0, dtype=mSO.dtype, units=mSO.units)
            MhotgasSO = unyt.unyt_array(0.0, dtype=mSO.dtype, units=mSO.units)
            MstarinitSO = unyt.unyt_array(0.0, dtype=mSO.dtype, units=mSO.units)
            MBHsubSO = unyt.unyt_array(0.0, dtype=mSO.dtype, units=mSO.units)
            TgasSO = unyt.unyt_array(0.0, dtype=Tgas.dtype, units=Tgas.units)
            XraylumSO = unyt.unyt_array(0.0, dtype=Xray_lum.dtype, units=Xray_lum.units)
            XrayphlumSO = unyt.unyt_array(
                0.0, dtype=Xray_phlum.dtype, units=Xray_phlum.units
            )
            compYSO = unyt.unyt_array(0.0, dtype=compY.dtype, units=compY.units)

        # Return value should be a dict containing unyt_arrays and descriptions.
        # The dict keys will be used as HDF5 dataset names in the output.
        halo_result.update({
            f"r_{name}": (rSO, f"Radius {label}"),
            f"m_{name}": (mSO, f"Mass within a sphere {label}"),
            f"com_{name}": (comSO, f"Centre of mass within a sphere {label}"),
            f"vcom_{name}": (
                vcomSO,
                f"Centre of mass velocity within a sphere {label}",
            ),
            f"Mgas_{name}": (MgasSO, f"Total gas mass within a sphere {label}"),
            f"Mdm_{name}": (MdmSO, f"Total DM mass within a sphere {label}"),
            f"Mstar_{name}": (MstarSO, f"Total stellar mass within a sphere {label}"),
            f"MBHdyn_{name}": (
                MBHdynSO,
                f"Total dynamical BH mass within a sphere {label}",
            ),
            f"Mstarinit_{name}": (
                MstarinitSO,
                f"Total initial stellar mass with a sphere {label}",
            ),
            f"MBHsub_{name}": (
                MBHsubSO,
                f"Total sub-grid BH mass within a sphere {label}",
            ),
            f"Mhotgas_{name}": (
                MhotgasSO,
                f"Total mass of gas with T > 1e5 K within a sphere {label}",
            ),
            f"Tgas_{name}": (
                TgasSO,
                f"Mass-weighted average temperature of gas with T > 1e5 K within a sphere {label}",
            ),
            f"Xraylum_{name}": (
                XraylumSO,
                f"Total Xray luminosity within a sphere {label}",
            ),
            f"Xrayphlum_{name}": (
                XrayphlumSO,
                f"Total Xray photon luminosity within a sphere {label}",
            ),
            f"compY_{name}": (compYSO, f"Total Compton y within a sphere {label}"),
        })
