import h5py
import numpy as np
from swiftsimio import load
from numba import jit
from unyt import g, cm, mp, erg, s
from tqdm import tqdm

class XrayCalculator:
    def init(self, cellgrid):
        self.z_now = cellgrid.cosmology["Redshift"]

    def load_table(self, table_name, band, observing_type):
        self.table = h5py.File(table_name, 'r')
        self.X_Ray = self.table[band][observing_type][()].astype(np.float32)
        self.He_bins = self.table['/Bins/He_bins'][()].astype(np.float32)
        self.missing_elements = self.table['/Bins/Missing_element'][()]
        self.element_masses = self.table['Bins/Element_masses'][()].astype(np.float32)

        self.density_bins = self.table['/Bins/Density_bins/'][()].astype(np.float32)
        self.temperature_bins = self.table['/Bins/Temperature_bins/'][()].astype(np.float32)
        self.redshift_bins = self.table['/Bins/Redshift_bins'][()].astype(np.float32)
        self.dn = 0.2
        self.dT = 0.1
        self.dz = 0.2

        self.log10_solar_metallicity = self.table['/Bins/Solar_metallicities/'][()].astype(np.float32)
        self.solar_metallicity = np.power(10, self.log10_solar_metallicity)

    @staticmethod
    @jit(nopython = True)
    def get_index_1d(bins, subdata):
        eps = 1e-4
        delta = (len(bins) - 1) / (bins[-1] - bins[0])

        idx = np.zeros_like(subdata)
        dx = np.zeros_like(subdata, dtype = np.float32)
        for i, x in enumerate(subdata):
            if x < bins[0] + eps:
                # We are below the first element
                idx[i] = 0
                dx[i] = 0
            elif x < bins[-1] - eps:
                # Normal case
                idx[i] = int((x - bins[0]) * delta)
                dx[i] = (x - bins[int(idx[i])]) * delta
            else:
                # We are after the last element
                idx[i] = len(bins) - 2
                dx[i] = 1
            
        return idx, dx

    @staticmethod
    @jit(nopython = True)
    def get_index_1d_irregular(bins, subdata):
        eps = 1e-6
        idx = np.zeros_like(subdata)
        dx = np.zeros_like(subdata, dtype = np.float32)

        for i, x in enumerate(subdata):
            if x < bins[0] + eps:
                idx[i] = 0
                dx[i] = 0
            elif x < bins[-1] - eps:
                min_idx = -1

                '''
                Do this the hard way: Search the table
                for the smallest index i in bins[i] such
                that table[i] < x
                '''
                for j in range(len(bins)):
                    if x - bins[j] <= 0:
                        # Found the first entry that is larger than x, go back by 1
                        min_idx = j - 1
                        break

                idx[i] = min_idx
                dx[i] = (x - bins[min_idx]) / (bins[min_idx + 1] - bins[min_idx])
            else:
                idx[i] = len(bins) - 2
                dx[i] = 1

        return idx, dx

    @staticmethod
    @jit(nopython = True)
    def get_table_interp(dn, dT, dx_T, dx_n, idx_T, idx_n, idx_he, dx_he, idx_z, dx_z, X_Ray, abundance_to_solar):
        f_n_T_Z = np.zeros_like(dx_n)

    
        # for i in tqdm(range(len(idx_n))):
        for i in range(len(idx_n)):
            t_z = 1 - dx_z[i]
            d_z = dx_z[i]

            # Compute temperature offset relative to bin
            t_T = 1 - dx_T[i]
            d_T = dx_T[i]

            # Compute density offset relative to bin
            t_nH = 1 - dx_n[i]
            d_nH = dx_n[i]

            # Compute Helium offset relative to bin
            t_He = 1 - dx_he[i]
            d_He = dx_he[i]

            # Do the actual 4D linear interpolation
            f_n_T = np.zeros(X_Ray.shape[2], dtype = np.float32)

            f_n_T += t_nH * t_He * t_T * t_z * X_Ray[idx_z[i], idx_he[i], :, idx_T[i], idx_n[i]]
            f_n_T += t_nH * t_He * d_T * t_z * X_Ray[idx_z[i], idx_he[i], :, idx_T[i] + 1, idx_n[i]]
            f_n_T += t_nH * d_He * t_T * t_z * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i], idx_n[i]]
            f_n_T += d_nH * t_He * t_T * t_z * X_Ray[idx_z[i], idx_he[i], :, idx_T[i], idx_n[i] + 1]

            f_n_T += t_nH * d_He * d_T * t_z * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i]]
            f_n_T += d_nH * t_He * d_T * t_z * X_Ray[idx_z[i], idx_he[i], :, idx_T[i] + 1, idx_n[i] + 1]
            f_n_T += d_nH * d_He * t_T * t_z * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i], idx_n[i] + 1]
            f_n_T += d_nH * d_He * d_T * t_z * X_Ray[idx_z[i], idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i] + 1]


            f_n_T += t_nH * t_He * t_T * d_z * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i], idx_n[i]]
            f_n_T += t_nH * t_He * d_T * d_z * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i] + 1, idx_n[i]]
            f_n_T += t_nH * d_He * t_T * d_z * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i], idx_n[i]]
            f_n_T += d_nH * t_He * t_T * d_z * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i], idx_n[i] + 1]

            f_n_T += t_nH * d_He * d_T * d_z * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i]]
            f_n_T += d_nH * t_He * d_T * d_z * X_Ray[idx_z[i] + 1, idx_he[i], :, idx_T[i] + 1, idx_n[i] + 1]
            f_n_T += d_nH * d_He * t_T * d_z * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i], idx_n[i] + 1]
            f_n_T += d_nH * d_He * d_T * d_z * X_Ray[idx_z[i] + 1, idx_he[i] + 1, :, idx_T[i] + 1, idx_n[i] + 1]


            # Add each metal contribution individually
            f_n_T_Z_temp = np.power(10, f_n_T[-1])
            for j in range(len(f_n_T) - 1):
                f_n_T_Z_temp += np.power(10, f_n_T[j]) * abundance_to_solar[i, j]

            f_n_T_Z[i] = np.log10(f_n_T_Z_temp)

        return f_n_T_Z

    def interpolate_X_Ray(self, table_name, densities, temperatures, element_mass_fractions, masses, bands = None, observing_types = None, unit_conversions = None, fill_value = None):
        redshift = self.z_now
        scale_factor = 1 / (1 + redshift)
        data_n = np.log10(element_mass_fractions.hydrogen * (1 / scale_factor**3) * densities.to(g * cm**-3) / mp)
        data_T = np.log10(temperatures)
        volumes = (masses / ((1 / scale_factor**3) * densities)).to(cm**3)


        if bands == None:
            print('Please specify the band you would like to generate emissivities for\n \
                Using the "band = " keyword\n\n \
                Available options are:\n \
                "erosita-low" (0.2-2.3 keV)\n \
                "erosita-high" (2.3-8.0 keV)\n \
                "ROSAT" (0.5-2.0 keV)')
            raise KeyError
        
        if observing_types == None:
            print('Please specify whether you would like to generate photon or energie emissivities\n \
                Using the "observing_type = " keyword\n\n \
                Available options are:\n \
                "energies"\n \
                "photons"\n \
                "energies_intrinsic"\n \
                "photons_intrinsic"')
            raise KeyError       

        if (bands != None) & (observing_types != None) & (unit_conversions != None):
            assert (len(bands) == len(observing_types)) & (len(bands) == len(unit_conversions))

        place_holder_band = bands[0]
        place_holder_observing_type = observing_types[0]

        # Initialise interpolation class
        self.load_table(table_name, place_holder_band, place_holder_observing_type)

        # Initialise the emissivity array which will be returned
        emissivities = np.zeros((data_n.shape[0], len(bands)), dtype = float)
        luminosities = np.zeros_like(emissivities)

        # Create density mask, round to avoid numerical errors
        density_mask = (data_n >= np.round(self.density_bins.min(), 1)) & (data_n <= np.round(self.density_bins.max(), 1))
        # Create temperature mask, round to avoid numerical errors
        temperature_mask = (data_T >= np.round(self.temperature_bins.min(), 1)) & (data_T <= np.round(self.temperature_bins.max(), 1))

        # Combine masks
        joint_mask = density_mask & temperature_mask


        # Check if within density and temperature bounds
        density_bounds = np.sum(density_mask) == density_mask.shape[0]
        temperature_bounds = np.sum(temperature_mask) == temperature_mask.shape[0]
        if ~(density_bounds & temperature_bounds):
            #If no fill_value is set, return an error with some explanation
            if fill_value == None:
                raise ValueError("Temperature or density are outside of the interpolation range and no fill_value is supplied\n \
                                Temperature ranges between log(T) = 5 and log(T) = 9.5\n \
                                Density ranges between log(nH) = -8 and log(nH) = 6\n \
                                Set the kwarg 'fill_value = some value' to set all particles outside of the interpolation range to 'some value'\n \
                                Or limit your particle data set to be within the interpolation range")
            else:
                emissivities[~joint_mask] = fill_value
        
        # If only a single redshift is received, use it for all particles
        if redshift.size == 1:
            redshift = np.ones_like(data_n) * redshift

        mass_fraction = np.zeros((len(data_n[joint_mask]), 9))

        #get individual mass fraction
        mass_fraction[:, 0] = element_mass_fractions.hydrogen[joint_mask]
        mass_fraction[:, 1] = element_mass_fractions.helium[joint_mask]
        mass_fraction[:, 2] = element_mass_fractions.carbon[joint_mask]
        mass_fraction[:, 3] = element_mass_fractions.nitrogen[joint_mask]
        mass_fraction[:, 4] = element_mass_fractions.oxygen[joint_mask]
        mass_fraction[:, 5] = element_mass_fractions.neon[joint_mask]
        mass_fraction[:, 6] = element_mass_fractions.magnesium[joint_mask]
        mass_fraction[:, 7] = element_mass_fractions.silicon[joint_mask]
        mass_fraction[:, 8] = element_mass_fractions.iron[joint_mask]

        # Find density offsets
        idx_n, dx_n = self.get_index_1d(self.density_bins, data_n[joint_mask])

        # Find temperature offsets
        idx_T, dx_T = self.get_index_1d(self.temperature_bins, data_T[joint_mask])

        # Calculate the abundance wrt to solar
        abundances = (mass_fraction / np.expand_dims(mass_fraction[:, 0], axis = 1)) * (self.element_masses[0] /  np.array(self.element_masses))

        # Calculate abundance offsets using solar abundances
        abundance_to_solar = abundances / self.solar_metallicity

        # Add columns for Calcium and Sulphur and move Iron to the end
        abundance_to_solar = np.c_[abundance_to_solar[:, :-1], abundance_to_solar[:, -2], abundance_to_solar[:, -2], abundance_to_solar[:, -1]] 

        #Find helium offsets
        idx_he, dx_he = self.get_index_1d_irregular(self.He_bins, np.log10(abundances[:, 1]))

        # Find redshift offsets
        idx_z, dx_z = self.get_index_1d(self.redshift_bins, redshift[joint_mask])

        # Interpolate the table for each specified band
        for i_interp, band, observing_type in zip(range(len(bands)), bands, observing_types):
            self.load_table(table_name, band, observing_type)
            emissivities[joint_mask, i_interp] = self.get_table_interp(self.dn, self.dT, dx_T, dx_n, idx_T.astype(int), idx_n.astype(int), idx_he.astype(int), dx_he, idx_z.astype(int), dx_z, self.X_Ray, abundance_to_solar[:, 2:])
            
            # Convert from erg cm^3 s^-1 to erg cm^-3 s^-1
            # To do so we multiply by nH^2, this is the actual nH not the nearest bin
            # It allows to extrapolate in density space without too much worry
            # log(emissivity * nH^2) = log(emissivity) + 2*log(nH)
            emissivities[joint_mask, i_interp] += 2*data_n[joint_mask]

            luminosities[joint_mask, i_interp] = np.power(10, emissivities[joint_mask, i_interp]) * volumes[joint_mask] * unit_conversions[i_interp]
            

        return luminosities
