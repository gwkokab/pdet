import os
from typing import Optional
from typing_extensions import List

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import Planck15, z_at_value
from jaxtyping import Array, PRNGKeyArray

from ._emulator import Emulator
from ._names import (
    A_1,
    A_2,
    COS_INCLINATION,
    COS_THETA_1,
    COS_THETA_2,
    MASS_1,
    MASS_2,
    PHI_12,
    POLARIZATION_ANGLE,
    REDSHIFT,
    RIGHT_ASCENSION,
    SIN_DECLINATION,
)
from ._transform import eta_from_q, mass_ratio
import time


class pdet_O3(Emulator):
    """Class implementing the LIGO-Hanford, LIGO-Livingston, and Virgo
    network's selection function during their O3 observing run.

    Used to evaluate the detection probability of compact binaries,
    assuming a false alarm threshold of below 1 per year. The computed
    detection probabilities include all variation in the detectors'
    sensitivities over the course of the O3 run and accounts for time in
    which the instruments were not in observing mode. They should
    therefore be interpreted as the probability of a CBC detection if
    that CBC occurred during a random time between the startdate and
    enddate of O3.
    """

    def __init__(
        self, model_weights=None, scaler=None, parameters: Optional[List[str]] = None
    ):
        """Instantiates a `p_det_O3` object, subclassed from the `emulator`
        class.

        Parameters
        ----------
        model_weights : `None` or `str`
            Filepath to .hdf5 file containing trained network weights, as saved
            by a `tensorflow.keras.Model.save_weights`, command, if one wishes
            to override the provided default weights (which are loaded when
            `model_weights==None`).
        scaler : `str`
            Filepath to saved `sklearn.preprocessing.StandardScaler` object, if
            one wishes to override the provided default (loaded when
            `scaler==None`).
        """

        current_time = time.time()

        print(time.time() - current_time, " init started.")

        current_time = time.time()

        if parameters is None:
            raise ValueError("Must provide list of parameters")

        self.parameters = parameters

        if model_weights is None:
            model_weights = os.path.join(
                os.path.dirname(__file__), "./../trained_weights/weights_HLV_O3.hdf5"
            )
        else:
            print("Overriding default weights")

        print(time.time() - current_time, " weights reading completed.")

        current_time = time.time()

        if scaler is None:
            scaler = os.path.join(
                os.path.dirname(__file__), "./../trained_weights/scaler_HLV_O3.json"
            )
        else:
            print("Overriding default weights")

        print(time.time() - current_time, " scalar reading completed.")
        current_time = time.time()

        input_dimension = 15
        hidden_width = 192
        hidden_depth = 4
        activation = lambda x: jax.nn.leaky_relu(x, 1e-3)
        final_activation = lambda x: (1.0 - 0.0589) * jax.nn.sigmoid(x)

        self.interp_DL = np.logspace(-4, jnp.log10(15.0), 500)
        self.interp_z = z_at_value(
            Planck15.luminosity_distance, self.interp_DL * u.Gpc
        ).value

        print(time.time() - current_time, " before starting super init.")
        current_time = time.time()

        super().__init__(
            model_weights,
            scaler,
            input_dimension,
            hidden_width,
            hidden_depth,
            activation,
            final_activation,
        )

        print(time.time() - current_time, " super init completed.")

    def _transform_parameters(
        self,
        m1_trials: Array,
        m2_trials: Array,
        a1_trials: Array,
        a2_trials: Array,
        cost1_trials: Array,
        cost2_trials: Array,
        z_trials: Array,
        cos_inclination_trials: Array,
        pol_trials: Array,
        phi12_trials: Array,
        ra_trials: Array,
        sin_dec_trials: Array,
    ) -> Array:
        q = mass_ratio(m1=m1_trials, m2=m2_trials)
        eta = eta_from_q(q=q)
        Mtot_det = (m1_trials + m2_trials) * (1.0 + z_trials)
        Mc_det = (eta**0.6) * Mtot_det

        DL = jnp.interp(z_trials, self.interp_z, self.interp_DL)
        log_Mc_DL_ratio = (5.0 / 6.0) * jnp.log(Mc_det) - jnp.log(DL)
        amp_factor_plus = 2.0 * (
            log_Mc_DL_ratio + jnp.log((1.0 + cos_inclination_trials**2)) + jnp.log(0.5)
        )
        amp_factor_cross = 2.0 * (log_Mc_DL_ratio + jnp.log(cos_inclination_trials))

        # Effective spins
        chi_effective = (a1_trials * cost1_trials + q * a2_trials * cost2_trials) / (
            1.0 + q
        )
        chi_diff = (a1_trials * cost1_trials - a2_trials * cost2_trials) * 0.5

        # Generalized precessing spin
        Omg = q * (3.0 + 4.0 * q) / (4.0 + 3.0 * q)
        chi_1p = a1_trials * jnp.sqrt(1.0 - cost1_trials**2)
        chi_2p = a2_trials * jnp.sqrt(1.0 - cost2_trials**2)
        chi_p_gen = jnp.sqrt(
            chi_1p**2
            + (Omg * chi_2p) ** 2
            + 2.0 * Omg * chi_1p * chi_2p * jnp.cos(phi12_trials)
        )

        return jnp.array(
            [
                amp_factor_plus,
                amp_factor_cross,
                Mc_det,
                Mtot_det,
                eta,
                q,
                DL,
                ra_trials,
                sin_dec_trials,
                jnp.abs(cos_inclination_trials),
                jnp.sin(pol_trials % np.pi),
                jnp.cos(pol_trials % np.pi),
                chi_effective,
                chi_diff,
                chi_p_gen,
            ]
        )

    def predict(self, key: PRNGKeyArray, params: Array) -> Array:
        # Copy so that we can safely modify dictionary in-place
        parameter_dict = {
            parameter: params[i] for i, parameter in enumerate(self.parameters)
        }

        # Check input
        parameter_dict = self.check_input(key, parameter_dict)

        features = jnp.stack(
            [
                parameter_dict[MASS_1],
                parameter_dict[MASS_2],
                parameter_dict[A_1],
                parameter_dict[A_2],
                parameter_dict[COS_THETA_1],
                parameter_dict[COS_THETA_2],
                parameter_dict[REDSHIFT],
                parameter_dict[COS_INCLINATION],
                parameter_dict[POLARIZATION_ANGLE],
                parameter_dict[PHI_12],
                parameter_dict[RIGHT_ASCENSION],
                parameter_dict[SIN_DECLINATION],
            ],
            axis=-1,
        )

        return self.__call__(features)
