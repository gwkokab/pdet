import json
import warnings
from collections.abc import Callable
from typing_extensions import LiteralString, Optional

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jrd
import wcosmo
from astropy import units
from jaxtyping import Array, PRNGKeyArray
from unxt import ustrip
from wcosmo import z_at_value

from ._names import (
    A_1,
    A_2,
    COMOVING_DISTANCE,
    COS_INCLINATION,
    COS_THETA_1,
    COS_THETA_2,
    INCLINATION,
    LUMINOSITY_DISTANCE,
    MASS_1,
    MASS_2,
    PHI_12,
    POLARIZATION_ANGLE,
    REDSHIFT,
    RIGHT_ASCENSION,
    SIN_DECLINATION,
)


Planck15: wcosmo.astropy.FlatLambdaCDM = getattr(wcosmo.astropy, "Planck15")


jax.config.update("jax_enable_x64", True)


class Emulator:
    """Base class implementing a generic detection probability emulator.

    Intended to be subclassed when constructing emulators for particular
    networks/observing runs.
    """

    def __init__(
        self,
        trained_weights: str,
        scaler: str,
        input_size: int,
        hidden_layer_width: int,
        hidden_layer_depth: int,
        activation: Callable,
        final_activation: Callable,
    ):
        """Instantiate an `emulator` object.

        Parameters
        ----------
        trained_weights : `str`
            Filepath to .hdf5 file containing trained network weights, as
            saved by a `tensorflow.keras.Model.save_weights` command
        scaler : `str`
            Filepath to saved `sklearn.preprocessing.StandardScaler` object,
            fitted during network training
        input_size : `int`
            Dimensionality of input feature vector
        hidden_layer_width : `int`
            Width of hidden layers
        hidden_layer_depth : `int`
            Number of hidden layers
        activation : `func`
            Activation function to be applied to hidden layers

        Returns
        -------
        None
        """

        # Instantiate neural network
        self.trained_weights = trained_weights
        self.nn = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            depth=hidden_layer_depth,
            width_size=hidden_layer_width,
            activation=activation,
            final_activation=final_activation,
            key=jax.random.PRNGKey(111),
        )

        # Load trained weights and biases
        weight_data = h5py.File(self.trained_weights, "r")

        # Load scaling parameters
        with open(scaler, "r") as f:
            self.scaler = json.load(f)
            self.scaler["mean"] = jnp.array(self.scaler["mean"])
            self.scaler["scale"] = jnp.array(self.scaler["scale"])

        # Define helper functions with which to access MLP weights and biases
        # Needed by `eqx.tree_at`
        def get_weights(i: int) -> Callable[[eqx.nn.MLP], Array]:
            return lambda t: t.layers[i].weight

        def get_biases(i: int) -> Callable[[eqx.nn.MLP], Optional[Array]]:
            return lambda t: t.layers[i].bias

        # Loop across layers, load pre-trained weights and biases
        for i in range(hidden_layer_depth + 1):
            if i == 0:
                key = "dense"
            else:
                key = "dense_{0}".format(i)

            layer_weights = weight_data["{0}/{0}/kernel:0".format(key)][()].T
            self.nn = eqx.tree_at(get_weights(i), self.nn, layer_weights)

            layer_biases = weight_data["{0}/{0}/bias:0".format(key)][()].T
            self.nn = eqx.tree_at(get_biases(i), self.nn, layer_biases)

        self.nn_vmapped = jax.vmap(self.nn)

    def _transform_parameters(self, *physical_params):
        """OVERWRITE UPON SUBCLASSING.

        Function to convert from a predetermined set of user-provided physical
        CBC parameters to the input space expected by the trained neural
        network. Used by `emulator.__call__` below.

        NOTE: This function should be JIT-able and differentiable, and so
        consistency/completeness checks should be performed upstream; we
        should be able to assume that `physical_params` is provided as
        expected.

        Parameters
        ----------
        physical_params : numpy.array or jax.numpy.array
            Array containing physical parameters characterizing CBC signals

        Returns
        -------
        transformed_parameters : jax.numpy.array
            Transformed parameter space expected by trained neural network
        """

        # APPLY REQUIRED TRANSFORMATION HERE
        # transformed_params = ...

        # Dummy transformation
        transformed_params = physical_params

        # Jaxify
        transformed_params = jnp.array(physical_params)

        return transformed_params

    def __call__(self, x):
        """Function to evaluate the trained neural network on a set of user-
        provided physical CBC parameters.

        NOTE: This function should be JIT-able and differentiable, and so any
        consistency or completeness checks should be performed upstream, such
        that we can assume the provided parameter vector `x` is already in the
        correct format expected by the `emulator._transform_parameters` method.
        """

        # Transform physical parameters to space expected by the neural network
        # transformed_x = self._transform_parameters(*x)
        transformed_x = self._transform_parameters(
            *[x[..., i] for i in range(x.shape[-1])]
        )

        # Apply scaling, evaluate the network, and return
        scaled_x = (transformed_x - self.scaler["mean"]) / self.scaler["scale"]
        # return jax.vmap(self.nn)(scaled_x)
        return self.nn_vmapped(scaled_x)

    def _check_distance(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        parameter_dict: dict[LiteralString, Array],
    ) -> tuple[PRNGKeyArray, dict[LiteralString, Array]]:
        """Helper function to check the presence of required distance
        arguments, and augment input parameters with additional quantities as
        needed.

        Parameters
        ----------
        parameter_dict : `dict` or `pd.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        None
        """

        # Check for distance parameters
        # If none are present, or if more than one is present, return an error
        allowed_distance_params = [LUMINOSITY_DISTANCE, COMOVING_DISTANCE, REDSHIFT]
        if not any(param in parameter_dict for param in allowed_distance_params):
            raise RuntimeError(
                "Missing distance parameter. Requires one of: ", allowed_distance_params
            )
        elif all(param in parameter_dict for param in allowed_distance_params):
            raise RuntimeError(
                "Multiple distance parameters present. Only one of the following allowed: ",
                allowed_distance_params,
            )

        missing_params = {}

        # Augment, such both redshift and luminosity distance are present
        if COMOVING_DISTANCE in parameter_dict:
            redshift = z_at_value(
                lambda z: ustrip(units.Gpc, Planck15.comoving_distance(z)),
                parameter_dict[COMOVING_DISTANCE],
            )
            luminosity_distance = Planck15.luminosity_distance(redshift).to_value(
                units.Gpc
            )

            redshift = jnp.broadcast_to(redshift, shape)
            luminosity_distance = jnp.broadcast_to(luminosity_distance, shape)

            missing_params[REDSHIFT] = redshift
            parameter_dict[LUMINOSITY_DISTANCE] = luminosity_distance

        elif LUMINOSITY_DISTANCE in parameter_dict:
            redshift = z_at_value(
                lambda z: ustrip(units.Gpc, Planck15.luminosity_distance(z)),
                parameter_dict[LUMINOSITY_DISTANCE],
            )
            redshift = jnp.broadcast_to(redshift, shape)
            missing_params[REDSHIFT] = redshift

        elif REDSHIFT in parameter_dict:
            luminosity_distance = Planck15.luminosity_distance(
                parameter_dict[REDSHIFT]
            ).to_value(units.Gpc)
            luminosity_distance = jnp.broadcast_to(luminosity_distance, shape)
            missing_params[LUMINOSITY_DISTANCE] = luminosity_distance

        return key, missing_params

    def _check_masses(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        parameter_dict: dict[LiteralString, Array],
    ) -> tuple[PRNGKeyArray, dict[LiteralString, Array]]:
        """Helper function to check the presence of required mass arguments,
        and augment input parameters with additional quantities needed for
        prediction.

        Parameters
        ----------
        parameter_dict : `dict` or `pd.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        None
        """

        # Check that mass parameters are present
        required_mass_params = [MASS_1, MASS_2]
        for param in required_mass_params:
            if param not in parameter_dict:
                raise RuntimeError("Must include {0} parameter".format(param))

        return key, {MASS_1: parameter_dict[MASS_1], MASS_2: parameter_dict[MASS_2]}

    def _check_spins(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        parameter_dict: dict[LiteralString, Array],
    ) -> tuple[PRNGKeyArray, dict[LiteralString, Array]]:
        """Helper function to check for the presence of required spin
        parameters and augment with additional quantities as needed.

        Parameters
        ----------
        parameter_dict : `dict` or `pd.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        None
        """

        missing_params = {}

        if A_1 not in parameter_dict:
            warnings.warn(
                f"Parameter {A_1} not present. Filling with random value from isotropic distribution."
            )
            missing_params[A_1] = jnp.zeros(shape)

        if A_2 not in parameter_dict:
            warnings.warn(
                f"Parameter {A_2} not present. Filling with random value from isotropic distribution."
            )
            missing_params[A_2] = jnp.zeros(shape)

        # Check for optional parameters, fill in if absent
        if COS_THETA_1 not in parameter_dict:
            warnings.warn(
                f"Parameter {COS_THETA_1} not present. Filling with random value from isotropic distribution."
            )
            missing_params[COS_THETA_1] = jnp.ones(shape)

        if COS_THETA_2 not in parameter_dict:
            warnings.warn(
                f"Parameter {COS_THETA_2} not present. Filling with random value from isotropic distribution."
            )
            missing_params[COS_THETA_2] = jnp.ones(shape)

        if PHI_12 not in parameter_dict:
            warnings.warn(
                f"Parameter {PHI_12} not present. Filling with random value from isotropic distribution."
            )
            missing_params[PHI_12] = jnp.zeros(shape)

        return key, missing_params

    def _check_extrinsic(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        parameter_dict: dict[LiteralString, Array],
    ) -> tuple[PRNGKeyArray, dict[LiteralString, Array]]:
        """Helper method to check required extrinsic parameters and augment as
        necessary.

        Parameters
        ----------
        parameter_dict : `dict` or `pd.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        None
        """

        missing_params = {}

        if RIGHT_ASCENSION not in parameter_dict:
            warnings.warn(
                f"Parameter {RIGHT_ASCENSION} not present. Filling with random value from isotropic distribution."
            )
            missing_params[RIGHT_ASCENSION] = jrd.uniform(
                key, shape, minval=0.0, maxval=2.0 * jnp.pi
            )
            _, key = jrd.split(key)
        if SIN_DECLINATION not in parameter_dict:
            warnings.warn(
                f"Parameter {SIN_DECLINATION} not present. Filling with random value from isotropic distribution."
            )
            missing_params[SIN_DECLINATION] = jrd.uniform(
                key, shape, minval=-1.0, maxval=1.0
            )
            _, key = jrd.split(key)

        if INCLINATION not in parameter_dict:
            if COS_INCLINATION not in parameter_dict:
                warnings.warn(
                    f"Parameter {INCLINATION} or {COS_INCLINATION} not present. Filling with random value from isotropic distribution."
                )
                missing_params[COS_INCLINATION] = jrd.uniform(
                    key, shape, minval=-1.0, maxval=1.0
                )
            _, key = jrd.split(key)
        else:
            missing_params[COS_INCLINATION] = jnp.cos(parameter_dict[INCLINATION])

        if POLARIZATION_ANGLE not in parameter_dict:
            warnings.warn(
                f"Parameter {POLARIZATION_ANGLE} not present. Filling with random value from isotropic distribution."
            )
            missing_params[POLARIZATION_ANGLE] = jrd.uniform(
                key, shape, minval=0.0, maxval=2.0 * jnp.pi
            )
            _, key = jrd.split(key)

        return key, missing_params

    def check_input(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        parameter_dict: dict[LiteralString, Array],
    ) -> dict[LiteralString, Array]:
        """Method to check provided set of compact binary parameters for any
        missing information, and/or to augment provided parameters with any
        additional derived information expected by the neural network. If
        extrinsic parameters (e.g. sky location, polarization angle, etc.) have
        not been provided, they will be randomly generated and appended to the
        given CBC parameters.

        Parameters
        ----------
        parameter_dict : `dict` or `pd.DataFrame`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        parameter_dict : `dict`
            Dictionary of CBC parameters, augmented with necessary derived
            parameters
        """

        # Check parameters
        key, missing_distance = self._check_distance(key, shape, parameter_dict)
        key, missing_masses = self._check_masses(key, shape, parameter_dict)
        key, missing_spins = self._check_spins(key, shape, parameter_dict)
        key, missing_extrinsic = self._check_extrinsic(key, shape, parameter_dict)

        parameter_dict.update(missing_distance)
        parameter_dict.update(missing_masses)
        parameter_dict.update(missing_spins)
        parameter_dict.update(missing_extrinsic)

        return parameter_dict
