import json
from collections.abc import Callable
from typing_extensions import LiteralString, Optional, Tuple

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import wcosmo
from jaxtyping import Array, PRNGKeyArray


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

    def _transform_parameters(self, *args, **kwargs) -> Array:
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
        *args : `jax.numpy.array`
            physical parameters characterizing CBC signals
        **kwargs : `jax.numpy.array`
            physical parameters characterizing CBC signals

        Returns
        -------
        transformed_parameters : `jax.numpy.array`
            Transformed parameter space expected by trained neural network
        """
        raise NotImplementedError

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

    def check_input(
        self,
        key: PRNGKeyArray,
        shape: tuple[int, ...],
        parameter_dict: dict[LiteralString, Array],
    ) -> Tuple[PRNGKeyArray, dict[LiteralString, Array]]:
        """Method to check provided set of compact binary parameters for any
        missing information, and/or to augment provided parameters with any
        additional derived information expected by the neural network. If
        extrinsic parameters (e.g. sky location, polarization angle, etc.) have
        not been provided, they will be randomly generated and appended to the
        given CBC parameters.

        Parameters
        ----------
        key: `jax.random.PRNGKey`
            Random key to be used for generating extrinsic parameters
        shape: `tuple`
            Shape of the input array
        parameter_dict : `dict`
            Set of compact binary parameters for which we want to evaluate Pdet

        Returns
        -------
        parameter_dict : `dict`
            Dictionary of CBC parameters, augmented with necessary derived parameters
        """
        raise NotImplementedError
