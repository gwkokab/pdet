import jax.numpy as jnp
from wcosmo.astropy import FlatLambdaCDM


__all__ = ["Planck13", "Planck15", "Planck18"]


# https://docs.astropy.org/en/stable/api/astropy.cosmology.realizations.Planck13.html
Planck13 = FlatLambdaCDM(
    H0=67.77,
    Om0=0.30721,
    Tcmb0=2.7255,
    Neff=3.046,
    m_nu=jnp.array([0.0, 0.0, 0.06]),
    Ob0=0.048252,
)

# https://docs.astropy.org/en/stable/api/astropy.cosmology.realizations.Planck15.html
Planck15 = FlatLambdaCDM(
    H0=67.74,
    Om0=0.3075,
    Tcmb0=2.7255,
    Neff=3.046,
    m_nu=jnp.array([0.0, 0.0, 0.06]),
    Ob0=0.0486,
)

# https://docs.astropy.org/en/stable/api/astropy.cosmology.realizations.Planck18.html
Planck18 = FlatLambdaCDM(
    H0=67.66,
    Om0=0.30966,
    Tcmb0=2.7255,
    Neff=3.046,
    m_nu=jnp.array([0.0, 0.0, 0.06]),
    Ob0=0.04897,
)
