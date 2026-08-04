# SPARK

SPARK is a physics-informed derivative Gaussian process for separable N-body Hamiltonians. It is
not a generic replacement for TERA: its advantage should be interpreted within the disclosed model
class below.

## Model

SPARK receives the particle count and spatial dimension, then uses training inputs and gradients to
identify which of the two equal state blocks has an affine gradient. It integrates that block into a
diagonal quadratic kinetic mean and infers relative particle masses from the reciprocal slopes. It
does not read the dataset's true masses, gravitational constant, softening, or test labels.

After subtracting the learned kinetic term, SPARK models the remaining potential as

```text
V(q) = sum_{i<j} w_ij u(||q_i - q_j||),    w_ij proportional to m_i m_j.
```

The shared radial law `u` is learned with a rank-128 Matérn-5/2 Nyström feature GP. Analytic radial
derivatives are pulled back through pair distances, enforcing translation and rotation invariance,
equal-and-opposite pair forces, zero net force, and zero torque.

## Scope and limitations

Pairwise additivity, particle grouping, central interactions, and a shared radial law are explicit
N-body priors. They closely match the softened-gravity generator and explain why SPARK can be both
more accurate and cheaper than a generic full-state kernel. Nonadditive many-body potentials,
velocity-dependent potentials, or ambiguous position/momentum blocks are outside the claimed model
class; ambiguous structure discovery fails closed.

The varying-`(n, d)` benchmark in `experiments/f01_spark_nd_sweep.py` tests the same frozen SPARK
configuration against TERA across particle counts `{2, 4, 10}`, spatial dimensions `{1, 2, 3}`, and
three independently generated systems per configuration.
