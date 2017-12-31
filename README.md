# east
A small personal project to reproduce some of the results in this paper (https://arxiv.org/abs/1710.04747) on the kinetically-constrained East model for glass formation.

Includes a Trajectory class (written in C++) representing a trajectory in the East model. This class includes functions for performing transition path sampling moves on the trajectory object. This class is exported to Python via pybind11, and all analysis is done Python-side.
