#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "trajectory.hpp"

namespace py = pybind11;

constexpr int N = 20;
constexpr int T = 20;

PYBIND11_MODULE(east, m) {
    py::class_<Trajectory<N, T>>(m, "Trajectory")
        //constructors
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def(py::init<uint32_t, double>())
        .def(py::init<const std::array<int, N>&, uint32_t>())
        .def(py::init<const Trajectory<N, T>&>())

        //utility functions
        .def("flip", &Trajectory<N, T>::flip)
        .def("get_labile_indices", &Trajectory<N, T>::get_labile_indices)
        .def("take_timestep", &Trajectory<N, T>::take_timestep,
             py::arg("time_at"), py::arg("forward") = true)
        .def("evolve_entire", &Trajectory<N, T>::evolve_entire,
             py::arg("start_at"), py::arg("forward") = true)
       
        //tps moves (low-level)
        .def("shooting_move", &Trajectory<N, T>::shooting_move,
             py::arg("shoot_from"), py::arg("forward") = true)
        .def("shifting_move", &Trajectory<N, T>::shifting_move,
             py::arg("shift_from"), py::arg("forward") = true)
        
        //tps moves (wrappers) 
        .def("random_shooting_move", &Trajectory<N, T>::random_shooting_move)
        .def("random_shifting_move", &Trajectory<N, T>::random_shifting_move)
        .def("random_tps_move", &Trajectory<N, T>::random_tps_move)

        //patch density getters
        .def("patch_num", &Trajectory<N, T>::get_num_in_patch)
        .def("patch_density", &Trajectory<N, T>::get_density_in_patch)

        //equality comparison
        .def("__eq__", &Trajectory<N, T>::operator==)

        //pretty-printing
        .def("__repr__", [](const Trajectory<N, T>& t){
                std::stringstream rep;
                rep << t;
                return rep.str();
            })

        //pickling support
        .def(py::pickle(

             //note: pickling loses the RNG state!
             [](const Trajectory<N, T>& self) {
                return py::make_tuple(self.data);
             },

             //reconstruction _only_ preserves the data state, not RNG state!
             [](py::tuple tup) {
                auto traj_ptr = new Trajectory<N, T>();
                auto data = tup[0].cast<std::array<int, N*T>>();
                for (int k = 0; k < N*T; k++) {
                    traj_ptr->data[k] = data[k];
                }
                return traj_ptr;
             })
            )

        //access density/activity like struct parameters
        .def_property_readonly("density", &Trajectory<N, T>::get_intensive_density)
        .def_property_readonly("actvity", &Trajectory<N, T>::get_intensive_activity);
}
