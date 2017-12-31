#include <stdio.h>
#include <iostream>
#include "trajectory.hxx"

int main() {
    constexpr int T = 5;
    constexpr int N = 5;

    auto traj = Trajectory<N, T>(0);
    traj.evolve_entire(0);
    std::cout << traj << std::endl;
    std::cout << "bkwd. shifting move from index 2" << std::endl;
    traj.shifting_move(2, false);
    std::cout << traj << std::endl;

    return 0;
}
