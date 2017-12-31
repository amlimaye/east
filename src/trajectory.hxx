#include <array>
#include <tuple>
#include <vector>
#include <random>

template <int N, int T>
class Trajectory {
public:
    //data members
    std::array<int, N*T> data;
    std::mt19937 gen;
    int activity = 0;

    //default constructor
    Trajectory();
    //random initial state constructor
    Trajectory(uint32_t seed);
    //random initial state with specified initial density
    Trajectory(uint32_t seed, double init_rho);
    //supplied initial state constructor
    Trajectory(const std::array<int, N>& initial, uint32_t seed);
    //copy ctor
    Trajectory(const Trajectory<N, T>& rhs);

    //copy the trajectory
    Trajectory& copy();

    //indexing operator
    inline int& operator()(const int space_idx, const int time_idx);

    //equality operator
    inline bool operator==(const Trajectory<N, T>& compare_to);

    //convenience function for flipping a spin
    inline void flip(const int space_idx, const int time_idx);

    //get indices of spins that can flip at the next time
    std::vector<int> get_labile_indices(const int time_idx);

    //take a timestep
    void take_timestep(const int time_at, const bool forward=true);

    //evolve the whole trajectory
    void evolve_entire(const int start_at, const bool forward=true);

    //initiate a shooting move at timeslice shoot_from
    void shooting_move(const int shoot_from, const bool forward=true);
    void random_shooting_move();

    //initiate a shifting move at timeslice shoot_from
    void shifting_move(const int shift_from, const bool forward=true);
    void random_shifting_move();

    //do a random tps move
    void random_tps_move();

    //getters for observables
    inline int get_activity() const;
    inline double get_intensive_activity() const;
    inline int get_num_up_spins() const;
    inline double get_intensive_density() const;

    //getter for a "patch" density
    inline int get_num_in_patch(const std::tuple<int, int> lower_left_corner, 
                                const int temporal_extent,
                                const int spatial_extent);
    inline double get_density_in_patch(const std::tuple<int, int> lower_left_corner, 
                                       const int temporal_extent,
                                       const int spatial_extent);
};

#include "trajectory.txx"
