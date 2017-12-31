template<int N, int T>
Trajectory<N, T>::Trajectory() {
    for (int k = 0; k < N*T; k++) {
        data[k] = 0;
    }
}

template <int N, int T>
Trajectory<N, T>::Trajectory(uint32_t seed) {
    //seed the generator
    gen = std::mt19937(seed);

    //initialize non-first timeslices to zero
    for (int k = N; k < N*T; k++) {
        data[k] = 0;
    }

    //tracks current index (from 0 -> N-1)
    int curr_idx = 0;

    //number of 32-bit random numbers we need to generate
    auto nblocks = (int) std::ceil(N / 32.0);
    for (int block_idx = 0; block_idx < nblocks; block_idx++) {
        //generate a fresh 32-bit random number
        uint32_t rand_num = gen();

        //loop through bits of the random number
        for (int i = 0; i < 32; i++) {
            //if we filled the first timeslice then nothing left to do
            if (curr_idx == N) {return;}

            //initialize this spin to the value of the bit
            data[curr_idx++] = (int) ((rand_num >> i) & 1) == 1;
        }
    }
};

template <int N, int T>
Trajectory<N, T>::Trajectory(uint32_t seed, double init_rho) {
    //seed the generator
    gen = std::mt19937(seed);

    //initialize non-first timeslices to zero
    for (int k = N; k < N*T; k++) {
        data[k] = 0;
    }

    //uniform random number within [0.0, 1.0)
    auto dist = std::uniform_real_distribution<>(0.0, 1.0);

    //for each number, if the random number is greater than the density
    //this should on average produce a starting density of rho
    for (int k = 0; k < N; k++) {
        data[k] = (dist(gen) > init_rho) ? 0 : 1;
    }
};

template <int N, int T>
Trajectory<N, T>::Trajectory(const std::array<int, N>& initial, uint32_t seed) {
    //seed the generator
    gen = std::mt19937(seed);

    //copy supplied initial array over to the initial timeslice
    for (int i = 0; i < N; i++) {
        data[i] = initial[i];
    }

    //zero out all other timeslices
    for (int k = N; k < N*T; k++) {
        data[k] = 0;
    }
};

template <int N, int T>
Trajectory<N, T>::Trajectory(const Trajectory<N, T>& rhs) {
    gen = rhs.gen;
    data = rhs.data;
};

template <int N, int T>
inline int& Trajectory<N, T>::operator()(const int space_idx, const int time_idx) {
    return data[time_idx*N + space_idx];
};

template <int N, int T>
inline bool Trajectory<N, T>::operator==(const Trajectory<N, T>& compare_to) {
    for (int i = 0; i < N*T; i++) {
        if (this->data[i] != compare_to.data[i]) {
            return false;
        }
    }
    return true;
}

//convenience function to flip a spin
template <int N, int T>
inline void Trajectory<N, T>::flip(const int space_idx, const int time_idx) {
    if (data[time_idx*N + space_idx] == 1) {
        data[time_idx*N + space_idx] = 0; 
    } else {
        data[time_idx*N + space_idx] = 1; 
    }
}

//find the indices of spins that are "allowed" to flip
template <int N, int T>
std::vector<int> Trajectory<N, T>::get_labile_indices(const int time_idx) {
    std::vector<int> labile_indices = {};
    for (int i = 0; i < N-1; i++) {
        if (this->operator()(i, time_idx) == 1) {
            labile_indices.push_back(i+1);
        }
    }
    return labile_indices;
}

template <int N, int T>
void Trajectory<N, T>::take_timestep(const int time_at, const bool forward) {
    if (forward) {
        //copy the old state over to the next timeslice
        for (int i = 0; i < N; i++) {
            this->operator()(i, time_at+1) = this->operator()(i, time_at);
        }

        //get the labile indices (could be optimized)
        auto labile_indices = this->get_labile_indices(time_at);
        auto sz = labile_indices.size();

        //only flip if spins are labile at all
        if (sz != 0) {
            //pick the spin to flip, then flip it
            std::uniform_int_distribution<> dist(0, labile_indices.size()-1);
            auto rand_idx = dist(gen);
            auto flip_me = labile_indices[rand_idx];
            this->flip(flip_me, time_at+1);

            //increase model's activity by one, since we flipped a spin
            this->activity++;
        }
    } else {
        //copy the state at time_at to the _previous_ timeslice
        for (int i = 0; i < N; i++) {
            this->operator()(i, time_at-1) = this->operator()(i, time_at);
        }

        //get the labile indices
        auto labile_indices = this->get_labile_indices(time_at);
        auto sz = labile_indices.size();

        if (sz != 0) {
            //pick the spin to flip, then flip it
            std::uniform_int_distribution<> dist(0, labile_indices.size()-1);
            auto rand_idx = dist(gen);
            auto flip_me = labile_indices[rand_idx];
            this->flip(flip_me, time_at-1);

            //increase model's activity by one, since we flipped a spin
            this->activity++;
        }
    }
}

template<int N, int T>
void Trajectory<N, T>::evolve_entire(const int start_at, const bool forward) {
    for (int t = start_at; t < (T - start_at); t++) {
        this->take_timestep(t, forward);
    }
}

template<int N, int T>
void Trajectory<N, T>::shifting_move(const int shift_from, const bool forward) {
    //"forward" shifting move means: terminate the trajectory at shift_from
    //re-evolve the timesteps backwards from T-shift_from -> 0
    if (forward) {
        //shift the trajectory
        for (int t = shift_from; t >= 0; t--) {
            auto shift_to = t + (T - shift_from) - 1;
            for (int i = 0; i < N; i++) {
                this->operator()(i, shift_to) = this->operator()(i, t);
            }
        }
        //zero-out, then re-evolve the trajectory backwards from (T - shift_from - 1)
        for (int t = 0; t < (T - shift_from - 1); t++) {
            for (int i = 0; i < N; i++) {
                this->operator()(i, t) = 0;
            }
        }
        for (int t = (T - shift_from); t > 0; t--) {
            this->take_timestep(t, false);
        }
    } else {
    //"backward" shifting move means: restart the trajectory at shift_from
    //re-evolve the timesteps forwards from (T - shift_from) -> T
        //shift the trajectory
        for (int t = shift_from; t <= T; t++) {
            auto shift_to = t - shift_from;
            for (int i = 0; i < N; i++) {
                this->operator()(i, shift_to) = this->operator()(i, t);
            }
        }
        
        //zero-out anything after T - shift_from, then re-evolve those times
        for (int t = (T - shift_from); t < T; t++) {
            for (int i = 0; i < N; i++) {
                this->operator()(i, t) = 0;
            }
        }
        for (int t = (T - shift_from) - 1; t < T; t++) {
            this->take_timestep(t, true);
        }
    }
}

template<int N, int T>
void Trajectory<N, T>::shooting_move(const int shoot_from, const bool forward) {
    if (forward) {
        for (int t = shoot_from; t < (T - shoot_from); t++) {
            this->take_timestep(t, true);
        }
    } else {
        for (int t = shoot_from; t > 0; t--) {
            this->take_timestep(t, false);
        }
    }
}

template<int N, int T>
void Trajectory<N, T>::random_tps_move() {
    //choose whether to use a shifting move or a shooting move
    uint32_t rand_num = gen();
    bool which_one = rand_num & 1;
    
    //do it 
    (which_one) ? this->random_shooting_move() : this->random_shifting_move();
}

template<int N, int T>
void Trajectory<N, T>::random_shooting_move() {
    //choose whether to go forwards or backwards
    uint32_t rand_num = gen();
    bool which_way = rand_num & 1;

    //choose an index to shoot from
    std::uniform_int_distribution<> dist(1, T-1);
    auto shoot_from = dist(gen);
    
    //perform the shooting move
    this->shooting_move(shoot_from, which_way);
}

template<int N, int T>
void Trajectory<N, T>::random_shifting_move() {
    //choose whether to go forwards or backwards
    uint32_t rand_num = gen();
    bool which_way = rand_num & 1;

    //choose an index to shift from
    std::uniform_int_distribution<> dist(1, T-1);
    auto shift_from = dist(gen);
    
    //perform the shifting move
    this->shifting_move(shift_from, which_way);
}

template <int N, int T>
inline int Trajectory<N, T>::get_activity() const {
    return this->activity;
}

template <int N, int T>
inline double Trajectory<N, T>::get_intensive_activity() const {
    return (double) this->get_activity() / (N*T);
}

template <int N, int T>
inline int Trajectory<N, T>::get_num_up_spins() const {
    int n_up_spins = 0;
    for (int i = 0; i < N*T; i++) {
        if (data[i] == 1) { n_up_spins++; }
    }
    return n_up_spins;
}

template <int N, int T>
inline double Trajectory<N, T>::get_intensive_density() const {
    return (double) this->get_num_up_spins() / (N*T);
}

template<int N, int T>
inline int Trajectory<N, T>::get_num_in_patch(const std::tuple<int, int> lower_left_corner,
                                              const int temporal_extent,
                                              const int spatial_extent) {
    int count = 0;
    auto lower_space = std::get<0>(lower_left_corner);
    auto lower_time = std::get<1>(lower_left_corner);
    for (int t = 0; t < temporal_extent; t++) {
        for (int i = 0; i < spatial_extent; i++) {
            if (this->operator()(lower_space+i, lower_time+t) == 1) {
                count++;
            }
        }
    }
    return count;
}

template<int N, int T>
inline double Trajectory<N, T>::get_density_in_patch(const std::tuple<int, int> lower_left_corner,
                                                     const int temporal_extent,
                                                     const int spatial_extent) {
    auto num_up_spins = this->get_num_in_patch(lower_left_corner, temporal_extent, spatial_extent);
    return (double) num_up_spins / (spatial_extent * temporal_extent) ;
}

template <int N, int T>
std::ostream& operator<<(std::ostream& out, Trajectory<N, T> traj) {
    for (int i = 0; i < N; i++) {
        for (int t = 0; t < T; t++) {
            out << traj(i, t) << " "; 
        }
        out << std::endl;
    }
    return out;
}
