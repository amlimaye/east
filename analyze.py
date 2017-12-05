import east
import copy
import sys
import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import IPython

def harvest_from_unrestricted_ensemble(num_to_harvest, cutoff):
    harvested = []
    all_trajectories = []
    pjk_running = []
    num_tries = 0

    while len(harvested) < num_to_harvest:
        t = east.Trajectory(num_tries, 0.50)
        t.evolve_entire(0)
        num_tries += 1
        dens = t.density

        #if it meets the cutoff, keep this trajectory around
        if dens < cutoff:
            harvested.append(t)
            print('harvested %03d/%03d' % (len(harvested), num_to_harvest), end='\r')

        #either way, keep around _all_ the trajectories from the unrestricted ensemble
        all_trajectories.append(t)
        pjk_running.append(len(harvested)/num_tries)

    return harvested, pjk_running, all_trajectories

def harvest_from_restricted_ensemble(inits, num_to_harvest, prev_cutoff, next_cutoff):
    harvested = []
    pjk_running = []
    num_tries = 0
    num_tps_moves = 20

    while len(harvested) < num_to_harvest:
        #pick an init to try tps on
        init_idx = random.randint(0, len(inits)-1)
        t = east.Trajectory(inits[init_idx])
        for _ in range(num_tps_moves): 
            t.random_tps_move()
            dens = t.density

            #reject this move if we're not under the _previous_ cutoff
            #we want to sample from the _restricted_ ensemble!
            if dens > prev_cutoff:
                continue
            
            #if this move does produce a member of the restricted ensemble,
            #increase the counter for the number of samples we've drawn
            num_tries += 1

            if dens < next_cutoff:
                #we want to save this trajectory, make a copy of it
                save_me = east.Trajectory(t)
                harvested.append(save_me)
                print_harvested_progress_message(len(harvested), num_to_harvest)

                #if we got enough trajectories, update P_{jk} one last time and then exit
                if len(harvested) >= num_to_harvest:
                    pjk_running.append(len(harvested)/num_tries)
                    return harvested, pjk_running
        
            #append to the running array of P_{jk}
            pjk_running.append(len(harvested)/num_tries)

    return harvested, pjk_running

def compute_aggregate_histogram(cutoffs, pjk, p_unrestricted):
    #for each point in the cutoffs, the pjk array contains:
    # P(m < cutoff[i+1]) / P(m < cutoff [i]) since we sampled in a restricted ensemble
    
    #first, convert to an array of P(m < cutoff[i])
    p_cutoff = [p_unrestricted*pjk[0]]
    for idx in range(len(cutoffs)-1):
        p_cutoff.append(pjk[idx+1]*p_cutoff[idx])

    #p_cutoff is now an array of [P(m < cutoff[i]), P(m < cutoff[i+1]), P(m < cutoff[i+2]), ...]
    #we want to convert this into histogram bins, which we get from just successively differencing
    #p_cutoff -- np.diff does n[i+1] - n[i], we really want n[i] - n[i+1], so flip the sign
    histo = p_cutoff[1:]

    #nice trick to average each element
    cutoffs = np.array(cutoffs)
    bins = (cutoffs[1:] + cutoffs[:-1]) / 2

    return bins, histo

def do_histo(cutoff_bins):
    #load up the pjk array(s)
    pjk_dir = os.path.join('results', 'pjk')
    pjk_unrestricted = load(os.path.join(pjk_dir, 'unrestricted.pkl'))
    pjk_list = []
    for cutoff in cutoff_bins:
        this_pjk = load(os.path.join(pjk_dir, 'cutoff_%0.4f.pkl' % cutoff))
        pjk_list.append(this_pjk[-1])

    #compute the approxmation sans importance sampling (from the unrestricted ensemble)
    harvested_path = os.path.join('results', 'harvested', 'all_unrestricted.pkl')
    harvested = load(harvested_path)
    mean, std = check_average_density(harvested)

    #compute the aggregate histogram
    bins, histo = compute_aggregate_histogram(cutoff_bins, pjk_list, pjk_unrestricted[-1])
    make_histo(bins, histo, fname=os.path.join('results', 'plots', 'histo-combined.png'),
               also_gaussian=(mean, std))

def do_tps(cutoff_bins):
    num_to_harvest = 10000
    cutoff = 0.50

    #harvest from the unrestricted ensemble first
    harvested, pjk_running, all_trajectories = harvest_from_unrestricted_ensemble(num_to_harvest, cutoff)

    #save the results and make some plots
    dump(all_trajectories, os.path.join('results','harvested', 'all_unrestricted.pkl'))
    dump(harvested, os.path.join('results','harvested', 'unrestricted.pkl'))
    dump(pjk_running, os.path.join('results', 'pjk', 'unrestricted.pkl'))
    make_ts(range(len(pjk_running)), pjk_running,
            fname=os.path.join('results', 'plots', 'ts_p0-1.png'))

    #print a couple messages from this sampling
    print_ensemble_statistics(None, *check_average_density(all_trajectories))

    #loop through all the restricted ensembles
    for cutoff_idx in range(len(cutoff_bins)):
        #get the cutoff in the _previous_ ensemble
        if cutoff_idx == 0:
            prev_cutoff = cutoff
        else:
            prev_cutoff = cutoff_bins[cutoff_idx - 1]

        #get the next cutoff in this ensemble
        next_cutoff = cutoff_bins[cutoff_idx]

        #harvest from the restricted ensemble
        harvested, pjk_running = harvest_from_restricted_ensemble(harvested, num_to_harvest, 
                                                                  prev_cutoff, next_cutoff)

        #save the results
        dump(harvested, os.path.join('results','harvested', 'cutoff_%0.4f.pkl' % next_cutoff))
        dump(pjk_running, os.path.join('results', 'pjk', 'cutoff_%0.4f.pkl' % next_cutoff))
        make_ts(range(len(pjk_running)), pjk_running,
                fname=os.path.join('results', 'plots', 'ts_p%d-%d.png' % (cutoff_idx+1, cutoff_idx+2)), 
                e1=cutoff_idx+1, e2=cutoff_idx+2)

        #check density to make sure everything is okay
        print_ensemble_statistics(prev_cutoff, *check_average_density(harvested))

def main(sargs):
    mode = sargs[0]
    
    #set of cutoffs for forward (backward?) flux sampling
    cutoff_bins = [0.5 - 0.01*i for i in range(1, 25)]

    if mode == 'tps':
        do_tps(cutoff_bins)
    elif mode == 'histo':
        do_histo(cutoff_bins)
    else:
        print('unsupported mode %s' % mode)
        sys.exit(1)

#utility functions
def check_average_density(harvested):
    dens = np.array([h.density for h in harvested])
    mean = np.mean(dens)
    std = np.std(dens)
    return mean, std

def print_harvested_progress_message(num_harvested, num_to_harvest):
    print('harvested %03d/%03d' % (num_harvested, num_to_harvest), end='\r')

def print_ensemble_statistics(cutoff, mean, std):
    if cutoff is None:
        print('for unrestricted ensemble, <m> = %0.4f, sqrt(<(dm)^2>) = %0.4f' % (mean, std))
    else: 
        print('for m < %0.4f ensemble, <m> = %0.4f, sqrt(<(dm)^2>) = %0.4f' % (cutoff, mean, std))

def dump(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

def make_histo(bins, histo, fname='histo.png', also_gaussian=None):
    plt.figure()
    log_histo = np.log(histo)
    plot_me = log_histo - np.max(log_histo)
    plt.plot(bins, plot_me, 'bo-', label='Sampled')

    if also_gaussian is not None:
        mean, std = also_gaussian
        x = np.linspace(0.3, 0.55, 100)
        gaussian = -0.5*((x - mean)**2)/(std**2)
        plt.plot(x, gaussian, 'k--', label='Gaussian')

    #cosmetics
    plt.ylabel(r'$\log P(m)$')
    plt.xlabel(r'$m$')
    plt.legend()
    plt.savefig(fname)
    plt.close()

def make_ts(t, data, fname='ts.png', e1=0, e2=1):
    plt.figure()
    plt.plot(t, data, 'ko-')
    plt.ylabel(r'$P_{%d-%d}$' % (e1, e2))
    plt.xlabel(r'$\mathrm{Step}$')
    plt.savefig(fname)
    plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
