import east
import copy
import sys
import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import IPython
from collections import namedtuple

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

def harvest_from_patch_unrestricted_ensemble(num_to_harvest, patch):
    harvested = []
    all_trajectories = []
    pjk_running = []
    num_tries = 0

    while len(harvested) < num_to_harvest:
        t = east.Trajectory(num_tries, 0.50)
        t.evolve_entire(0)
        num_tries += 1

        num_patch = t.patch_num(patch.corner, patch.T, patch.S)

        #if it meets the cutoff, keep this trajectory around
        if num_patch == 0:
            harvested.append(t)
            print('harvested %03d/%03d' % (len(harvested), num_to_harvest), end='\r')

        #either way, keep around _all_ the trajectories from the unrestricted ensemble
        all_trajectories.append(t)
        pjk_running.append(len(harvested)/num_tries)

    return harvested, pjk_running, all_trajectories

def harvest_from_patch_restricted_ensemble(inits, num_to_harvest, prev_patch, next_patch):
    harvested = []
    pjk_running = []
    num_tries = 0
    num_tps_moves = 20

    while len(harvested) < num_to_harvest:
        #pick an init to try tps on
        init_idx = random.randint(0, len(inits)-1)
        t = east.Trajectory(inits[init_idx])
        for _ in range(num_tps_moves): 
            t.random_shooting_move()

            num_prev_patch = t.patch_num(prev_patch.corner, prev_patch.T, prev_patch.S)

            #reject this move if we don't have zero occupancy in the _previous_ patch
            #we want to sample from the _restricted_ ensemble!
            if num_prev_patch != 0:
                continue
            
            #if this move does produce a member of the restricted ensemble,
            #increase the counter for the number of samples we've drawn
            num_tries += 1

            #compute the number in the next patch
            num_next_patch = t.patch_num(next_patch.corner, next_patch.T, next_patch.S)

            if num_next_patch == 0:
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
    # P_i(m < cutoff[i+1]) where the subscript denotes estimate in a restricted ensemble with cutoff i
    #to correct this probability, we need to multiply it by 
    
    #first, convert to an array of P(m < cutoff[i])
    p_cutoff = [p_unrestricted*pjk[0]]
    for idx in range(len(cutoffs)-1):
        p_cutoff.append(pjk[idx+1]*p_cutoff[idx])

    #p_cutoff is now an array of [P(m < cutoff[i]), P(m < cutoff[i+1]), P(m < cutoff[i+2]), ...]
    #we want to convert this into histogram bins, which we get from just successively differencing
    #p_cutoff -- np.diff does n[i+1] - n[i], we really want n[i] - n[i+1], so flip the sign
    #histo = -1*np.diff(p_cutoff)
    histo = -1*np.diff((np.array([p_unrestricted] + p_cutoff)))

    #nice trick to average each element
    cutoffs = np.array([0.5] + cutoffs)
    bins = (cutoffs[1:] + cutoffs[:-1]) / 2
    
    IPython.embed()

    return bins, histo

def compute_aggregate_patch_histogram(patch_sizes, pjk, p_unrestricted):
    #for each point in the cutoffs, the pjk array contains:
    # P_i(m < cutoff[i+1]) where the subscript denotes estimate in a restricted ensemble with cutoff i
    #to correct this probability, we need to multiply it by 
    
    #first, convert to an array of P(m < cutoff[i])
    p_patch = [p_unrestricted*pjk[0]]
    for idx in range(len(patch_sizes)-1):
        p_patch.append(pjk[idx+1]*p_patch[idx])

    #p_cutoff is now an array of [P(m < cutoff[i]), P(m < cutoff[i+1]), P(m < cutoff[i+2]), ...]
    #we want to convert this into histogram bins, which we get from just successively differencing
    #p_cutoff -- np.diff does n[i+1] - n[i], we really want n[i] - n[i+1], so flip the sign
    #histo = -1*np.diff(p_cutoff)
    histo = np.array(p_patch)

    #nice trick to average each element
    patch_sizes = np.array(patch_sizes)
    bins = patch_sizes

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

def do_histo_patch(patches):
    #load up the pjk array(s)
    pjk_dir = os.path.join('results-patch', 'pjk')
    pjk_unrestricted = load(os.path.join(pjk_dir, 'unrestricted.pkl'))
    pjk_list = []
    for patch in patches[:-1]:
        this_pjk = load(os.path.join(pjk_dir, 'cutoff_psize_%d.pkl' % patch.T))
        pjk_list.append(this_pjk[-1])

    #compute the aggregate histogram
    patch_sizes = [patch.T for patch in patches[:-1]]
    bins, histo = compute_aggregate_patch_histogram(patch_sizes, pjk_list, pjk_unrestricted[-1])
    make_histo(bins, histo, fname=os.path.join('results-patch', 'plots', 'histo-combined.png'), flip=True)
 
def do_tps(cutoff_bins):
    num_to_harvest = 50000
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

def do_tps_patch(patches):
    num_to_harvest = 100000
    patch = patches[0]
    
    #harvest from the unrestricted ensemble first
    harvested, pjk_running, all_trajectories = harvest_from_patch_unrestricted_ensemble(num_to_harvest/10, patch)

    #save the results and make some plots
    dump(all_trajectories, os.path.join('results-patch','harvested', 'all_unrestricted.pkl'))
    dump(harvested, os.path.join('results-patch','harvested', 'unrestricted.pkl'))
    dump(pjk_running, os.path.join('results-patch', 'pjk', 'unrestricted.pkl'))
    make_ts(range(len(pjk_running)), pjk_running,
            fname=os.path.join('results-patch', 'plots', 'ts_p0-1.png'))

    #loop through all the restricted patch ensembles
    for patch_idx in range(1, len(patches)):
        #get the patch in the _previous_ ensemble
        prev_patch = patches[patch_idx - 1]

        #get the next cutoff in this ensemble
        next_patch = patches[patch_idx]

        #harvest from the restricted ensemble
        if patch_idx > (len(patches) - 7):
            num_to_harvest = 1000000
            print("Boosting number of harvested trajectories by 10x to: %d" % num_to_harvest)

        harvested, pjk_running = harvest_from_patch_restricted_ensemble(harvested, num_to_harvest, 
                                                                        prev_patch, next_patch)

        #save the results
        dump(harvested, os.path.join('results-patch','harvested', 'cutoff_psize_%d.pkl' % prev_patch.T))
        dump(pjk_running, os.path.join('results-patch', 'pjk', 'cutoff_psize_%d.pkl' % prev_patch.T))
        make_ts(range(len(pjk_running)), pjk_running,
                fname=os.path.join('results-patch', 'plots', 'ts_p%d-%d.png' % (patch_idx+1, patch_idx+2)), 
                e1=patch_idx+1, e2=patch_idx+2)

        #check density to make sure everything is okay
        print_ensemble_statistics(prev_patch.T, *check_patch_occupancy(harvested, prev_patch))

    IPython.embed()

def main(sargs):
    mode = sargs[0]
    
    #set of cutoffs for forward (backward?) flux sampling
    cutoff_bins = [0.5 - 0.05*i for i in range(1, 9)]

    #set of patches to evacuate spins from
    Patch = namedtuple('Patch', ['corner','S','T'])
    patch_corner = (10, 5)
    patch_extents = list(range(1, 20))
    patches = [Patch(corner=patch_corner, S=3, T=e) for e in patch_extents]

    if mode == 'tps':
        do_tps(cutoff_bins)
    elif mode == 'histo':
        do_histo(cutoff_bins)
    elif mode == 'patch-tps':
        do_tps_patch(patches)
    elif mode =='patch-histo':
        do_histo_patch(patches)
    else:
        print('unsupported mode %s' % mode)
        sys.exit(1)

#utility functions
def check_average_density(harvested):
    dens = np.array([h.density for h in harvested])
    mean = np.mean(dens)
    std = np.std(dens)
    return mean, std

def check_patch_occupancy(harvested, patch):
    occ = np.array([h.patch_num(patch.corner, patch.T, patch.S) for h in harvested])
    mean = np.mean(occ)
    std = np.std(occ)
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
        pickle.dump(obj, f, protocol=2)

def load(fname):
    with open(fname, 'rb') as f:
        pickle.load(f, protocol=2)

    return obj

def make_histo(bins, histo, fname='histo.png', also_gaussian=None, flip=False):
    plt.figure()
    log_histo = np.log(histo)
    plot_me = log_histo - np.max(log_histo)
    
    if flip:
        plot_me *= -1

    plt.plot(bins, plot_me, 'bo-', label='Sampled')

    if also_gaussian is not None:
        mean, std = also_gaussian
        x = np.linspace(0.20, 0.55, 100)
        gaussian = -0.5*((x - mean)**2)/(std**2)
        plt.plot(x, gaussian, 'k--', label='Gaussian')

    #cosmetics
    if flip:
        plt.ylabel(r'$-\log P(m)$')
    else:
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
