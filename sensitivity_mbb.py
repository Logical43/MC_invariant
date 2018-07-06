from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import scale
from datetime import datetime
from six.moves import zip

def calc_sensitivity_with_error_mbb(df):
    """Calculate sensitivity from dataframe with error"""
    
    bins, bin_sums_w2_s, bin_sums_w2_b = trafoD_with_error_mbb(df)

    # Initialise sensitivity and error.
    sens_sq = 0
    error_sq = 0
    
    #Split into signal and background events
    classes = df['Class']
    dec_vals = df['mBB_raw']
    weights = df['EventWeight']
    
    bins = np.linspace(0,500000,26)
    print("bins set")
    
    y_data = list(zip(classes, dec_vals, weights))
    
    events_sb = [[a[1] for a in y_data if a[0] == 1], [a[1] for a in y_data if a[0] == 0]]
    weights_sb = [[a[2] for a in y_data if a[0] == 1], [a[2] for a in y_data if a[0] == 0]]

    #plots histogram with optimised bins and counts number of signal and background events in each bin
    plt.ioff()
    counts_sb = plt.hist(events_sb,
                         bins=bins,
                         weights=weights_sb)[0]
    plt.close()
    plt.ion()

    # Reverse the counts before calculating.
    # Zip up S, B, DS and DB per bin.
    s_stack = counts_sb[0][::-1]   #counts height of signal in each bin from +1 to -1
    b_stack = counts_sb[1][::-1]    #counts height of bkground in each bin from +1 to -1
    ds_sq_stack = bin_sums_w2_s[::-1]
    db_sq_stack = bin_sums_w2_b[::-1]
    
    for s, b, ds_sq, db_sq in zip(s_stack, b_stack, ds_sq_stack, db_sq_stack): #iterates through every bin
        this_sens = 2 * ((s + b) * math.log(1 + s / b) - s) #calcs sensivity for each bin
        this_dsens_ds = 2 * math.log(1 + s/b)
        this_dsens_db = 2 * (math.log(1 + s/b) - s/b)
        this_error = (this_dsens_ds ** 2) * ds_sq + (this_dsens_db ** 2) * db_sq
        if not math.isnan(this_sens):   #unless bin empty add this_sense to sens_sq total (sums each bin sensitivity)
            sens_sq += this_sens
        if not math.isnan(this_error):  #unless bin empty add this_error to error_sq total
            error_sq += this_error

    # Sqrt operations and error equation balancing.
    sens = math.sqrt(sens_sq)
    error = 0.5 * math.sqrt(error_sq/sens_sq)
    
    return sens, error


def trafoD_with_error_mbb(df, initial_bins=10000, z_s=10, z_b=10): #total number of bins = z_s + z_b
    """Output optimised histogram bin widths from a list of events"""
    
    df = df.sort_values(by='mBB_raw')

    N_s = sum(df['post_fit_weight']*df['Class'])
    N_b = sum(df['post_fit_weight']*(1-df['Class']))
    

    # Set up scan parameters.
    scan_points = np.linspace(00, 4000000, num=initial_bins).tolist()[1:-1]
    scan_points = scan_points[::-1] #invert list
    
    # Initialise z and bin list.
    z = 0
    bins = [4005000]
    sum_w2_s = 0
    sum_w2_b = 0
    delta_bins_s = list()
    delta_bins_b = list()

    decision_values_list = df['mBB_raw'].tolist()
    class_values_list = df['Class'].tolist()
    post_fit_weights_values_list = df['post_fit_weight'].tolist()

    try:
        # Iterate over bin low edges in scan.
        for p in scan_points:
            # Initialise freq count for this bin
            sig_bin = 0
            back_bin = 0
            
            # Current bin loop.
            # Remember, events are in descending DV order.
            while True:
                """ This loop sums the post_fit_weight and p.f.w squared of signal and of background events contained in each of the initial bins"""
                # End algo if no events left - update z and then IndexError
                
                if not decision_values_list: #if not empty (NEVER CALLS THIS CODE)
                    z += z_s * sig_bin / N_s + z_b * back_bin / N_b
                    if z > 1:
                        bins.insert(0, p)
                        delta_bins_s.insert(0, sum_w2_s)
                        delta_bins_b.insert(0, sum_w2_b)
                    raise IndexError
            
                
                # Break if DV not in bin. (i.e when finished scanning over each of the inital bins)
                if decision_values_list[-1] < p:  #negative index counts from the right (i.e last object)
                    break
                
                # Pop the event.
                decison_val = decision_values_list.pop()
                class_val = class_values_list.pop()
                post_fit_weight_val = post_fit_weights_values_list.pop()
                
                # Add freq to S/B count, and the square to sums of w2.
                if class_val == 1:
                    sig_bin += post_fit_weight_val
                    sum_w2_s += post_fit_weight_val ** 2
                else:
                    back_bin += post_fit_weight_val
                    sum_w2_b += post_fit_weight_val ** 2
        
            # Update z for current bin.
            z += z_s * sig_bin / N_s + z_b * back_bin / N_b   #10*(% of total signal + # of total background)
            
            # Reset z and update bin
            if z > 1:
                bins.insert(0, p)
                z = 0
                
                # Update sum_w2 for this bin.
                delta_bins_s.insert(0, sum_w2_s)
                delta_bins_b.insert(0, sum_w2_b)
                sum_w2_s = 0 #not sure why this is at the end and sig_bin/back_bin reset at the beginning
                sum_w2_b = 0
            
    except IndexError:
        rewje = 0

    finally:
        bins.insert(0,0)
        delta_bins_s.insert(0, sum_w2_s)  #sum of signal event weights^2 for each bin
        delta_bins_b.insert(0, sum_w2_b)  #sum of background event weights^2 for each bin
        return bins, delta_bins_s, delta_bins_b

