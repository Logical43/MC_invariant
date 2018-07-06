from __future__ import absolute_import
import sys
from six.moves import range
sys.path.append("/Users/patrickgreenway/ENV/lib/python2.7/site-packages")

import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.text import OffsetFrom
from sensitivity_mbb import *

nJets = 2

def main():
    df1_even = pd.read_csv('./adv_results/even_batch_end_400.csv', index_col=0)
    df1_odd = pd.read_csv('./adv_results/odd_batch_end_400.csv', index_col=0)
    df = df1_odd.append(df1_even)
    ## delete the following line if needed:
#    df = df1_even
    df = df.reset_index(drop=True)
    
    cut = 0.3
    df_cut = df.loc[df['decision_value'] >= cut]
    df_cut = df_cut.reset_index(drop=True)
    
    # Initialise plot stuff
    plt.ion()
    plt.clf()
    plot_range = (0, 600)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)
    
    plotmBB(df,df_cut)

def plotmBB(df,df_cut):
    
    bins = np.linspace(0,500,51)

    # Initialise plot stuff
    plt.ion()
    plt.clf()
    plot_range = (0, 500)
    plot_data = []
    plot_weights = []
    plot_colors = []
    plt.rc('font', weight='bold')
    plt.rc('xtick.major', size=5, pad=7)
    plt.rc('xtick', labelsize=10)

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["mathtext.default"] = "regular"
    
    df_signal = df.loc[df['Class'] ==1.0]
    df_background = df.loc[df['Class'] ==0.0]
    
    df_signal_cut = df_cut.loc[df_cut['Class'] ==1.0]
    df_background_cut = df_cut.loc[df_cut['Class'] ==0.0]

    plot_weights_signal = df_signal['post_fit_weight']
    plot_weights_signal = (1/sum(plot_weights_signal))*plot_weights_signal
    plot_weights_signal_cut = df_signal_cut['post_fit_weight']
    plot_weights_signal_cut = (1/sum(plot_weights_signal_cut))*plot_weights_signal_cut

    plot_weights_background = df_background['post_fit_weight']
    plot_weights_background = (1/sum(plot_weights_background))*plot_weights_background
    plot_weights_background_cut = df_background_cut['post_fit_weight']
    plot_weights_background_cut = (1/sum(plot_weights_background_cut))*plot_weights_background_cut
    
    mbb_vals = df['mBB'].tolist()
    for i in range(len(df)):
        if mbb_vals[i]>=500000:
            mbb_vals[i] = 499500

    df['mBB'] = mbb_vals

    mbb_vals_cut = df_cut['mBB'].tolist()
    for i in range(len(df_cut)):
        if mbb_vals_cut[i]>=500000:
            mbb_vals_cut[i] = 499500

    df_cut['mBB'] = mbb_vals_cut

    variable_values_signal = df_signal['mBB']/1000
    variable_values_background = df_background['mBB']/1000
    variable_values_signal_cut = df_signal_cut['mBB']/1000
    variable_values_background_cut = df_background_cut['mBB']/1000



    if True == False:
        count_sig_cut=plt.hist(variable_values_signal_cut,
             bins=bins,
             weights=plot_weights_signal_cut,
             range=plot_range,
             rwidth=1,
             color='#b32400', #dark red
             label='Signal after cut',
             stacked=False,
             alpha=1,
             #hatch=4*'\\'
             )
    if True != False:
        count_back_cut=plt.hist(variable_values_background_cut,
              bins=bins,
              weights=plot_weights_background_cut,
              range=plot_range,
              rwidth=1,
              linewidth=3,
              color='#0066CC',     #dark blue
              label='Background after cut',
              alpha=1,
              #hatch=4*'//'
              )
    if True == False:
        # Plot.
        count_sig=plt.hist(variable_values_signal,
                 bins=bins,
                 weights=plot_weights_signal,
                 range=plot_range,
                 rwidth=1,
                 histtype = 'step',
                 linewidth=3,
                 edgecolor='#FF0000',
                 #edgecolor='red',
                 label='Signal',
                 stacked=False,
                 alpha=1,
                 #hatch=4*'\\'
                 )
    if True != False:
        count_back=plt.hist(variable_values_background,
                  bins=bins,
                  weights=plot_weights_background,
                  range=plot_range,
                  rwidth=1,
                  histtype = 'step',
                  linewidth=3,
                  edgecolor='#0000b3',
                 #edgecolor='blue',
                  label='Background',
                  alpha=1,
                 #hatch=4*'//'
                 )
   
    x1, x2, y1, y2 = plt.axis()
    #plt.yscale('log', nonposy='clip')
    plt.axis((x1, x2, y1, y2 * 1.2))
    axes = plt.gca()
    axes.set_ylim([0,0.12])
    axes.set_xlim(plot_range)
    x = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]
    #plt.xticks(x, x,fontweight = 'normal',fontsize = 16)
    y = ["10",r"10$^{2}$",r"10$^{3}$",r"10$^{4}$",r"10$^{5}$"]
    yi = [10,100,1000,10000,100000]
    #plt.yticks(yi, y,fontweight = 'normal',fontsize = 16)
    axes.yaxis.set_ticks_position('both')
    axes.yaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(which='both', direction='in')
    axes.xaxis.set_minor_locator(AutoMinorLocator(4))
    axes.yaxis.set_minor_locator(AutoMinorLocator(4))
    handles, labels = axes.get_legend_handles_labels()
    plt.legend(loc='upper right', ncol=1, prop={'size': 12},frameon=False,
                handles=handles[::-1])
    plt.ylabel('Normalised Number of Events',fontsize = 16,fontweight='normal')
    axes.yaxis.set_label_coords(-0.1,0.5)
    plt.xlabel("$m_{bb}$ [GeV]",fontsize = 16,fontweight='normal')
    axes.xaxis.set_label_coords(0.89, -0.07)
    an1 = axes.annotate("ATLAS", xy=(0.05, 0.91), xycoords=axes.transAxes,fontstyle = 'italic',fontsize = 16)
     
    offset_from = OffsetFrom(an1, (0, -1.4))
    an2 = axes.annotate(r'$\sqrt{s}$' + " = 13 TeV , 36.1 fb$^{-1}$", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from, fontweight='normal',fontsize = 12)
     
    offset_from = OffsetFrom(an2, (0, -1.4))
    an3 = axes.annotate("1 lepton, "+str(nJets)+" jets, 2 b-tags", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)
     
    offset_from = OffsetFrom(an3, (0, -1.6))
    an4 = axes.annotate("p$^V_T \geq$ 150 GeV", xy=(0.05,0.91), xycoords=axes.transAxes, textcoords=offset_from,fontstyle = 'italic',fontsize = 12)
    plt.show(block=True)
#print count_sig

#print count_back
#print count_sig_cut
#print count_back_cut

if __name__ == '__main__':
    main()



