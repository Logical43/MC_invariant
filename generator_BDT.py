### this class aims to construct a BDT to distinguish between different MC generator based on
### the initial decision value of the classifier. 
import pandas
import numpy
import sys
import pickle

import matplotlib.cm as cm
from sklearn.preprocessing import scale
from sensitivity import *
from xgboost import XGBClassifier
from IPython.display import  display
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import threading
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import auc

# variables for the dataset
variables_map = {
#    '2': ['nTrackJetsOR', 'mBB','dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    # variables used for adversarial NN
    '2': ['mBB','decision_value', 'pTB1', 'dPhiVBB','mTW','Mtop','pTV'],

    '3': ['nTrackJetsOR', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3']
}
# hyperparams
nJets = '2'
variables = variables_map[nJets]
n_estimators = 200
max_depth = 4
learning_rate = 0.15

def main():
    # Prepare data
    df_even = pd.read_csv('DNN/adversary_even_log.csv', index_col=0)
    df_even = df_even.loc[df_even['Class']==0]
    df_even = df_even.loc[df_even['generator']!=2]
    df_even = df_even.loc[df_even['nTags']==2]

    df_odd = pd.read_csv('DNN/adversary_odd_log.csv', index_col=0)
    df_odd = df_odd.loc[df_odd['Class']==0]
    df_odd = df_odd.loc[df_odd['generator']!=2]
    df_odd = df_odd.loc[df_odd['nTags']==2]
    
    df_odd[variables] = scale(df_odd[variables])
    df_even[variables] = scale(df_even[variables])
    
    # construction of bdt
    bdt_even = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=0.01),
                                          learning_rate=learning_rate,
                                          algorithm="SAMME",
                                          n_estimators=n_estimators
                                          )
    bdt_even.n_classes_ = 2
    bdt_odd = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=0.01),
                                         learning_rate=0.15,
                                         algorithm="SAMME",
                                         n_estimators=n_estimators
                                         )
    bdt_odd.n_classes_ = 2
    
    # Convert generator class to categorical
    z_even = to_categorical(df_even['generator'], num_classes=2)
    z_odd = to_categorical(df_odd['generator'], num_classes=2)
    
    # fitting to generators
    bdt_even.fit(df_even[variables], df_even['generator'], sample_weight=df_even['EventWeight'])
    bdt_odd.fit(df_odd[variables], df_odd['generator'], sample_weight=df_odd['EventWeight'])
    
    # Scoring
    df_odd['bdt_outcome'] = bdt_odd.decision_function(df_odd[variables]).tolist()
    df_even['bdt_outcome'] = bdt_even.decision_function(df_even[variables]).tolist()

    print(bdt_odd.score())
    df = pd.concat([df_odd,df_even])
    
    # plotting BDT outcome for different generators
    gen1 = df.loc[df['generator']==0]
    gen2 = df.loc[df['generator']==1]
    
#    plt.hist(gen1['bdt_outcome'],bins=70,color='red',alpha=0.5,density=True)
#    plt.hist(gen2['bdt_outcome'],bins=70,color='blue',alpha=0.5,density=True)
#    
#    plt.show()
    
    # calculating fpr, tpr and auc for roc curve
    fpr = dict()
    tpr = dict()
    area = dict()
    
    z_all = to_categorical(df['generator'], num_classes=2)

    fpr[0], tpr[0], _ = roc(z_all[:,0], df['bdt_outcome'],sample_weight=df['EventWeight'])
#    area[0] = auc(fpr[0], tpr[0])
    
    # plotting the roc curve
#    plt.plot(fpr[0], tpr[0], color='darkorange',lw=1,label='PYTHIA, ROC curve (area = %0.2f)' % area[0])
#    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
#    plt.legend()
#    plt.savefig('roc.png', bbox_inches='tight')
#    plt.show(block=True)

    print('It runs. ')

if __name__ == '__main__':
    main()
