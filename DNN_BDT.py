### this class tries to combine a NN signal-bkgd classifier and a generator BDT. 

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
#from histogramPlotATLAS import *
from sensitivity import *
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import *
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import scale
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import threading
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve as roc
from sklearn.metrics import auc,accuracy_score
from sklearn.model_selection import cross_val_score
from keras.utils import plot_model
from keras.callbacks import History
from six.moves import range
import os

# general parameters
variables_map = {
    '2': ['nTrackJetsOR', 'mBB','dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    '3': ['nTrackJetsOR', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3']
}
# BDT_variables = ['mBB','decision_value', 'pTB1', 'dPhiVBB','mTW','Mtop','pTV']
BDT_variables = ['mBB','decision_value']
nJets = '2'
variables = variables_map[nJets]
np.random.RandomState(21)
train = 'odd'
test  = 'even'

# NN parameters
lam = 1600
LR = 0.03      # learning rate
MOM = 0.5       # momentum
DEC = 0.0002   # decay rate
BATCH = 200     # batch size
EPOCH = 2     # number of epochs

# BDT parameters
variables = variables_map[nJets]
n_estimators = 200
max_depth = 4
learning_rate = 0.15

# Setting NN 
def NN():
    inputs = Input(shape=(len(variables),))
    Dx = Dense(400, activation="linear")(inputs)
    
    Dx = Dense(400, activation="tanh")(Dx)
#    Dx = Dense(400, activation="tanh")(Dx)
#    Dx = Dense(400, activation="tanh")(Dx)
#    Dx = Dense(400, activation="tanh")(Dx)
#    Dx = Dense(400, activation="tanh")(Dx)

    Dx = Dense(1, activation="sigmoid", name='classifier_output')(Dx)
    D = Model(input=[inputs], output=[Dx])
    D.name = 'D'
    
    return D

# Setting BDT 
def DT():
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=0.01),
                                          learning_rate=learning_rate,
                                          algorithm="SAMME",
                                          n_estimators=n_estimators
                                          )
    bdt.n_classes_ = 2
    return bdt
    
# Selecting data for the BDT 
def BDTdata(df):
    df = df.loc[df['Class']==0]        # background event
    df = df.loc[df['generator']!=2]    # excluding ph7; pythia and amcp8 only

    return df

# Setting adversarial NN based on the BDT output
def adversarial_NN():

def main():    
    # Prepare data
    df_train = pd.read_csv('results_v18/group_'+train+'.csv', index_col=0)
    df_test = pd.read_csv('results_v18/group_'+test+'.csv', index_col=0)

    df_train[variables] = scale(df_train[variables])
    df_test[variables] = scale(df_test[variables])
    
    # training the NN
    inputs = Input(shape=(len(variables),))
    DNN = NN()
    DNN.name = 'DNN'
    opt = SGD(lr=LR, momentum=MOM, decay=DEC)   # optimiser
    
    DNN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    DNN.fit(df_train[variables],df_train['Class'],sample_weight=df_train['training_weight'],
            epochs=EPOCH,batch_size=BATCH)
    
    # NN prediction, printing sensitivity
    df_train['decision_value'] = DNN.predict(df_train[variables])[:,0]
    df_test['decision_value'] = DNN.predict(df_test[variables])[:,0]
    sensitivity, error, bins = calc_sensitivity_with_error(df_test)
    print('Sensitivity for 2 Jet Dataset is: {:f}'.format(sensitivity)+' +/- {:f}'.format(error))
    
    # constructiong of BDT
    BDT = DT()
    BDT.name = 'BDT'
    
    # selecting generator events for the BDT
    bdt_train = BDTdata(df_train)
    bdt_test = BDTdata(df_test)
    
    BDT.fit(bdt_train[BDT_variables],bdt_train['generator'],sample_weight=bdt_train['EventWeight'])
    BDT_score = cross_val_score(BDT,bdt_test[BDT_variables],bdt_test['generator']).mean()

    bdt_test['bdt_decision'] = BDT.decision_function(bdt_test[BDT_variables]).tolist()
    bdt_train['bdt_decision'] = BDT.decision_function(bdt_train[BDT_variables]).tolist()

    
    # labels for the adversarial NN
    z_train = to_categorical(bdt_train['generator'], num_classes=2)
    z_test = to_categorical(bdt_test['generator'], num_classes=2)

    adv_inputs = Input(shape=(1,))
    Dx = Dense(10, activation="linear")(adv_inputs)
    Dx = Dense(10, activation="tanh")(adv_inputs)
    Dx = Dense(2, activation="softmax", name='adversary_output')(Dx)
    adv_NN = Model(input=[adv_inputs], output=[Dx])
    adv_NN.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    
    # fitting the adversarial
    adv_NN.fit(bdt_test['bdt_decision'],z_test,epochs=2,batch_size=32) # ignored event weight for
                                                                         # BDT output classification
    
    inputs_all = Input(shape=(13,))
    # building combined model
    opt_combined = SGD(lr=0.001, momentum=0.5, decay=0.00001)
    combined_NN = Model(input=[inputs_all], output=[DNN(inputs), adv_NN(adv_inputs)])
    combined_NN.compile(loss={'DNN':'binary_crossentropy','adv_NN':'binary_crossentropy'}, optimizer=opt_DRf, metrics=['accuracy'], loss_weights={'DNN': 1.0,'adv_NN': -lam})

    
    print('miraculously, this compiled')
    


if __name__ == '__main__':
    main()

