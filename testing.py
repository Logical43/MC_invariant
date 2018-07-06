### this is the class where DNN and adversary DNN are defined 

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import pandas as pd
from sensitivity import *
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import *
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import scale
from keras.utils import plot_model
from keras.callbacks import History
from six.moves import range
import os

# global variables
variables_map = {
    2: ['nTrackJetsOR', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV','generator'],
    3: ['nTrackJetsOR', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3', 'MV1cJ3_cont','generator']
}
nJets = 2
variables = variables_map[nJets]

# making directory if not already exists
outdir = './adv_results/'
if not os.path.exists(outdir):
    os.mkdir(outdir)

# structuring, building and training NNs 
def main():
    # Set parameters
    lam = 400
    np.random.RandomState(21)   #random seed generator
    
    test = 'odd'
    train  = 'even'

    ### swapping even and odd
#    test = 'even'
#    train  = 'odd'

    # Prepare data
    df_train = pd.read_csv('results_v18/group_'+train+'.csv', index_col=0)
    df_test = pd.read_csv('results_v18/group_'+test+'.csv', index_col=0)
    #Convert MC to categorical
    z_train = to_categorical(df_train['generator'], num_classes=4)

    # Set network architectures for the classifier
    inputs = Input(shape=(df_train[variables].shape[1],))
    Dx = Dense(40, activation="linear")(inputs)
    Dx = Dense(40, activation="tanh")(Dx)
    Dx = Dense(40, activation="tanh")(Dx)
    Dx = Dense(1, activation="sigmoid", name='classifier_output')(Dx)
    D = Model(input=[inputs], output=[Dx])
    D.name = 'D'

    # Set network architectures for the adversary
    Rx = D(inputs)
    Rx = Dense(30, activation="tanh")(Rx)
    Rx = Dense(4, activation="softmax", name='adversary_output')(Rx)
    R = Model(input=[inputs], output=[Rx])
    R.name = 'R'

    #Build and compile models
    """ Build D (Classifier)"""
    opt_D = SGD(lr=0.00001, momentum=0.5, decay=0.00001)
    D.trainable = True; R.trainable = False     # train classifier, freeze adversary 
    D.compile(loss='binary_crossentropy', optimizer=opt_D, metrics=['binary_accuracy'])
    D.fit(df_train[variables], df_train['Class'], sample_weight=df_train['training_weight'], epochs=5, batch_size=32)
    
    test_classifier(D,df_test,lam,'start',test)

    # Pretraining of R
    opt_DfR = SGD(lr=1, momentum=0.5, decay=0.00001)
    D.trainable = False; R.trainable = True
    R.compile(loss='categorical_crossentropy', optimizer=opt_DfR, metrics=['mse'])
    R.fit(df_train[variables], z_train, sample_weight=df_train['adversary_weights'],batch_size=128, epochs=5)

    """ Build DRf (Model with combined loss function) """
    opt_DRf = SGD(lr=0.001, momentum=0.5, decay=0.00001)
    DRf = Model(input=[inputs], output=[D(inputs), R(inputs)])
    D.trainable = True; R.trainable = False
    DRf.compile(loss={'D':'binary_crossentropy','R':'categorical_crossentropy'}, optimizer=opt_DRf, metrics=['accuracy'], loss_weights={'D': 1.0,'R': -lam})

    #Set to 100 epochs
    max = len(df_train)
    batch_size = 100
    
    #table tennis
    for i in range(1,max):
        
        if (i%1000 == 0):
            print("Iteration Number: " + str(i))
        
        indices = np.random.permutation(len(df_train))[:batch_size]
        DRf.train_on_batch(df_train[variables].iloc[indices], [df_train['Class'].iloc[indices], z_train[indices]], sample_weight=[df_train['training_weight'].iloc[indices],df_train['adversary_weights'].iloc[indices]])
        R.train_on_batch(df_train[variables].iloc[indices], z_train[indices], df_train['adversary_weights'].iloc[indices])
        
        if (i%2500 == 0):
            test_classifier(D,df_test,lam,str(i),test)

    test_classifier(D,df_test,lam,"end",test)
    D.save_weights(str(nJets)+"_"+str(lam)+"_"+train+".h5")

def test_classifier(D,df,lam,stage,test):    #tests classifier model and saves results
    scores = D.predict(df[variables])[:,0]
    df['decision_value'] = ((scores-0.5)*2)
    sensitivity, error = calc_sensitivity_with_error(df)
    print('Sensitivity for 2 Jet Dataset is: {:f}'.format(sensitivity)+' +/- {:f}'.format(error))
    df['sensitivity'] = sensitivity
#    df.to_csv(path_or_buf='./adv_results/'+test+'_batch_'+stage+'_'+str(lam)+'.csv')
    df.to_csv(path_or_buf= os.path.join(outdir+test+'_batch_'+stage+'_'+str(lam)+'.csv'))

if __name__ == '__main__':
    main()



