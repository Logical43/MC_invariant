### this class aims to construct a DNN to distinguish between 3 different MC generator based on
### the initial decision value of the classifier. 

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

# variables for the dataset
variables_map = {
    '2': ['nTrackJetsOR', 'mBB','dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    '3': ['nTrackJetsOR', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3']
}
nJets = '2'
variables = variables_map[nJets]

# variables for the NN
LR = 0.01      # learning rate
MOM = 0.5       # momentum
DEC = 0.00005   # decay rate
BATCH = 16     # batch size
EPOCH = 1000    # number of epochs

def main():

    # Prepare data
    df_even = pd.read_csv('DNN/adversary_even_log.csv', index_col=0)
    df_even = df_even.loc[df_even['Class']==0]
#    df_even = df_even.loc[df_even['decision_value']<0.2]
    df_even = df_even.loc[df_even['pTB1']>0.3]
    df_even = df_even.loc[df_even['mBB']>0.3]
    print('generator 1:',len(df_even.loc[df_even['generator']==0]))
    print('generator 2:',len(df_even.loc[df_even['generator']==1]))
    print('generator 3:',len(df_even.loc[df_even['generator']==2]))

    df_odd = pd.read_csv('DNN/adversary_odd_log.csv', index_col=0)
    df_odd = df_odd.loc[df_odd['Class']==0]
    df_odd = df_odd.loc[df_odd['pTB1']>0.3]
    df_odd = df_odd.loc[df_odd['mBB']>0.3]
#    df_odd = df_odd.loc[df_odd['decision_value']<0.2]
    
    # scaling variables
    df_odd[variables] = scale(df_odd[variables])
    df_even[variables] = scale(df_even[variables])

    # Convert generator class to categorical
    z_even = to_categorical(df_even['generator'], num_classes=3)
    z_odd = to_categorical(df_odd['generator'], num_classes=3)
    
    # Construction of the neural network
    inputs_even = Input(shape=(7,))            # change it wrt number of variables R is trained on
    Rx_even = Sequential()
    Rx_even = Dense(1000, activation="linear")(inputs_even)
    
    Rx_even = Dense(10000, activation="tanh")(inputs_even)
    Rx_even = Dense(10000, activation="tanh")(inputs_even)

    Rx_even = Dense(3, activation="softmax", name='adversary_output')(Rx_even)
    R_even = Model(input=[inputs_even], output=[Rx_even])

    R_even.name = 'R_even'

    #class weight based on the sample number
    class_weight = {0: 1.,
                    1: 3.75,
#                    2: 4.892}
                    2: 4.25}
                    
    # optimiser of R
    opt_R_even = SGD(lr=LR, momentum=MOM, decay=DEC)
    R_even.compile(loss='categorical_crossentropy', optimizer=opt_R_even, metrics=['categorical_accuracy'])
    R_even.fit(df_even[['mBB','decision_value', 'pTB1', 'dPhiVBB','mTW','Mtop','pTV']], z_even, class_weight=class_weight, batch_size=BATCH, epochs=EPOCH)
    
    # making predictions and comparing results
    init_scores = R_even.predict(df_odd[['mBB','decision_value', 'pTB1', 'dPhiVBB','mTW','Mtop','pTV']])

    scores = []

    for i in range(0,np.size(init_scores, axis = 0)):
        scores.append(maxProb(init_scores[i]))
        
    # adding another column (the categorical probability) to the df object
    df_odd['gen_value_0'] = np.array(scores)[:,0]
    df_odd['gen_value_1'] = np.array(scores)[:,1]
    df_odd['gen_value_2'] = np.array(scores)[:,2]

    # locating the data for different generators
    df_gen0 = df_odd.loc[df_odd['generator'] == 0]  # pythiapw - normal
    df_gen1 = df_odd.loc[df_odd['generator'] == 1]  # aMCP8
    df_gen2 = df_odd.loc[df_odd['generator'] == 2]  # PH7

    # getting the generator scores from the df object
    gen0_scores = df_gen0['gen_value_0'].tolist()
    gen1_scores = df_gen1['gen_value_1'].tolist()
    gen2_scores = df_gen2['gen_value_2'].tolist()

    # obtaining the average decision value across the df object
    gen0_mean = sum(gen0_scores)/len(gen0_scores)
    gen1_mean = sum(gen1_scores)/len(gen1_scores)
    gen2_mean = sum(gen2_scores)/len(gen2_scores)

#    average_guess = (gen0_mean + gen1_mean) / 2
    average_guess = (sum(gen0_scores) + sum(gen1_scores))/(len(gen0_scores) + len(gen1_scores))
    
    # printing results
    print("generator 1 guess: ",gen0_mean)
    print("generator 2 guess: ",gen1_mean)
    print("generator 3 guess: ",gen2_mean)
    print("% above random guess: ",(gen0_mean+gen1_mean+gen2_mean-1)*100,"%")

    print("generator 1 loss: ",len(gen0_scores)-sum(gen0_scores))
    print("generator 2 loss: ",len(gen1_scores)-sum(gen1_scores))
    print("generator 3 loss: ",len(gen2_scores)-sum(gen2_scores))

    print ('average guess: ',average_guess)
    
# put 1 for the maximum prediction, 0 otherwise
#### better rewrite this
def maxProb(sing_pred):
    maxed_pred = np.zeros(3)
    maxed_pred[np.argmax(sing_pred)] = 1 
    return maxed_pred

if __name__ == '__main__':
    main()
