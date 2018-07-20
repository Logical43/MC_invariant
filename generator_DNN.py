### this class aims to construct a DNN to distinguish between different MC generator based on
### the initial decision value of the classifier. 

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
from keras.utils import plot_model
from keras.callbacks import History

# variables for the dataset
variables_map = {
    '2': ['nTrackJetsOR', 'mBB','dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV'],
    '3': ['nTrackJetsOR', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3']
}
nJets = '2'
variables = variables_map[nJets]

# custom note for file
seperator = "=========================================================================\n"
structure = "(100   linear)     \n(100   tanh * 2)  \n(100   relu * 2)    \n"
note = "Maximum decision metric, training on decision_value and mBB, binary crossentropy, hiding the third generator\n"
repeat = False 

# variables for the NN
LR = 0.005      # learning rate
MOM = 0.5       # momentum
DEC = 0.0001   # decay rate
BATCH = 200     # batch size
EPOCH = 200    # number of epochs

def main():
	# Prepare data
    df_even = pd.read_csv('DNN/adversary_even_log.csv', index_col=0)
    df_even = df_even.loc[df_even['Class']==0]
    df_even = df_even.loc[df_even['generator']!=2]

    df_odd = pd.read_csv('DNN/adversary_odd_log.csv', index_col=0)
    df_odd = df_odd.loc[df_odd['Class']==0]
    df_odd = df_odd.loc[df_odd['generator']!=2]

    ### is there anything to be done here? 
    df_odd[variables] = scale(df_odd[variables])
    df_even[variables] = scale(df_even[variables])

    #Convert mass bin number to categorical
    z_even = to_categorical(df_even['generator'], num_classes=2)
    z_odd = to_categorical(df_even['generator'], num_classes=2)

    # Construction of the neural network
    inputs_even = Input(shape=(2,))			# change it wrt number of variables R is trained on
    Rx_even = Dense(100, activation="linear")(inputs_even)

    Rx_even = Dense(100, activation="tanh")(inputs_even)
    Rx_even = Dense(100, activation="tanh")(inputs_even)
#    Rx_even = Dense(100, activation="tanh")(inputs_even)

    Rx_even = Dense(100, activation="relu")(inputs_even)
    Rx_even = Dense(100, activation="relu")(inputs_even)

    Rx_even = Dense(2, activation="softmax", name='adversary_output')(Rx_even)
    R_even = Model(input=[inputs_even], output=[Rx_even])

    R_even.name = 'R_even'

    # optimiser of R
    opt_R_even = SGD(lr=LR, momentum=MOM, decay=DEC)
    R_even.compile(loss='binary_crossentropy', optimizer=opt_R_even, metrics=['accuracy'])
    R_even.fit(df_even[['decision_value','mBB']], z_even, sample_weight=df_even['adversary_weights'], batch_size=BATCH, epochs=EPOCH)
    
    print("Model Evaluation:")
    print(R_even.metrics_names)
    print(R_even.evaluate(df_even[['decision_value','mBB']], z_even,steps=1))
    
    # making predictions and comparing results
    init_scores = R_even.predict(df_odd[['decision_value','mBB']])
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

    average_guess = (gen2_mean + gen1_mean + gen0_mean) / 3

    # printing and saving the results
    print("generator 1 guess: ",gen0_mean)
    print("generator 2 guess: ",gen1_mean)
    print("generator 3 guess: ",gen2_mean)
    
    print("generator 1 loss: ",len(gen0_scores)-sum(gen0_scores))
    print("generator 2 loss: ",len(gen1_scores)-sum(gen1_scores))
    print("generator 3 loss: ",len(gen2_scores)-sum(gen2_scores))

    print ('average guess: ',average_guess)
    
    # record the result in a separate txt file, easier to comment out
    record(gen0_mean,gen1_mean,gen2_mean)


# put 1 for the maximum prediction, 0 otherwise
#### best rewrite this
def maxProb(sing_pred):
    maxed_pred = np.zeros(3)
    max = sing_pred[np.argmax(sing_pred)]
#
#    if sing_pred[0] == max and sing_pred[1] == max and sing_pred[2] == max: 
#        return np.array([1/3,1/3,1/3])
#    if sing_pred[0] == max and sing_pred[1] == max: 
#        return np.array([0.5,0.5,0])
#    if sing_pred[0] == max and sing_pred[2] == max: 
#        return np.array([0.5,0,0.5])
#    if sing_pred[1] == max and sing_pred[2] == max: 
#        return np.array([0,0.5,0.5])
#        
#    else:
#        maxed_pred[np.argmax(sing_pred)] = 1 
#        return maxed_pred
    
    maxed_pred[np.argmax(sing_pred)] = 1 
    return maxed_pred
    
# record the result in a separate txt file from the global variables
def record(gen0_mean,gen1_mean,gen2_mean): 

    # opening file
    f= open("DNN_results.txt","a+")
    
    # initial variables, only prints if repeat == False
    if repeat == False: 
        f.write(seperator)
        f.write(structure)
        f.write(note)
        f.write("\nLearning rate: "+str(LR)+"      Momentum: "+str(MOM)+"\n")
        f.write("Decay rate: "+str(DEC)+"      Batch size: "+str(BATCH)+"\n")
        f.write("Number of epochs: "+str(EPOCH)+"\n")
        f.write(seperator)
    else: 
        f.write("---------\nrepeat: \n")
    
    # recording guess values
    f.write("generator 1 guess: "+str(gen0_mean))
    f.write("\n")
    f.write("generator 2 guess: "+str(gen1_mean))
    f.write("\n")
    f.write("generator 3 guess: "+str(gen2_mean))
    f.write("\n\n")
    
    average_guess = (gen2_mean + gen1_mean + gen0_mean) / 3
    
    f.write("average guess: "+str(average_guess))
    f.write("\n\n")
    f.close()

if __name__ == '__main__':
    main()
