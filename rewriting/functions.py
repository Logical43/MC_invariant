from importing import *
from param import *

# Selecting data for the BDT 
def BDTdata(df):
    df = df.loc[df['Class']==0]        # background event
    df = df.loc[df['generator']!=2]    # excluding ph7; pythia and amcp8 only

    return df

# setting classifier structure
def classifier(inputs):
    """A 4 layer NN, input -> linear -> tanh -> sigmoid (output)"""
    Dx = Dense(400, activation='linear')(inputs)
    Dx = Dense(400, activation='tanh')(Dx)
    
    Dx = Dense(1, activation='sigmoid', name='classifier_output')(Dx)
    DNN = Model(input=[inputs], output=[Dx])
    DNN.name = 'DNN'
    return DNN

# Setting BDT 
def DT():
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=0.01),
                                          learning_rate=learning_rate,
                                          algorithm="SAMME",
                                          n_estimators=n_estimators
                                          )
    bdt.n_classes_ = 2
    bdt.name = 'BDT'
    return bdt

# Setting adversarial NN based on the BDT output
def adversarial_NN(DNN,adv_inputs):
    """A 4 layer NN, input from classifier -> linear -> tanh -> sigmoid (output)"""
    Dx = DNN(adv_inputs)
    Dx = Dense(10, activation="linear")(adv_inputs)
    Dx = Dense(10, activation="tanh")(Dx)
    Dx = Dense(2, activation="softmax", name='adversary_output')(Dx)
    adv_NN = Model(input=[adv_inputs], output=[Dx])
    adv_NN.name = 'adv_NN'
    return adv_NN
