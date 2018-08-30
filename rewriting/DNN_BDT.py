from importing import *
from param import *
from functions import *

np.random.RandomState(21)

# 
def main():    
    # Getting data from csv file
    df_train = pd.read_csv ('../results_v18/group_'+train+'.csv', index_col=0)
    df_test = pd.read_csv('../results_v18/group_'+test+'.csv', index_col=0)
    
    df_train['bdt_decision'] = np.zeros(len(df_train))  # initialising the bdt_decision var
    df_test['bdt_decision'] = np.zeros(len(df_test))
        
    # scaling
    df_train[variables] = scale(df_train[variables])
    df_test[variables] = scale(df_test[variables])

    # constructing classifier and adversary
    inputs = Input(shape=(len(variables),))
    
    DNN = classifier(inputs)                      # classifier NN structure (see functions.py)
    DNN_opt = SGD(lr=LR, momentum=MOM, decay=DEC) # classifier optimiser

    adv_NN = adversarial_NN(DNN,inputs)             # adversarial NN structure (see functions.py)
    adv_opt = SGD(lr=LR, momentum=MOM, decay=DEC)   # adversarial optimiser

    # classifier training
    DNN.trainable = True; adv_NN.trainable = False
    DNN.compile(loss='binary_crossentropy', optimizer=DNN_opt, metrics=['binary_accuracy'])
    DNN.fit(df_train[variables],df_train['Class'],sample_weight=df_train['training_weight'],
            epochs=EPOCH,batch_size=BATCH)
    
    # NN prediction, printing sensitivity
    df_train['decision_value'] = DNN.predict(df_train[variables])[:,0]
    df_test['decision_value'] = DNN.predict(df_test[variables])[:,0]
    
    # selecting background data for BDT
    bdt_train = BDTdata(df_train)
    bdt_test = BDTdata(df_test)

    sensitivity, error, bins = calc_sensitivity_with_error(df_test)
    print('Sensitivity for 2 Jet Dataset is: {:f}'.format(sensitivity)+' +/- {:f}'.format(error))
    
    # constructiong of BDT
    BDT = DT()
    
    # fitting the BDT
    BDT.fit(bdt_train[BDT_variables],bdt_train['generator'],sample_weight=bdt_train['EventWeight'])
    BDT_score = cross_val_score(BDT,bdt_test[BDT_variables],bdt_test['generator']).mean()
    print('BDT cross validation score: ',BDT_score)
    
    # recording bdt_decision for adversarial
    bdt_test['bdt_decision'] = BDT.decision_function(bdt_test[BDT_variables]).tolist()
    bdt_train['bdt_decision'] = BDT.decision_function(bdt_train[BDT_variables]).tolist()

    # labels for the adversarial NN
    z_train = to_categorical(bdt_train['generator'], num_classes=2)
    z_test = to_categorical(bdt_test['generator'], num_classes=2)

    # training adversial
#    adv_NN.compile(loss='binary_crossentropy', optimizer=adv_opt, metrics=['binary_accuracy'])
#    adv_NN.fit(bdt_test[variables],z_test,epochs=20,batch_size=32) # ignored event weight for
#                                                                   # BDT output classification
    adv = Model(input=[inputs], output=[adv_NN(inputs)])
    opt_adv = SGD(lr=1, momentum=0.5, decay=0.00001)
    adv.trainable = True; DNN.trainable = False
    adv.compile(loss='categorical_crossentropy', optimizer=opt_adv, metrics=['accuracy'])
    adv.fit(bdt_train[variables], z_train, batch_size=128, epochs=2)

    # constructing combined model
    opt_combined = SGD(lr=0.001, momentum=0.5, decay=0.00001)    
    combined_NN = Model(input=[inputs], output=[DNN(inputs), adv_NN(inputs)])
    DNN.trainable = True; adv_NN.trainable = False
    
    combined_NN.compile(loss={'DNN':'binary_crossentropy','adv_NN':'binary_crossentropy'}, optimizer=opt_combined, metrics=['accuracy'], loss_weights={'DNN': 1.0,'adv_NN': 1.0})
    
    # combined model training
    for i in range(1,50):
        print('Iteration: ',i)
        indices = np.random.permutation(len(bdt_train))[:32]
        
        # refitting the BDT
        BDT.fit(bdt_test[BDT_variables],bdt_test['generator'],
        sample_weight=bdt_test['EventWeight'])
        BDT_score = cross_val_score(BDT,bdt_test[BDT_variables],bdt_test['generator']).mean()
        print('BDT cross validation score: ',BDT_score)
        
        
        df_test['bdt_decision'] = BDT.decision_function(df_test[BDT_variables]).tolist()
        adv.train_on_batch(bdt_test[variables],z_test)
        
        combined_NN.train_on_batch(df_test[variables].iloc[indices], [df_test['Class'].iloc[indices],z_test[indices]],sample_weight = [df_test['training_weight'].iloc[indices],df_test['EventWeight'].iloc[indices]])        
        
        df_test['decision_value'] = DNN.predict(df_test[variables])[:,0]

        bdt_test = BDTdata(df_test)

        sensitivity, error, bins = calc_sensitivity_with_error(df_test)
        print('Sensitivity for 2 Jet Dataset is: {:f}'.format(sensitivity)+' +/- {:f}'.format(error))
        print()
        
    print('miraculously, this compiled')

if __name__ == '__main__':
    main()
