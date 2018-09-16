### optimising the classifier NN wrt number of nodes, layers, epoches, etc

# importing general modules
from importing import *

# setting seeds
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

variables = ['nTrackJetsOR', 'mBB','dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV']

nodes = [12,24,36,48,60,100,150,200,250,300,400,500,750,1000]
#nodes = 400
layerNum = 5
epochs = 50
batch = 64

# compiles and fits the neural net
def oneNN(df_train,df_test, variables, opt, nodes, layers, epochs, batch): 

    # constructing classifier model based on nodes and number of layers
    inputs = Input(shape=(len(variables),))
    L = Dense(nodes, activation = "linear")(inputs)
    for i in range (0,layers): 
        L = Dense(nodes, activation = "tanh")(L)
    L = Dense(1, activation="sigmoid", name='classifier_output')(L)
    L = Model(input=[inputs], output=[L])
    
    # compiling and fitting
    L.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    history = L.fit(np.array(df_train[variables]), np.array(df_train['Class']),sample_weight=np.array(df_train['training_weight']), epochs=epochs, batch_size=batch,shuffle=True)
    scores = L.predict(df_test[variables])[:,0]
    df_test['decision_value'] = ((scores-0.5)*2)
    sensitivity, error, bins = calc_sensitivity_with_error(df_test)
    
    return sensitivity, error, history

# plotting the history object
def plotHistory(history, nodes, layerNum):
    epochs = np.arange(1,len(history.history['binary_accuracy'])+1,1)
    
    plt.figure(1)
    plt.plot(epochs,history.history['binary_accuracy'],'b')

    plt.xlim(0.5,np.max(epochs)+0.5)
    title = ""
    plt.title('model accuracy ('+str(layerNum)+'layers,'+str(nodes)+'nodes)')
    plt.ylabel('binary_accuracy')
    plt.xlabel('epoch')
    plt.savefig(str(nodes)+" acc")
    plt.show(block=True)

    plt.plot(epochs,history.history['loss'],'b')
    plt.title('model loss ('+str(layerNum)+'layers,'+str(nodes)+'nodes)')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(str(nodes)+" loss")
    plt.show(block=True)

def main():    
    # Getting data from csv file
    df_train = pd.read_csv ('../results_v18/group_even.csv', index_col=0)
    df_test = pd.read_csv('../results_v18/group_odd.csv', index_col=0)
    
    df_train[variables] = scale(df_train[variables])
    df_test[variables] = scale(df_test[variables])
                
    # initialising the optimiser
#    opt = SGD(lr=0.001, momentum=0.5, decay=0.00001)
    opt = SGD(lr=0.03, momentum=0.5, decay=0.0002)

    all_sens = []
    all_err = []
    all_history = []
    for node in nodes:
        print('======================')
        print('Number of nodes: ',node)
        print('======================')
        sensitivity, error, history = oneNN(df_train,df_test, variables, opt, node, layerNum, epochs, batch)
        all_sens.append([node,sensitivity])
        all_err.append([node,error])
        all_history.append([node,history])
        print('Sensitivity: ', sensitivity)
    print('Sensitivity: ',all_sens)
    print('Error: ',all_err)
    
    for history_node in all_history:
        node = history_node[0]
        history = history_node[1]
        plotHistory(history, node, layerNum)
    
    plt.errorbar(np.array(all_sens)[:,0],np.array(all_sens)[:,1],yerr=np.array(all_err)[:,1],marker='.',c='b',ls='None')
    plt.xlabel('nodes')
    plt.ylabel('sensitivity')
    plt.title('Sensitivity for '+str(layerNum)+' layers')
    plt.savefig('sensitivity')
    plt.show(block=True)
    
if __name__ == '__main__':
    main()
