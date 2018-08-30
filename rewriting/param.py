# general parameters
variables_map = {
    '2': ['nTrackJetsOR', 'mBB','dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV','bdt_decision'],
    '3': ['nTrackJetsOR', 'mBB', 'dRBB', 'pTB1', 'pTB2', 'MET', 'dPhiVBB','dPhiLBmin', 'Mtop', 'dYWH', 'mTW', 'pTV', 'mBBJ', 'pTJ3']
}

# BDT_variables = ['mBB','decision_value', 'pTB1', 'dPhiVBB','mTW','Mtop','pTV']
BDT_variables = ['bdt_decision','decision_value']
nJets = '2'
variables = variables_map[nJets]
train = 'odd'
test  = 'even'

# NN parameters
lam = 1600
LR = 0.03      # learning rate
MOM = 0.5       # momentum
DEC = 0.0002   # decay rate
BATCH = 200     # batch size
EPOCH = 5     # number of epochs

# BDT parameters
variables = variables_map[nJets]
n_estimators = 200
max_depth = 4
learning_rate = 0.15

