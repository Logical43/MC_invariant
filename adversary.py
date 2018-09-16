### trying to separate MC generators by Tensorflow NN
from importing import *
from param import *
from functions import *

from sklearn.neural_network import MLPClassifier
import tensorflow as tf

variables = ['mBB','decision_value', 'pTB1', 'dPhiVBB','mTW','Mtop','pTV']

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Getting data from csv file and selecting two MC generator
df_even = pd.read_csv('../DNN/adversary_even_log.csv', index_col=0)
df_even = df_even.loc[df_even['Class']==0]
df_even = df_even.loc[df_even['generator']!=2]
df_even = df_even.loc[df_even['nTags']==2]

df_odd = pd.read_csv('../DNN/adversary_odd_log.csv', index_col=0)
df_odd = df_odd.loc[df_odd['Class']==0]
df_odd = df_odd.loc[df_odd['generator']!=2]
df_odd = df_odd.loc[df_odd['nTags']==2]

df_odd[variables] = scale(df_odd[variables])
df_even[variables] = scale(df_even[variables])

# changing the label to matrix
train_label = to_categorical(df_even['generator'], num_classes=2)
test_label = to_categorical(df_odd['generator'], num_classes=2)

n_input = df_even.shape[1]
n_classes = train_label.shape[1]

n_hidden_1 = 24

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

training_epochs = 5000
display_step = 1000
batch_size = 32

def main():    
    
    # placeholders to initialise model
    x = tf.placeholder("float", [n_input, None])
    y = tf.placeholder("float", [33, None])
    
    # constructing and compiling the model
    predictions = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(df_even[variables]) / batch_size)
            x_batches = np.array_split(df_even[variables], total_batch)
            y_batches = np.array_split(df_even['generator'], total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimizer, cost], 
                                feed_dict={
                                    x: batch_x, 
                                    y: batch_y, 
                                })
                avg_cost += c / total_batch
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: df_odd[variables], y: df_odd['generator'], keep_prob: 1.0}))

### tensorflow with keras ###
"""    
    # model construction
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(len(variables),)),
      tf.keras.layers.Dense(400, activation=tf.nn.tanh),
      tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
    ])
    
    # compiling and fitting
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(df_even[variables], train_label, sample_weight = df_even['adversary_weights'], epochs=20)
    
    # printing loss and accuracy
    loss, acc = model.evaluate(df_odd[variables], test_label)
    print('Loss: ',loss)
    print('Accuracy: ',acc)
    
    # making predictions and evaluating predictions
    scores = model.predict(df_odd[variables])
    print(scores)
    df_odd['gen_0'] = np.array(scores)[:,0]
    df_odd['gen_1'] = np.array(scores)[:,1]
    
    df_gen0 = df_odd.loc[df_odd['generator'] == 0]
    df_gen1 = df_odd.loc[df_odd['generator'] == 1]

    gen0_scores = df_gen0['gen_0'].tolist()
    gen0_mean = sum(gen0_scores)/len(gen0_scores)
    gen1_scores = df_gen1['gen_1'].tolist()
    gen1_mean = sum(gen1_scores)/len(gen1_scores)

    print("generator 1 guess: ",gen0_mean)
    print("generator 2 guess: ",gen1_mean)
    plt.hist(df_gen0['gen_0'],bins = 20, color = 'r', alpha=0.5, density=True, label ='pythiapw')
    plt.hist(df_gen1['gen_1'],bins = 20, color = 'b', alpha=0.5, density=True, label ='aMCP8')
    plt.legend()
    plt.xlabel('NN true positive rate')
    plt.ylabel('Number of events')
    plt.show(block=True)
"""    
if __name__ == '__main__':
    main()
