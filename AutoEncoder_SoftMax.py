

# coding: utf-8

# In[1]:


# Here are some imports that are used along this notebook
import math
import itertools
import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from collections import OrderedDict
import glob
get_ipython().magic('matplotlib inline')
gt0 = time()


# In[2]:


train20_nsl_kdd_dataset_path = "C:/Users/user/MSDS/Capstone/NSL_KDD-master/NSL_KDD-master/KDDTrain+_20Percent.txt"
train_nsl_kdd_dataset_path = "C:/Users/user/MSDS/Capstone/NSL_KDD-master/NSL_KDD-master/KDDTrain+.txt"
test_nsl_kdd_dataset_path = "C:/Users/user/MSDS/Capstone/NSL_KDD-master/NSL_KDD-master/KDDTest-21.txt"

col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","attrib43"])


nominal_inx = [1, 2, 3]
binary_inx = [6, 11, 13, 14, 20, 21]
numeric_inx = list(set(range(41)).difference(nominal_inx).difference(binary_inx))

nominal_cols = col_names[nominal_inx].tolist()
binary_cols = col_names[binary_inx].tolist()
numeric_cols = col_names[numeric_inx].tolist()

# Dictionary that contains mapping of various attacks to the four main categories
attack_dict = {
    'normal': 'normal',
    
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',
    
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',
    
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',    
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}


# In[3]:


def _label2(x):
    if x['labels'] == 'normal':
        return 'normal'
    else:
        return 'attack'

def returnvalue(x):
    return attack_dict.get(x['labels'])


# In[4]:


df_kdd_dataset_train = pd.read_csv(train20_nsl_kdd_dataset_path, index_col=None, header=0, names=col_names)
df_kdd_dataset_train['label2'] = df_kdd_dataset_train.apply(_label2,axis=1)
df_kdd_dataset_train['label3'] = df_kdd_dataset_train.apply(returnvalue,axis=1)


# In[5]:


df_kdd_dataset_test = pd.read_csv(test_nsl_kdd_dataset_path, index_col=None, header=0, names=col_names)
df_kdd_dataset_test['label2'] = df_kdd_dataset_test.apply(_label2,axis=1)
df_kdd_dataset_test['label3'] = df_kdd_dataset_test.apply(returnvalue,axis=1)


# In[6]:


print(df_kdd_dataset_train.shape)
print(df_kdd_dataset_test.shape)


# In[7]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
df_kdd_dataset_train_tranformed = df_kdd_dataset_train.apply(le.fit_transform)
df_kdd_dataset_train_tranformed.head()
df_kdd_dataset_train_saved = df_kdd_dataset_train_tranformed


# In[8]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# 2/3. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
df_kdd_dataset_test_tranformed = df_kdd_dataset_test.apply(le.fit_transform)
df_kdd_dataset_test_tranformed.head()
df_kdd_dataset_test_saved = df_kdd_dataset_test_tranformed


# In[9]:


print(df_kdd_dataset_train_tranformed.shape)
print(df_kdd_dataset_test_tranformed.shape)


# In[10]:


def one_hot_encode(x, n_classes):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
     """
    return np.eye(n_classes)[x]


# In[11]:


y_test = df_kdd_dataset_test['label2']
le.fit(y_test)
y_test = le.transform(y_test)
print(y_test)
y_test = one_hot_encode(y_test,2)
print(y_test)


# In[12]:


y = df_kdd_dataset_train['label2']
print(y[1])
le.fit(y)
y = le.transform(y)
print(y)
y = one_hot_encode(y,2)
print(y)


# In[13]:


df_kdd_dataset_train_tranformed.drop('labels', axis=1, inplace=True)
df_kdd_dataset_train_tranformed.drop('attrib43',axis=1,inplace=True)
df_kdd_dataset_train_tranformed.drop('label2',axis=1,inplace=True)
df_kdd_dataset_train_tranformed.drop('label3',axis=1,inplace=True)

enc = preprocessing.OneHotEncoder(categorical_features=[1, 2, 3])

# 2. FIT
#enc.fit(df_kdd_dataset_train_tranformed)

# 3. Transform
#onehotlabels = enc.transform(df_kdd_dataset_train_tranformed).toarray()
#print(onehotlabels.shape)

df_kdd_dataset_test_tranformed.drop('labels', axis=1, inplace=True)
df_kdd_dataset_test_tranformed.drop('attrib43',axis=1,inplace=True)
df_kdd_dataset_test_tranformed.drop('label2',axis=1,inplace=True)
df_kdd_dataset_test_tranformed.drop('label3',axis=1,inplace=True)

#enc = preprocessing.OneHotEncoder(categorical_features=[1, 2, 3])
# 2. FIT
print(df_kdd_dataset_train_tranformed.shape)
print(df_kdd_dataset_test_tranformed.shape)
df_kdd_entire_set = np.vstack((df_kdd_dataset_train_tranformed, df_kdd_dataset_test_tranformed))
enc.fit(df_kdd_entire_set)

# 3. Transform
onehotlabels_train_test = enc.transform(df_kdd_entire_set).toarray()
print(onehotlabels_train_test.shape)


# In[14]:


onehotlabels_train = onehotlabels_train_test[0:25191]
onehotlabels_test = onehotlabels_train_test[25191:37040]
print(onehotlabels_train.shape)
print(onehotlabels_test.shape)


# In[15]:


import csv
csvfile = "C:/Users/user/MSDS/Capstone/NSL_KDD-master/NSL_KDD-master/output.txt"

#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in onehotlabels_train:
        writer.writerow([val])   


# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_train_transform = scaler.fit_transform(onehotlabels_train)
df_test_transform = scaler.fit_transform(onehotlabels_test)


# In[17]:


print(df_train_transform.shape[0])


# In[18]:


print(y.shape)
df_train_transform_with_label=np.c_[df_train_transform,y]
df_test_transform_with_label=np.c_[df_test_transform,y_test]
print(df_train_transform_with_label.shape)
print(df_test_transform_with_label.shape)


# In[19]:


def get_next_batch(i,batch_size,dataset):
    start = i*batch_size
    end = (i+1)*batch_size
    return dataset[start:end]


# In[20]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Parameters
learning_rate = 0.01
training_epochs = 100
batch_size = 55
display_step = 1
examples_to_show = 10

# Network Parameters
n_hidden_1 = 55 # 1st layer num features
#n_hidden_2 = 25 # 2nd layer num features
n_input = 118 # Number of features

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])
autoencoder_op = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    #'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    #'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    #'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    #'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
    print('encoder')
    return layer_1


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
    print('decoder')
    return layer_1

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)
    
model_path = "C:\\users\\user\MSDS\\Capstone\\Models"

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(df_train_transform.shape[0]/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = get_next_batch(i,batch_size,df_train_transform)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs}) 
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))            
    print("Optimization Finished!")
    #c2,dec_op2 = sess.run([cost,decoder_op], feed_dict={X: df_train_transform}) 
    save_path = saver.save(sess,model_path)
    # model has been trained pass the training data with labels
    cost_train,encoder_op_train_wo_label = sess.run([cost,encoder_op], feed_dict={X: df_train_transform}) 
    print(cost_train)
    cost_test,encoder_op_test_wo_label = sess.run([cost,encoder_op], feed_dict={X: df_test_transform}) 
    print(cost_test)


# In[ ]:





# In[ ]:


print(encoder_op_train_wo_label.shape)
print(encoder_op_test_wo_label.shape)


# In[21]:


def get_next_batch_label(i,batch_size,dataset):
    start = i*batch_size
    end = (i+1)*batch_size
    return dataset[start:end]


# In[ ]:





# In[25]:


def get_next_batch_sftmax(i,batch_size):
    start = i*batch_size
    end = (i+1)*batch_size
    return encoder_op_train_wo_label[start:end,0:55],df_train_transform_with_label[start:end,118:120]


# In[26]:


## Now we build the softmax regressor

# Parameters
learning_rate = 0.35
training_epochs = 1000
batch_size = 55
display_step = 10

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 55]) # 118 features
y = tf.placeholder(tf.float32, [None, 2]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([55, 2]))
b = tf.Variable(tf.zeros([2]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(df_train_transform_with_label.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = get_next_batch_sftmax(i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    print("accuracy", sess.run(accuracy, feed_dict={x: encoder_op_train_wo_label[:,0:55], y: df_train_transform_with_label[:,118:120]}))
    #prediction=tf.argmax(y,1)
    #print ("predictions", prediction.eval(feed_dict={x: df_train_transform_with_label[0:118]}, session=sess))
    print("accuracy", sess.run(accuracy, feed_dict={x: encoder_op_test_wo_label[:,0:55], y: df_test_transform_with_label[:,118:120]}))
    #print("accuracy", sess.run(accuracy, feed_dict={x: df_train_transform_with_label[:,0:118], y: df_train_transform_with_label[:,118:120]}))
    


# In[ ]:


print(dec_op_test_label[1])


# In[27]:


def get_next_batch_sftmax_wostl(i,batch_size):
    start = i*batch_size
    end = (i+1)*batch_size
    return df_train_transform_with_label[start:end,0:118],df_train_transform_with_label[start:end,118:120]


# In[28]:


## Now we build the softmax regressor

# Parameters
learning_rate = 0.3
training_epochs = 1000
batch_size = 55
display_step = 10

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 118]) # 118 features
y = tf.placeholder(tf.float32, [None, 2]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([118, 2]))
b = tf.Variable(tf.zeros([2]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(df_train_transform_with_label.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = get_next_batch_sftmax_wostl(i,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    print("accuracy", sess.run(accuracy, feed_dict={x: df_train_transform_with_label[:,0:118], y: df_train_transform_with_label[:,118:120]}))
    #prediction=tf.argmax(y,1)
    #print ("predictions", prediction.eval(feed_dict={x: df_train_transform_with_label[0:118]}, session=sess))
    print("accuracy", sess.run(accuracy, feed_dict={x: df_test_transform_with_label[:,0:118], y: df_test_transform_with_label[:,118:120]}))
    #print("accuracy", sess.run(accuracy, feed_dict={x: df_train_transform_with_label[:,0:118], y: df_train_transform_with_label[:,118:120]}))
    


# In[ ]:








