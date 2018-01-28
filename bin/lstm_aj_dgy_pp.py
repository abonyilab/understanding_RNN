# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:05:13 2017

@author: resea
"""

# 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import GRU
from keras.layers import Flatten, Dropout, Dense, Embedding
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from keras.callbacks import BaseLogger
#from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from matplotlib.colors import BoundaryNorm
import pandas as pd
import numpy as np
import random
from time import time
import time
from sklearn.manifold import TSNE
from sklearn import decomposition
import itertools
from scipy.cluster.hierarchy import linkage
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import dendrogram
from pcaplot import hyperellipsoid

#%% FUNCTION FOR TRAINING THE MODEL 

def lstmtrain(X_train,Y_train,X_test,Y_test):
    
    
    # Compile and train different models while measuring performance.
    results = []
    max_features = X.max()+1
    
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim,mask_zero=True)) # , input_length=SEQLEN 
    
    #if you would lke to add an additional convolutional layer
    #model.add(Conv1D(filters=embedding_dim, kernel_size=3, padding='same', activation='relu'))
    #model.add(MaxPooling1D()) #pool_size=Full)
    
    #model.add(Dropout(0.05))
    
    model.add(LSTM(nUNIT)) #   GRU dropout=0.01, recurrent_dropout=0.01,

 #  model.add(LSTM(nUNIT,  implementation=1)) #dropout=0.01, recurrent_dropout=0.01,

    model.add(Dense(Y_train.shape[1], activation='softmax')) #
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    #tensorboard = TensorBoard(
    #    log_dir="./tensorboard/",
    #    write_images=True, 
    #    histogram_freq=1,
    #    embeddings_freq=250, 
    #    embeddings_metadata='map.tsv'
    #)
    #tensorboard --logdir=C:\Users\resea\Dropbox\deep_chem\Modellek\FINAL
    #http://fmt_otka:6006
    print(model.summary())
    
    start_time = time.time()
    history = model.fit( X_train,  Y_train,  validation_data=(X_test, Y_test), batch_size=batch_size, epochs=epochs)
    average_time_per_epoch = (time.time() - start_time) / epochs
    results.append((history, average_time_per_epoch))
    
        
    return(model,results)

#%% EZT MOST NEM HASZNALJUK 
def stackedtrain(X_train,Y_train,X_test,Y_test):


    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(Embedding(max_features, embedding_dim, input_length=SEQLEN,mask_zero=True))
    model.add(LSTM(nUNIT, return_sequences=True,
                   input_shape=(SEQLEN, embedding_dim)))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(nUNIT, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(nUNIT))  # return a single vector of dimension 32
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', #rmsprop
                  metrics=['accuracy'])
    
    print(model.summary())
    
    start_time = time.time()
    history = model.fit( X_train,  Y_train,  validation_data=(X_test, Y_test), batch_size=X.shape[0], epochs=epochs)
    average_time_per_epoch = (time.time() - start_time) / epochs
    results.append((history, average_time_per_epoch))
    

    return(model,results)

#%% FUNCTION FOR THE X-FOLD CROSS VALDATION 

def crossval(X,Y,n_fold):
    seed = 42
    np.random.seed(seed)
    K_fold = StratifiedKFold(n_fold, shuffle=False, random_state=seed)
    cv_scores=[]
    cv_models = []
    for train, test in K_fold.split(X,y_label): #numerikus label kell 
        (model,results)=lstmtrain(X[train],  Y[train], X[test],  Y[test])
        scores = model.evaluate(X[test], Y[test], verbose=0)
        print("Model Accuracy: %.2f%%" % (scores[1]*100))
        cv_scores.append(scores[1] * 100)
        cv_models.append(model)
    return(cv_models, cv_scores)

#%% FUNCTION FOR LOADING THE DATA

def loadandprepare(filename,withR,nEVENT):
    
    print('Loading data...')
    X_data = pd.read_excel(filename, header=None, dtype=float)
    y_data = pd.read_excel('Fault_0822.xlsx', header=None, dtype=int)
    X = X_data.as_matrix()
    Y = y_data.as_matrix()
    
    if revFlag:
        X=np.flip(X,1)

    if withR:
        SEQLEN=nEVENT*2-1
    else:
        SEQLEN=nEVENT
    
    if not(withR):
        X=X[:,::2] 
        
# ANALYSIS OF THE INFORMATION CONTENT 
#    nz=np.count_nonzero((X+5),axis=0)
#
#    nx=range(1,len(nz)+1)
#    plt.bar(nx,nz)
#    plt.xlabel('$T$')
#    plt.ylabel('$ \#$ of seq with length $T$')
#    if (type(SEQLEN)==list):
#        cutTper=5 #% hany szazalek ferjen
#        
#        X.shape[0]*cutTper/100
#        
#        SEQLEN=max((np.where(nz>220)[0]))
#        
#        if withR:
#            SEQLEN=SEQLEN+1
    # SEQUENCE generation 
    if revFlag:
        X = sequence.pad_sequences(X, maxlen=SEQLEN, dtype='int32', padding='pre', truncating='pre', value=-5.)
    else:
        X = sequence.pad_sequences(X, maxlen=SEQLEN, dtype='int32', padding='post', truncating='post', value=-5.)
        

    shape_X = X.shape    
    shape_Y = Y.shape    
    
    # CODING for ONE- HOT ENCODING (and decoding transf)
    XEncoder = preprocessing.LabelEncoder()
    XEncoder.fit(X.flatten()) 
    X_encoded = XEncoder.transform(X.flatten()) #inverse_transform-al majd vissza
    X=X_encoded.reshape((shape_X))
    
    YEncoder = preprocessing.LabelEncoder()
    YEncoder.fit(Y.flatten()) 
    Y_encoded = YEncoder.fit_transform(Y)
    y_label=Y_encoded
    Y=Y_encoded.reshape((shape_Y)) 
    
    Y = np_utils.to_categorical(Y)
    
    
    return(X,Y,SEQLEN,XEncoder,YEncoder,y_label)

#%%  MAIN PARAMETERS


filenames=['Sequences_simple_alarm_0918.xlsx', 'Sequences_alarm_warning_0918.xlsx', 'Sequences_quantized_0918.xlsx']
dataname=['A','B','C']




n_fold = 7
epochs = 500 # useful for logging and periodic evaluation - ennyinként fogja naplózni és kiértékelni az eredményeket
batch_size = 512 # set of inputs, a batch generally approximates the distribution of the input data better than a single input.


#%% ANALYZIS OF the effect of the training data

#nEVENT=5
#revFlag=False 
#nUNIT=11
#embedding_dim = 4
## withR, tehat a szekvencia hossza R-ekkel 2*5-1 = 9, 11 LSTM unitba lep be! 
#
#withR=True
#
#acc_data1 = []
#for di in range(3):
#    filename=filenames[di]
#    (X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)
#    (m1,cv_scores)=crossval(X,Y,n_fold)
#    acc_data1.append(cv_scores)
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#bp = ax.boxplot(acc_data1)
##plt.ylim((80,100))
##plt.xlabel('With temporal relations', fontsize=5)
#plt.ylabel('Correct Classification Rate (%)', fontsize=14)
#ax.set_xticklabels(['A/1','B/1','C/1'], fontsize=12)
##ax.set_yticklabels(fontsize=14)
#plt.xlabel('Datasets', fontsize=14)
#plt.yticks(fontsize=12)
#plt.savefig('acc_data_ccr.png', dpi=300)
#plt.show()
#
#ad1 = pd.DataFrame(acc_data1)
#writer = pd.ExcelWriter('acc_data.xlsx')
#ad1.to_excel(writer,'with R')

#%% ANALYZIS OF the effect of the number of events and the temporal relationship

#nUNIT=11
#revFlag=False 
#
#di=1 #0-2
#filename=filenames[di]
#
#
#embedding_dim=4
#
#VnEVENT=[2, 3, 4, 5, 6]
#
#casename=['with R', 'without R']
#case=[True, False]
#
#
#withR=True
#acc_R = []
#for di in range(len(VnEVENT)):
#    nEVENT=VnEVENT[di]
#    (X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)
#    (m2,cv_scores)=crossval(X,Y,n_fold)
#    acc_R.append(cv_scores)
#
#withR=False
#acc_Rwo = []
#for di in range(len(VnEVENT)):
#    nEVENT=VnEVENT[di]
#    (X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)
#    (m2,cv_scores)=crossval(X,Y,n_fold)
#    acc_Rwo.append(cv_scores)
#
#fig = plt.figure() 
#ax = fig.add_subplot(211)
#bp = ax.boxplot(acc_R)
#plt.ylabel('Corr. Class. Rate (%)')
#plt.title('With R',Fontsize=8)
##ax.set_xticklabels(VnEVENT)
#ax.set_xticklabels([])
#plt.ylim((87.5,95))
#
#ax = fig.add_subplot(212)
#plt.title('Without R',Fontsize=8)
#bp = ax.boxplot(acc_Rwo)
#plt.ylabel('Corr. Class. Rate (%)')
#ax.set_xticklabels(VnEVENT)
#plt.xlabel('# of Events')
#plt.ylim((87.5,95))
#
#plt.savefig('R_type2.png', dpi=300)
#plt.show()
##
#adR1 = pd.DataFrame(acc_R)
#adR2 = pd.DataFrame(acc_Rwo)
#
#writer = pd.ExcelWriter('acc_ER2.xlsx')
#adR1.to_excel(writer,'R')
#adR2.to_excel(writer,'Rwo')
#writer.save()    


#%% ANALYZIS OF the effect of lstm length

# Egyelőre legyen 4, valahogy meg kell magyarázni, mert így jól látszik a tendencia!!!
#embedding_dim=4
#nEVENT=4
#
#withR=False
#
###
#di=1 #○ez a második
#filename=filenames[di]
#(X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)
#
#
#lstm_length = [4,5,11,17,21]  #nUNIT ,SEQLEN+15
#
#acc_nUNIT = []
#for nUNIT in lstm_length:
#    (m4,cv_scores) = crossval(X,Y,n_fold)
#    acc_nUNIT.append(cv_scores)
#
#fig = plt.figure() 
#ax = fig.add_subplot(111)
#bp = ax.boxplot(acc_nUNIT)
##plt.title('Effect of LSTM unit length on accuracy')
#plt.ylabel('Correct Classification Rate (%)', fontsize=14)
#plt.xlabel('LSTM length', fontsize=14)
#plt.yticks(fontsize=12)
#ax.set_xticklabels(lstm_length, fontsize=12)
#plt.savefig('nUnit_crossval.png', dpi=300)
#plt.show() 
#
#adnU = pd.DataFrame(acc_nUNIT)
#writer = pd.ExcelWriter('acc_nU.xlsx')
#adnU .to_excel(writer,'nU')
#writer.save()    


#%% ANALYZIS OF the effect of embedding layer

#nEVENT=4
#nUNIT=17
#
#di=1 #○ez a második
#filename=filenames[di]
#
#withR=False
#
#(X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,SEQLEN)
#
#dims = [2,3,4,6]  #embedding_dim
#
#acc_emb = []
#for embedding_dim in dims:
#    (m3,cv_scores) = crossval(X,Y,n_fold)
#    acc_emb.append(cv_scores)
#    
#fig = plt.figure()
#ax = fig.add_subplot(111)
#bp = ax.boxplot(acc_emb)
##plt.title('Effect of embedding dimension on accuracy', fontsize=8)
#plt.ylabel('Correct Classification Rate (%)', fontsize=14)
#plt.xlabel('Embedding Dimension', fontsize=14)
#plt.yticks(fontsize=12)
#ax.set_xticklabels(dims, fontsize=12)
#plt.savefig('nEsmb_crossval_dgy.png', dpi=300)
#plt.show()
##
#adEmb = pd.DataFrame(acc_emb)
#
#writer = pd.ExcelWriter('acc_emb.xlsx')
#adEmb.to_excel(writer,'Emb')
#writer.save()    



#%% confusion matrix


seed = 42
np.random.seed(seed)
  
di=1 #○ez a második
filename=filenames[di]

revFlag=False
nEVENT=4
nUNIT=17
embedding_dim = 4
withR=False


(X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT)


(model,results)=lstmtrain(X,Y,X,Y)
scores = model.evaluate(X, Y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

predictions=model.predict(X, batch_size=batch_size)

Confusion=confusion_matrix(np.argmax(Y,axis=1), np.argmax(predictions,axis=1))

#categories = list(set(YEncoder.inverse_transform(y_label)))
categories = list(set(y_label+1))

tick_marks = np.arange(len(categories))
plt.figure(figsize=(11,11))
plt.imshow(Confusion, cmap=plt.cm.Blues)
plt.xticks(tick_marks,categories)
plt.yticks(tick_marks,categories)
plt.title('')
plt.xlabel('Predicted class', fontsize=20)
plt.ylabel('True class', fontsize=20)
for i, j in itertools.product(range(Confusion.shape[0]), range(Confusion.shape[1])):
    c='black'
    if Confusion[i, j]/200 > .50:
        c='white'
    plt.text(j, i, format(Confusion[i, j]/200*100, '.2f')+'%' , horizontalalignment="center",color=c)

plt.savefig('Confmatabszolut.png', dpi=300)
plt.show()


#%% PCA of the embedding layer
## (X,Y,SEQLEN,XEncoder,YEncoder,y_label)=loadandprepare(filename,withR,nEVENT) # kell az XEncoder miatt!!!!!!!!!!
# extract embedding layer weights
#from keras.models import load_model
#model = load_model('Model.h5')
#model.load_weights('Model_weights.h5')

#We=model.layers[0].get_weights()
#We=We[0]
#
## get event names
#labels=XEncoder.inverse_transform(range(We.shape[0]))
#
## create pca and transport
#pca = decomposition.PCA()
#pca.fit(We)
#Wep = pca.transform(We)
#Wep_variance = pca.explained_variance_ratio_ # get variance ratio
#Wep_covariance = pca.get_covariance()
#_, _, _, _, _ = hyperellipsoid(Wep[:,:2], covar=Wep_covariance[:2,:2], variance=Wep_variance,scatter_labels=labels, savefig='Embed_PCA_Elipse.png')
#
#
##%% Dendrogram of the embedidng weighs
#We=model.layers[0].get_weights()
#We=We[0][1::];
#
#D_We = pairwise_distances(We)
#
#DD=D_We/D_We.max()
#Z = linkage(DD, method='average')
#dendrogram(Z, labels=labels)
#plt.xlabel('Event ID')
#plt.ylabel('Distance')
#
#plt.savefig('den_pca_emb_ID_Distance.png', dpi=300)
#
##%% PCA of Output layer
#def get_activations(model, layer, X):
#    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
#    activations = get_activations([X,0])
#    return activations
#
#
#LACT = get_activations(model,1,X)
#Lpca = decomposition.PCA()
#Lpca.fit(LACT[0])
#Lp = Lpca.transform(LACT[0])
#variance = Lpca.explained_variance_ratio_
#
#plt.figure(figsize=(11,11))
##plt.title('PCA of LSTM activation')
#xplot_label = '$t_1$ ({}% of variance)'.format(int(variance[0]*100))
#yplot_label = '$t_2$ ({}% of variance)'.format(int(variance[1]*100))
#plt.xlabel(xplot_label, fontsize=20)
#plt.ylabel(yplot_label, fontsize=20)
#yhat=np.argmax(predictions,axis=1)+1
#bounds = np.linspace(0,11,12)
#cmap = plt.get_cmap('jet', 11)
#norm = BoundaryNorm(bounds, cmap.N)
#plt.scatter(Lp[:,0], Lp[:,1], cmap=cmap, c=yhat, norm=norm)
#cbar = plt.colorbar(ticks=bounds, boundaries=bounds)
#cbar.ax.get_yaxis().set_ticks([])
#cats = list(set(YEncoder.inverse_transform(y_label)))
#for cat in range(11):
#    cbar.ax.text(1.5,np.linspace(0,1,12)[cat]+0.045,cat+1,ha='center', va='center')
#plt.savefig('PCA_act_label.png', dpi=300)
#plt.show()
#
#plt.bar(np.arange(len(variance)),variance*100)
#plt.plot(variance.cumsum()*100)
#plt.xticks(np.arange(len(variance)),np.arange(1,len(variance)+1))
#plt.xlabel('No. of eigenvalues')
#plt.ylabel('Cumulated variance percentage (%)')
#plt.savefig('PCA_var_act.png', dpi=300)
#plt.show()
#
#from mpl_toolkits.mplot3d import Axes3D
#fig = plt.figure()
#ax = fig.add_axes([0, 0, 1, 1], projection='3d')
#ax.view_init(20, 30)
#ax.scatter(Lp[:, 0], Lp[:, 1], Lp[:, 2], '.', c=yhat, cmap=cmap)
#plt.savefig('3d_lstm_activation_pca.png',dpi=600)
#
#import pickle
#pickle.dump(fig, open('3d_lstm_pca.fig.pickle', 'wb'))
#
#plt.show()
#
