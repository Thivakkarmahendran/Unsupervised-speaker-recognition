from tensorflow import keras
import numpy as np
import pyaudio
import time
import librosa
import os
from glob import glob

import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt


speakers = ["No Speaker"]
numClips = 15


for i in range(numClips):
    speakers.append(str(i+1))
    
    
model = keras.models.load_model('Trained Model')

def getFeatures(audio_Paths):
    
    data_X = []
    
    for path in audio_Paths:
        
        audioLength = librosa.get_duration(filename=path)
        
        if audioLength != 1.0:
            continue
        
        audioFeatureArray = []        
        y, sr = librosa.load(path)
        
        #mfcc
        mfccArray = librosa.feature.mfcc(y=y, sr=sr)
        audioFeatureArray.append(mfccArray.flatten())
        audioFeatureNumpyArray = np.array(audioFeatureArray)
        
        #zero_crossing_rate
        zeroCrossingArray = librosa.feature.zero_crossing_rate(y=y)
        np.append(audioFeatureNumpyArray, zeroCrossingArray.flatten())
        
        #spectral_rolloff
        spectralRollOffArray = librosa.feature.spectral_rolloff(y=y, sr=sr)
        np.append(audioFeatureNumpyArray, spectralRollOffArray.flatten())
        
        data_X.append(audioFeatureNumpyArray.flatten())
        
    return data_X


def findAllAudioFilePaths(speaker):
    audioFilesPaths = [y for x in os.walk("Dataset/Youtube Speech Dataset/Dataset/{}".format(speaker)) for y in glob(os.path.join(x[0], '*.wav'))]
    return audioFilesPaths



def runModel(speaker):
   
    audio_Paths = findAllAudioFilePaths(speaker)

    audioFeatureArray = getFeatures(audio_Paths) 
            
    validation_x = np.array(audioFeatureArray)

    predictScore = model.predict(validation_x)        
    classes = np.argmax(predictScore, axis = 1)

    speakerCount = dict()
    for classPredict in classes:
        speakerCount[classPredict] = speakerCount.get(classPredict, 0) + 1



    print("")
    print("-------------------------")
    print("Testing model for {}".format(speaker))
    
    for speaker in speakerCount: 
        print("Speaker: {} ----> Votes: {}".format(speakers[int(speaker)], speakerCount[speaker] )) 
    print("-------------------------")
    print("")



###############################

"""
runModel("No Speaker")
for speaker in speakers:
    runModel(speaker)
"""

#layers =  model.layers
#numLayers = len(layers)
#embeddings = model.layers[numLayers-1].get_weights()[0]
#print(embeddings)

model.pop()
model.pop()
 
 
audio_Paths = findAllAudioFilePaths("1")

audioFeatureArray = getFeatures(audio_Paths) 
            
validation_x = np.array(audioFeatureArray)

predictScore = model.predict(validation_x) 


thresh = 100
clusters = hcluster.fclusterdata(predictScore, thresh, criterion="distance")

print( len(set(clusters))  )
print(clusters)

"""
plt.scatter(np.transpose(predictScore), c=clusters)
plt.axis("equal")
title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
plt.title(title)
plt.show()
"""
 
 
"""    
def getdata(file):
    audio_Paths = findAllAudioFilePaths(file)

    onePath = []
    onePath.append(audio_Paths[0])

    audioFeatureArray = getFeatures(audio_Paths) 

    validation_x = np.array(audioFeatureArray)
    
    predictScore = model.predict(validation_x)        

    return predictScore


predictScor = []
predictScore = np.array(predictScor)
for speaker in speakers:
    np.append(predictScore, getdata(speaker).flatten())


print(predictScore.shape)
"""

"""
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(predictScore)
scaled_data = scaler.transform(predictScore)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

print(scaled_data.shape)
print(x_pca.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],cmap='rainbow')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()
"""