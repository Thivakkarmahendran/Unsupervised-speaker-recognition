from tensorflow import keras
import numpy as np
import pyaudio
import time
import librosa
import os
from glob import glob


speakers = ["Obama", "Hillary", "Ivanka", "Trump", "No Speaker", "Modi", "Xi-Jinping", "Chadwick-Boseman"]
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
for speaker in speakers:
    runModel(speaker)
"""

"""
layers =  model.layers
numLayers = len(layers)

embeddings = model.layers[numLayers-1].get_weights()[0]

print(embeddings.shape)
"""

runModel("No Speaker")