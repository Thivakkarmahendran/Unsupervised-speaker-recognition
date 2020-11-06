#imports
from tensorflow import keras
import numpy as np
import pyaudio
import time
import librosa



class AudioHandler(object):
    def __init__(self):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 1024*21+550
        self.p = None
        self.stream = None
        
        self.model = keras.models.load_model('Trained Model')
        self.speakers = ["Obama", "Hillary", "Ivanka", "Trump", "No Speaker", "Modi", "Xi-Jinping", "Chadwick-Boseman"]

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()
        
        
    def getFeatures(self, array):
        audioFeatureArray = []        
        
        #mfcc
        mfccArray = librosa.feature.mfcc(array)
        audioFeatureArray.append(mfccArray.flatten())
        audioFeatureNumpyArray = np.array(audioFeatureArray)
        
        #zero_crossing_rate
        zeroCrossingArray = librosa.feature.zero_crossing_rate(array)
        np.append(audioFeatureNumpyArray, zeroCrossingArray.flatten())
        
        #spectral_rolloff
        spectralRollOffArray = librosa.feature.spectral_rolloff(array)
        np.append(audioFeatureNumpyArray, spectralRollOffArray.flatten())
        
        #pitch and magnitude
        pitchArray, magnitudeArray = librosa.piptrack(array)
        np.append(audioFeatureNumpyArray, pitchArray.flatten())
        np.append(audioFeatureNumpyArray, magnitudeArray.flatten())
        
        audioFeatureNumpyArray = audioFeatureNumpyArray.flatten()
        
        return audioFeatureNumpyArray

           

    def callback(self, in_data, frame_count, time_info, flag):
        
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        
        samples_to_predict = []
        audioFeatureArray = self.getFeatures(numpy_array) 
        
        samples_to_predict.append(audioFeatureArray)
        validation_x = np.asarray(samples_to_predict)
        
        predictScore = self.model.predict(validation_x)
        
        classes = np.argmax(predictScore, axis = 1)
        print(self.speakers[classes[0]])
        
        
        return None, pyaudio.paContinue


    def mainloop(self):
        while (self.stream.is_active()):
            time.sleep(1.0)



audio = AudioHandler()
audio.start()     # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()

