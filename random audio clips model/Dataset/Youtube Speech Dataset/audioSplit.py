from pydub import AudioSegment
from pydub.utils import make_chunks
import os
from glob import glob



def findAllAudioFilePaths():
    audioFilesPaths = [y for x in os.walk("Orginal") for y in glob(os.path.join(x[0], '*.mp3'))]
    return audioFilesPaths

def splitAudio(audioPaths, splitLength):
    
    for path in audioPaths:
        speakerName = path.split("/")[1]
        speakerIndex = path.split("/")[2].split(" ")[1].split(".")[0]
        
        
        audio = AudioSegment.from_file(path, "mp3")
        chunks = make_chunks(audio, splitLength)
        
        for i, chunk in enumerate(chunks):
            
            chunk_name = "Dataset/{}/{}-{} {}.wav".format(speakerName, speakerName, speakerIndex, i)
            
            #create directory if not found
            if not os.path.exists("Dataset/{}".format(speakerName)):
                os.makedirs("Dataset/{}".format(speakerName))
            
            if os.path.exists(chunk_name):
                print("** Already Exported {}-{} **".format(speakerName, speakerIndex))
                break
            
            chunk.export(chunk_name, format="wav")
        
        print ("Exported", speakerName)
        
        
        
################################################

chunk_length_ms = 1000 #1 second 

paths = findAllAudioFilePaths()
splitAudio(paths, chunk_length_ms)