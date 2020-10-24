# -- coding: utf-8 --
import librosa
import os
import time
import pyaudio
import multiprocessing
from multiprocessing import Process,Pipe
import struct as st
import wave

from Analyser import analyser

def listen(channels,sample_rate,chunk,writer):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk  #pyaudio内置缓存区大小
                    )

    # Determine the timestamp of the start of the response interval
    print('* Start Recording *')
    stream.start_stream()
    # Record audio until timeout
    while True:
        # Record data audio data
        data = stream.read(chunk)
        writer.send(data)

    # Close the audio recording stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('* End Recording * ')

def draw(reader):
    pass

def main(filename=None):
    ser=analyser('cache/1.h5')

    channels=1
    sample_rate=16000
    chunk=1600

    if(filename==None):
        one,another=Pipe()
        subprocess=Process(target=listen,args=(channels,sample_rate,chunk,one))
        subprocess.start()

        frame=[]
        while True:
            try:
                recv=another.recv()
            except EOFError:
                break
            slice = st.unpack(str(chunk)+'h', recv)
            if(len(frame)<sample_rate*3):
                frame.extend(slice)

            else:
                signal=ser.endpoint_detection(frame.copy())
                frame.clear()
                if(len(signal)<8000):
                    continue
                else:
                    ser.predict(signal)

    else:
        pass



    one.close()
    another.close()

    print('=======================================END==================================')

if __name__ == "__main__":
    main()
