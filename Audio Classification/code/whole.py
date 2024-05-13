import pyaudio 
import wave
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import tkinter as tk

def start_recording():
    chunk = 1024
    sample_format = pyaudio.paInt16
    channels = 1
    fs = 44100 # 取樣頻率，常見值為 44100 ( CD )、48000 ( DVD )、22050、24000、12000 和 11025。
    seconds = 10
    filename = "input.wav"

    audio = pyaudio.PyAudio() #build object for audio
    stream = audio.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)

    frames = []                      # 建立聲音串列

    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)          # 將聲音記錄到串列中

    stream.stop_stream()             # 停止錄音
    stream.close()                   # 關閉串流
    audio .terminate()

    print('錄音結束...')

    wf = wave.open(filename, 'wb')   # 開啟聲音記錄檔
    wf.setnchannels(channels)        # 設定聲道
    wf.setsampwidth(audio.get_sample_size(sample_format))  # 設定格式
    wf.setframerate(fs)              # 設定取樣頻率
    wf.writeframes(b''.join(frames)) # 存檔
    wf.close()

# -------------------------------------------------------------------------------------------------------------
# transform

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)
    fig.savefig(image_file)
    plt.close(fig)

    # create_spectrogram('input.wav', 'input.png')

# --------------------------------------------------------------------------------------------------------------------
# prediction
max_label = None
def predict():
    model = load_model('test/audio_classification.h5')

    class_labels = ['background', 'chainsaw', 'engine', 'storm']

    k = Image.open('input.png')
    k = k.resize((224, 224))

    k = k.convert('RGB')

    x_array = image.img_to_array(k)

    x_array = np.expand_dims(x_array, axis=0)

    y = model.predict(x_array)
    print(y)
    for i, label in enumerate(class_labels):
        print(f'{label}: {y[0][i]}')  
    max_index = np.argmax(y)
    global max_label
    max_label = class_labels[max_index]
    a.set(max_label) 
# -------------------------------------------------------------------------------------------------------------------------
def run_functions():
    start_recording()
    create_spectrogram('input.wav', 'input.png')
    predict()
# -------------------------------------------------------------------------------------------------------------------------
root = tk.Tk()
root.title('Audio')
root.geometry('380x400')

a = tk.StringVar()  # 設定 a 為文字變數
a.set(max_label)            # 設定 a 的內容
btn_record = tk.Button(root, text="Start Recording", command=run_functions)
btn_record.pack()

result_label = tk.Label(root, textvariable=a, font=('Arial', 20))
result_label.pack()

root.mainloop()
