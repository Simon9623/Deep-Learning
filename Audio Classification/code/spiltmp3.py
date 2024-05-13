from pydub import AudioSegment
from pydub.utils import make_chunks

audio = AudioSegment.from_file("电视剧.wav", "wav")

size = 10000  #切割的毫秒数 10s=10000

chunks = make_chunks(audio, size)  #将文件切割为10s一块

for i, chunk in enumerate(chunks):
    chunk_name = "dianshiju-{0}.wav".format(i)
    print(chunk_name)
    chunk.export(chunk_name, format="wav")
