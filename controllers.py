from pytube import YouTube
import moviepy.editor as mp
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import speech_recognition as sr
import math
import wave
import contextlib
from pydub.utils import make_chunks
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from summarizer import Summarizer
import torch
import glob


def download(url, outpath="./videos"):
	# Write code to download video from youtube
	
	yt = YouTube(url)
	print("Downloading video...")
	path = yt.streams.filter(file_extension="mp4").get_by_resolution("360p").download(outpath)
	print("Path of the downloaded file:",path)
	return path

def convertAudio(path):
	# Write code here to convert into audio file
	print("Converting to audio file")
	clip = mp.VideoFileClip(path)
	clip.audio.write_audiofile("./audios/test.wav",codec='pcm_s16le')
	print("Audio file has been generated")
	return './audios/test.wav'

def generateText(path,method):
	# Generate text

	r = sr.Recognizer()
	sound = AudioSegment.from_wav(path)	

	if method == 'abstractive':
		chunk_size = 20#math.ceil(ratio_percent*seconds)
		myaudio = AudioSegment.from_file(path, "wav") 
		chunk_length_ms = chunk_size*1000 # pydub calculates in millisec 
		chunks = make_chunks(myaudio,chunk_length_ms) #Make chunks of one sec 
	else:
		chunks = split_on_silence(sound,min_silence_len = 500,silence_thresh = sound.dBFS-14,keep_silence=2000)

	folder_name = "./audios/chunks"
	if not os.path.isdir(folder_name):
		os.mkdir(folder_name)

	print("running...")
	whole_text = ""
	for i, audio_chunk in enumerate(chunks, start=1):
		print("Processing chunk ",i)
		chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
		audio_chunk.export(chunk_filename, format="wav")
		with sr.AudioFile(chunk_filename) as source:
			audio_listened = r.record(source)
			try:
				text = r.recognize_google(audio_listened)
			except sr.UnknownValueError as e:
				print("")
			else:
				text = f"{text.capitalize()}."
				whole_text += text
	print("final text has been generated")
	print("Final text:")
	print(whole_text)
	return whole_text

def generateSummaryXt(text, x = 0.25):
	print("Executing Extractive Method")
	model = Summarizer('distilbert-base-uncased', hidden=[-1,-2], hidden_concat=True)
	summaryText = model(text)
	print("Summary:")
	print(summaryText)
	return summaryText

def generateSummaryAb(text, x = 0.25):
	print("Executing Abstractive Method")
	tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-wikihow")
	model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-wikihow")

	listTemp = []
	main_listText = []
	
	inputList = list(text.split('.'))
	l = len(inputList)
	chunk_size = math.ceil(l * x)      
	i = 0

	while i < l:
		listTemp.append(inputList[i])
		if len(listTemp) >= chunk_size:
			main_listText.append('.'.join(listTemp))
			listTemp = []
		i += 1
	if len(listTemp) != 0:
		main_listText.append('.'.join(listTemp))
	
	# Create tokens - number representation of our text
	token = tokenizer(main_listText, truncation = True, padding = "longest", return_tensors = "pt")
	# Summarize
	summary = model.generate(**token)
	i = 0
	summaryText = ""
	while(i < len(summary)):
		summaryText += tokenizer.decode(summary[i])
		i += 1
	
	print("Summary:")
	print(summaryText)
	return summaryText

def clearFiles(path = './audios/chunks'):
	files = glob.glob(path)
	for f in files:
		os.remove(f)
	