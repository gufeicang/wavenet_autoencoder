from audio_func import mu_law_decode
from collections import OrderedDict
from train import load_model
import json
import numpy as np 
from torch.autograd import Variable
import torch.nn.functional as F 
import torch
import os
import librosa
from model import wavenet_autoencoder
import pickle 
import string
def predict_next(net,input_wav,quantization_channel = 100):
	out_wav = net(input_wav)
	out = out_wav.view(-1,quantization_channel)
	last = out[-1,:]
	last = last.view(-1)
	_,predict = torch.topk(last, 1)
	return int(predict)
def create_dictionary():
    """
    create char2id, id2char and vocab_size
    from printable ascii characters.
    """
    chars = sorted(ch for ch in string.printable if ch not in ("\x0b", "\x0c", "\r"))
    char2id = dict((ch, i + 1) for i, ch in enumerate(chars))
    char2id.update({"": 0})
    id2char = dict((char2id[ch], ch) for ch in char2id)
    vocab_size = len(char2id)
    id2char.update({98:'\\unk',99:'\\unk'})
    return char2id, id2char, vocab_size,chars

def generate(model_path,model_name,generate_path,generate_name,start_piece=None,sr = 1600,duration=1):
	if os.path.exists(generate_path) is False:
		os.makedirs(generate_path)
	with open('./params/model_params_text.json') as f :
		model_params = json.load(f)
	f.close()
	net = wavenet_autoencoder(**model_params)
	net = load_model(net,model_path,model_name)
	cuda_available = torch.cuda.is_available()
	if cuda_available is True:
		net = net.cuda()
#	print(net.receptive_field)
	if start_piece is None:
		data= open('../np_text1.pkl','rb')
		data = pickle.load(data)
		data = np.array(data)
		data = data[0]
		data = torch.from_numpy(data)
		data = data[-net.receptive_field-512:]
		start_piece = torch.zeros(100,net.receptive_field+512)
		start_piece[data.numpy(),np.arange(net.receptive_field+512)] = 1
#		start_piece = torch.from_numpy(start_piece)
		start_piece = start_piece.view(1,100,net.receptive_field+512)
		start_piece = Variable(start_piece)
		del data		
	#	start_piece = torch.zeros(1, 256, net.receptive_field+512)
	#	start_piece[:, 128, :] = 1.0
	#	start_piece = Variable(start_piece)
	if cuda_available is True:
		start_piece = start_piece.cuda()
	note_num = duration * sr
	note = start_piece
	state_queue = None
	generated_piece = []
	input_wav = start_piece
	char2id, id2char, vocab_size,chars = create_dictionary()
	for i in range(note_num):
		print(i)
		predict_note = predict_next(net, input_wav)
		generated_piece.append(predict_note)
		temp = torch.zeros(net.quantization_channel,1)
		temp[predict_note] =1
		temp = temp.view(1,net.quantization_channel,1)
#		temp = torch.zeros(1, net.quantization_channel, 1)
#		temp[:, predict_note, :] = 1.0
		note = Variable(temp)
		note = note.cuda()
#	#	print(note.size())
#	#	print(input_wav.size())
		input_wav = torch.cat((input_wav[:,-net.receptive_field-510:],note), 2)
	print(generated_piece)
	generated_piece = torch.LongTensor(generated_piece)
	generated_piece = [id2char[i] for i in generated_piece ]
	generated_piece = ''.join(generated_piece)
	output = open(generate_path+'generated_piece2.txt','w')
#       pickle.dump(generated_piece,output)
	output.write(str(generated_piece))
	output.close()

if __name__ =='__main__':
	generate('./restore_text1/', 'wavenet_autoencoder10000.model','./generate_text/','generate2' )
