import fnmatch
import os 
import logging
import numpy as np 
import pickle
import math
import string
import codecs
## reference https://github.com/ThibaultMaho/TextGeneration/blob/master/char-rnn-generation.py#L36
## https://github.com/Zeta36/tensorflow-tex-wavenet/blob/master/wavenet/text_reader.py
## https://github.com/yxtay/char-rnn-text-generation/blob/master/utils.py

def search_files(directory, pattern='*.c'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
           
    return files

def read_text(filename):
    data = []
    with codecs.open(filename, "r",'utf-8',errors='ignore') as f:
        for line in f.readlines():
           # print(line)
            
            line = line.strip('\n')
            if len(line) > 0:
                data.append(line)
    logging.info("Length of Data: {} \n".format(len(data)))
   # logging.info("Random Text: {}".format(data[random.randint(0, len(data))]))
    text_joined = ''.join(data)
 #   list_characters = list(set(text_joined))
 #   logging.info("{} characters".format(len(list_characters)))
    return list(text_joined)

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

def main(savedir,filedir,sr):
    files = search_files(filedir)
    np_text = []
    char2id , _, vocab_size ,chars=create_dictionary()
    for filename in files:
        text = read_text(filename)        
        for index, item in enumerate(text):
            #print (text[index])
            if item in chars:
                text[index] = char2id[text[index]]
            else:
                text[index] = 99   
        text = np.array(text,dtype='int32')
        num_pieces = math.floor(len(text)/sr)
        if num_pieces > 0:
            text = text[:sr*num_pieces]
            text = list(text.reshape(num_pieces, sr))
            np_text.extend(text)
    output = open(savedir+'np_text.pkl','wb')
    pickle.dump(np_text,output)
    output.close()

main('/home/yang/', '/home/yang/Downloads/linux-master/kernel/', 1600)
