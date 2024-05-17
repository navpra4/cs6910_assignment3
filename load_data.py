import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 31
class Lang:
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS

    def addWord(self, word):
        for char in word:
            self.addchar(char)

    def addchar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

def readLangs(lang1, lang2,path='/kaggle/input/tamil-lang/tam_train.csv'):
    print("Reading lines...")

    # Read the file and split into lines
    data = pd.read_csv(path,names = ['eng','tam'] )
    data = data.values.tolist()
    pairs = [[(s) for s in l] for l in data]

    # Reverse pairs, make Lang instances
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def prepareData(lang1, lang2,path='/kaggle/input/tamil-lang/tam_train.csv'):
    input_lang, output_lang, pairs = readLangs(lang1, lang2,path)
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
    return input_lang, output_lang, pairs

def indexesFromWord(lang, word):
    return [lang.char2index[l] for l in word]

def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

'''def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang, pair[0])
    target_tensor = tensorFromWord(output_lang, pair[1])
    return (input_tensor, target_tensor)'''

def get_dataloader(batch_size,input_lang,output_lang,path='/kaggle/input/tamil-lang/tam_train.csv'):
    _,_, pairs = prepareData('eng', 'tam',path)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromWord(input_lang, inp)
        tgt_ids = indexesFromWord(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def WordFromTensor(lang, tensor):
    word = ''
    for i in tensor:
        index = i.item()
        if index == EOS_token:
            break
        if index == SOS_token:
            continue
        word += lang.index2char[index]
    return word