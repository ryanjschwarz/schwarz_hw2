from torch.utils.data import Dataset
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class WiCDataset(Dataset):
    
    def clean_data(d):
        try:
            return [word.lower() for word in d if word.lower().isalpha()]

        except:
            print("something went wrong with sample: " + str(d))
    

    def __init__(self, identifier):
        self.dataset = pd.read_table("/homes/cs577/WiC_dataset/"+identifier+"/"+identifier+".data.txt",header=None)
        self.identifier = identifier
        self.labels = ""
        if identifier not in ['test']:
            self.labels = pd.read_table("/homes/cs577/WiC_dataset/"+identifier+"/"+identifier+".gold.txt",header=None)
        pass

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        
      
        #df['sentence_one'] = df.apply(lambda x: clean_data(x['sentence_one']))
        row[0] = " ".join([word.lower() for word in row[0].split() if word.lower().isalpha()])
        row[3] = " ".join([word.lower() for word in row[3].split() if word.lower().isalpha()])
        row[4] = " ".join([word.lower() for word in row[4].split() if word.lower().isalpha()])

        target_word = row[0]
        pos = row[1]
        loc_idx = row[2]
        loc_1 = loc_idx.split('-')[0]
        loc_2 = loc_idx.split('-')[1]
        sentence_one = row[3]
        sentence_two = row[4]

        out = {}
        if self.identifier in ['train', 'dev']:
            label = self.labels.iloc[idx][0]
            out = {'target_word': target_word, 'pos':pos, 'loc1':loc_1, 'loc2':loc_2, 'sentence_one':sentence_one, 'sentence_two':sentence_two,'label':label }

        else:
            out = {'target_word': target_word, 'pos':pos, 'loc1':loc_1, 'loc2':loc_2, 'sentence_one':sentence_one, 'sentence_two':sentence_two}

        return out
    

class Processed_Dataset_train(Dataset):
    def __init__(self, in_one_index, in_two_index, labels):
        self.in_one_index = in_one_index
        self.in_two_index = in_two_index
        self.labels = labels
    
    def __len__(self):
        return len(self.in_one_index)
    
    def __getitem__(self, idx):
        return self.in_one_index[idx], self.in_two_index[idx], self.labels[idx]


class Processed_Dataset_test(Dataset):
    def __init__(self, in_one_index, in_two_index):
        self.in_one_index = in_one_index 
        self.in_two_index = in_two_index
    
    def __len__(self):
        return len(self.in_one_index)
    
    def __getitem__(self, idx):
            return self.in_one_index[idx], self.in_two_index[idx]
