import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
random.seed(577)

import numpy as np
np.random.seed(577)

import torch
torch.set_default_tensor_type(torch.FloatTensor)
torch.use_deterministic_algorithms(True)
torch.manual_seed(577)
torch_device = torch.device("cpu")

'''
NOTE: Do not change any of the statements above regarding random/numpy/pytorch.
You can import other built-in libraries (e.g. collections) or pre-specified external libraries
such as pandas, nltk and gensim below. 
Also, if you'd like to declare some helper functions, please do so in utils.py and
change the last import statement below.
'''
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

from neural_archs import DAN, RNN, LSTM
from utils import WiCDataset
from utils import Processed_Dataset_train
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
from utils import Processed_Dataset_test


if __name__ == "__main__":
    print('main func')
    parser = argparse.ArgumentParser()

    # TODO: change the `default' attribute in the following 3 lines with the choice
    # that achieved the best performance for your case
    parser.add_argument('--neural_arch', choices=['dan', 'rnn', 'lstm'], default='rnn', type=str)
    parser.add_argument('--rnn_bidirect', default=False, action='store_true')
    parser.add_argument('--init_word_embs', choices=['scratch', 'glove'], default='glove', type=str)

    args = parser.parse_args()


    
    
    
    
    #HYPERPARAMETERS  - Currently passed individually to models
    INPUT_SIZE = 3
    HIDDEN_SIZE = 2
    SEQ_LENGTH = 5
    NUM_LAYERS = 1
    BATCH_SIZE = 4
    EMBED_DIM = 50 #glove 50
    MAX_LENGTH = 30


    #pads the tokenized values to a determined max length
    def pad(lst, pad_token, max_length):
        if (len(lst) < max_length):
            while (len(lst) < max_length):
                lst.append(pad_token)
        
        else:
            lst = lst[:max_length]
        
        return lst



    #Goes through the words in WicDataset and adds them to the word embeddings as zeros if not already present
    dic = {}
    tokenized_sentence_one = []
    tokenized_sentence_two = []
    if args.init_word_embs == "glove":
        # TODO: Feed the GloVe embeddings to NN modules appropriately
        # for initializing the embeddings
        glove_embs = api.load("glove-wiki-gigaword-50")
        loc = api.load("glove-wiki-gigaword-50", return_path=True)

        #the training dataset
        ds = WiCDataset('train')
    
        #this is the dictionary of words in glove
        dic = list(glove_embs.index_to_key)    
            
        for sample in ds:

            #get the sentences padded
            tokenized = sample['sentence_one'].split()
            tokenized = pad(tokenized, '0', MAX_LENGTH)
            tokenized_sentence_one.append(tokenized)

            tokenized = sample['sentence_two'].split()
            tokenized = pad(tokenized, '0', MAX_LENGTH)
            tokenized_sentence_two.append(tokenized)


            for word in sample['sentence_one'].split():
                if word not in dic:
                    dic.append(word)
                    glove_embs[word] = (np.zeros(50))
            
            for word in sample['sentence_two'].split(): 
                if word not in dic:
                    dic.append(word)
                    glove_embs[word] = (np.zeros(50))

        
        #this is a hack way to assign unique numbers to each word for embedding
        word_dic = {}
        for i in range(0,len(dic)):
            word_dic[dic[i]] = i
        
        dic = word_dic
        weights = torch.FloatTensor(glove_embs.vectors)
        emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(weights))
        #print(emb)
        

    else: #random embeddings
        ds = WiCDataset('train')
        
        print("do random embeddings")
        i=0
        for sample in ds:
            tokenized = sample['sentence_one'].split()
            tokenized = pad(tokenized, '0', MAX_LENGTH)
            tokenized_sentence_one.append(tokenized)

            tokenized = sample['sentence_two'].split()
            tokenized = pad(tokenized, '0', MAX_LENGTH)
            tokenized_sentence_two.append(tokenized)

            for word in sample['sentence_one'].split():
                if word not in dic:
                    dic[word] = i
                    i+=1
            
            for word in sample['sentence_two'].split():
                    dic[word] = i
                    i+=1
        emb = torch.nn.Embedding(i, 50)#just using 50 to match glove dimensions
        #print(emb)
        #These embeddings should be initialized randomly inside of pytorch during instantiation

    tokenizer = lambda x: x.split()
    print(type(ds[1]))
    print(ds[1])

    labels = [ds[i]['label'] for i in range(0,len(ds))]
    labels = [1 if labels[i]=='T' else 0 for i in range(len(labels))]
    labels = torch.DoubleTensor(labels)



    # TODO: Freely modify the inputs to the declaration of each module below
    if args.neural_arch == "dan":
        print("dan is not working")
        model = DAN(50,70,1,emb).to(torch_device)
        print('dan is working')
    elif args.neural_arch == "rnn":
        if args.rnn_bidirect:
            model = RNN(50,30,1,1,emb,True).to(torch_device)
        else:
            model = RNN(50,30,1,1,emb,False).to(torch_device)
    elif args.neural_arch == "lstm":
        if args.rnn_bidirect:
            model = LSTM(50,30,1,1,emb,True).to(torch_device)
        else:
            model = LSTM(50,30,1,1,emb,False).to(torch_device)

    running_in_one = []
    running_in_two = []


    for i in range(len(ds)):
        s_one = [dic[w] for w in tokenized_sentence_one[i]]
        s_two = [dic[w] for w in tokenized_sentence_two[i]]
        #print(s_one)
        running_in_one.append(s_one)
        running_in_two.append(s_two)
    
    #print(running_in_one[:5])
    print(running_in_one)

    running_in_one = torch.LongTensor(running_in_one)
    running_in_two = torch.LongTensor(running_in_two)
    


    data_after_processing = Processed_Dataset_train(running_in_one, running_in_two, labels)
    data_processed_dataloader = DataLoader(data_after_processing, batch_size = 500, shuffle = False)



    # TODO: Training and validation loop here
    #criterion = nn.functional.binary_cross_entropy()



    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-4)
    epochs = 5 #CHANGE THIS BACK to 5

    
    model.train()
    torch.autograd.set_detect_anomaly(True)

    
    for epoch in range(epochs):
        print('=---------EPOCHS-----------------' + str(epoch))
        running_loss = 0.0
        total = 0
        acc = 0.0
        #optimizer.zero_grad()
        for batch_data in data_processed_dataloader: 
            optimizer.zero_grad()

            sent1 = batch_data[0].to(torch_device)

            sent2 = batch_data[1].to(torch_device)

            label_batch = batch_data[2].to(torch_device)

            outputs = model(sent1,sent2)
            outputs_fin = torch.round(outputs).squeeze(1)

            loss = torch.nn.functional.binary_cross_entropy(outputs_fin.float(), label_batch.float())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            acc += (outputs_fin == label_batch).sum().item()
        


    # TODO: Testing loop
    # Write predictions (F or T) for each test example into test.pred.txt


    testset = WiCDataset('test')

    i=0
    tokenized_sentence_one = []
    tokenized_sentence_two = []

    for sample in testset:
        tokenized = sample['sentence_one'].split()
        tokenized = pad(tokenized, '0', MAX_LENGTH)
        tokenized_sentence_one.append(tokenized)

        tokenized = sample['sentence_two'].split()
        tokenized = pad(tokenized, '0', MAX_LENGTH)
        tokenized_sentence_two.append(tokenized)
    
    
    print(tokenized_sentence_one)
    print(len(tokenized_sentence_one))
    print(len(tokenized_sentence_one[0]))
    tokenized_padded_data_s1test = pad(tokenized_sentence_one, '0', 30)
    tokenized_padded_data_s2test = pad(tokenized_sentence_two, '0', 30)

    #print('---------------tokenized data ------------')

    in_one_test = []
    in_two_test = []

    for i in range(len(testset)):
        x1test = [dic[w] if w in dic else 1 for w in tokenized_sentence_one[i]]
        x2test = [dic[w] if w in dic else 1 for w in tokenized_sentence_two[i]]


        in_one_test.append(x1test)
        in_two_test.append(x2test)
    
    print(x1test)
    x1test = torch.LongTensor(in_one_test)
    x2test = torch.LongTensor(in_two_test)

    data_after_processingtest = Processed_Dataset_test(in_one_test, in_two_test)
    data_processed_dataloadertest = DataLoader(data_after_processingtest, batch_size = 100, shuffle = False) #Optimal Bsize for DAN


    test_acc = 0.0
    test_loss = 0.0
    out_file = open("WiC_dataset/test/test.pred.txt", "a")
    for testbatch in data_processed_dataloadertest: 
        
        s1_test = torch.stack(testbatch[0], dim = 1)
        s2_test = torch.stack(testbatch[1], dim = 1) 
        

        predict_test = model(s1_test,s2_test)
        predict_fin_test = torch.round(predict_test).squeeze(1)
        
       
        #print(predict_fin_test)
        running_total = []
        for pred_val in predict_test:
            running_total.append(pred_val.item())

        avg = sum(running_total)/len(running_total)
        for pred_val in predict_test: 
            

            #apparently using the avg isnt correct, and the answers seem random
            if((pred_val.item()) < avg):
                out_file.write('F')
            else: 
                out_file.write('T')
            out_file.write('\n')   
        out_file.close()
    
    #gold = open("WiC_dataset/test/test.gold.txt")

    '''
    #THIS CODE WAS USED FOR TESTIng ACCURACY - It didnt work :(
    
    gold = pd.read_csv("WiC_dataset/test/test.gold.txt",sep="\t", header=None)
    predz = pd.read_csv("WiC_dataset/test/test.pred.txt", sep="\t", header=None)
    print(len(gold))
    print(gold[0])
    print(gold.columns)
    print('^^^^^^^^^^^^^^^^^^^^^^^^^')
    total_acc = 0
    running_total = []
    for i in range(0,len(gold[0])):
        if gold[0][i] is predz[0][i]:
            running_total.append(1)
        else:
            running_total.append(0)

    print(gold)
    print(predz)
    total_acc = sum(running_total)/len(running_total)
    print('ACCC:')
    print(total_acc)
    '''