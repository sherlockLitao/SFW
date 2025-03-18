# nohup python -u preprocessing.py --embedding "poincare" --dataset "bbcsport" > preprocessing.log 2>&1 &
# nohup python -u preprocessing.py --embedding "euclidean" --dataset "bbcsport" > preprocessing.log 2>&1 &

import numpy as np
import pandas as pd
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--embedding", type=str, default="poincare", help="Which word2vec to use")
parser.add_argument("--dataset", type=str, default="amazon", help="Which dataset to use")
args = parser.parse_args()


assert (args.embedding=="poincare" or args.embedding=="euclidean"), \
            "Word2vec should only be Poincare or Euclidean"


if args.embedding=="poincare":
    data_ = pd.read_table("word2vec/poincare_glove_100D_cosh-dist-sq_init_trick.txt", delimiter=" ", skiprows=1)
    data = np.array(data_.iloc[:,1:101])
else:
    data_ = pd.read_table("word2vec/vanilla_glove_100D_init_trick.txt", delimiter=" ", skiprows=1)
    data = np.array(data_.iloc[:,1:101])
   

def read_line_by_line(dataset_name, data_vec, data_index, vec_size):
    
    # load stop words
    SW = set()
    for line in open('raw_dataset/stop_words.txt'):
        line = line.strip()
        if line != '':
            SW.add(line)

    stop = list(SW)


    if args.dataset=="bbcsport":
        f = open(dataset_name, encoding="ISO-8859-1")
    else:
        f = open(dataset_name)

    C = np.array([], dtype=object)
    if args.dataset=="bbcsport":
        num_lines = sum(1 for line in open(dataset_name, encoding="ISO-8859-1"))
    else:
        num_lines = sum(1 for line in open(dataset_name))

    y = np.zeros((num_lines,))
    X = np.zeros((num_lines,), dtype=object)
    BOW_X = np.zeros((num_lines,), dtype=object)
    count = 0
    remain = np.zeros((num_lines,), dtype=object)
    the_words = np.zeros((num_lines,), dtype=object)
    for line in f:
        line = line.strip()
        line = line.translate(str.maketrans("",""))
        T = line.split('\t')
        classID = T[0]
        if classID in C:
            IXC = np.where(C==classID)
            y[count] = IXC[0]+1
        else:
            C = np.append(C,classID)
            y[count] = len(C)
        W = line.split()
        F = np.zeros((vec_size,len(W)-1))
        inner = 0
        RC = np.zeros((len(W)-1,), dtype=object)
        word_order = np.zeros((len(W)-1), dtype=object)
        bow_x = np.zeros((len(W)-1,))
        for word in W[1:len(W)]:
            try:
                if word in stop:
                    word_order[inner] = ''
                    continue
                if word in word_order:
                    IXW = np.where(word_order==word)
                    bow_x[IXW] += 1
                    word_order[inner] = ''
                elif word in list(data_index):
                    word_order[inner] = word
                    bow_x[inner] += 1
                    F[:,inner] = data_vec[np.where(word == data_index)[0][0],:]
            except KeyError:
                word_order[inner] = ''
            inner = inner + 1
        Fs = F.T[~np.all(F.T == 0, axis=1)]
        word_orders = word_order[word_order != '']
        bow_xs = bow_x[bow_x != 0]
        X[count] = Fs.T
        the_words[count] = word_orders
        BOW_X[count] = bow_xs
        count = count + 1
    return (X,BOW_X,y,C,the_words)



vec_size = 100
train_dataset = 'raw_dataset/' + args.dataset +'.txt'
(X,BOW_X,y,C,words)  = read_line_by_line(train_dataset, data, data_.iloc[:,0], vec_size)


# remove empty sample 
id = []
for i in range(X.shape[0]):
    if X[i].shape[1] >0:
        id.append(i)

X = X[id]
y = y[id]
BOW_X = BOW_X[id]


with open('./preprocess_dataset/'+ args.dataset +'/' + args.dataset + '_' + args.embedding + '_X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('./preprocess_dataset/'+ args.dataset +'/' + args.dataset + '_' + args.embedding + '_w.pkl', 'wb') as f:
    pickle.dump(BOW_X, f)
np.save('./preprocess_dataset/'+ args.dataset +'/' + args.dataset + '_' + args.embedding + '_y.npy',y)