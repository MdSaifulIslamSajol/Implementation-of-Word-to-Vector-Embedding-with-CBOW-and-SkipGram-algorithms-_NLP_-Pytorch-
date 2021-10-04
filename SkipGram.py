# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:07:11 2021

@author: saifu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

#%%
#Downloading the data if not available
#!wget -qq https://s3.amazonaws.com/video.udacity-data.com/topher/2018/October/5bbe6499_text8/text8.zip

# Unzip the folder
#with zipfile.ZipFile("text8.zip","r") as zip_ref:
 #   zip_ref.extractall("C:/Users/saifu")

#%%
#reading the text data from the file
with open('C:/Users/saifu/text8') as f:    #open the content of the file 
    text8 = f.read()         # read it
    print("lenght of original dataset:{}".format(len(text8)) )
#printing the first 500 characters
print("Some sample texts:" ,text8[:500])     #printing the first 500 characters
  
text= text8.lower().split()
text= text[0:500]    # taking only irst 200 words
print("\nlenght of dataset after slicing:{}".format(len(text)) )

vocab = set(text)   #   {'plow', 'his', 'wind', 'silver', '}
vocab_size = len(vocab)  
print('vocab_size:', vocab_size)

word2index = {w: i for i, w in enumerate(vocab)}
index2word = {i: w for i, w in enumerate(vocab)}

sentences= text
vocab2id= word2index
Window_Size=3
embed_dim=70

def make_dataset(sentences, vocab2id, window_size):
    data = list()

    ids = [vocab2id[vocab] for vocab in sentences]
    for i in range(len(ids)):
      
            X = ids[i]
            random_window = torch.LongTensor(1).random_(1, window_size+1).item()
            for j in range(max(i-random_window, 0), min(i+1+random_window, len(sentences))):
                if j != i:
                    y = ids[j]
                    data.append((X, y))

    return DataLoader(data, pin_memory=True, num_workers=0)
    
train_dataset = make_dataset( sentences, vocab2id, window_size=Window_Size)

print("text[:10]")
print(text[:10])

class SkipGram(nn.Module):
    def __init__(self, vocab_dim, embed_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_dim, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_dim)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        outs = self.linear(embeds)
        log_probs = F.log_softmax(outs)
        return log_probs

loss_list=[]
model = SkipGram(vocab_size, embed_dim)
        
def train(training_data, num_epochs=30, learning_rate=0.025):
    
    device = torch.device('cuda:0')
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.NLLLoss()
    
    # start training
    for epoch in range(num_epochs):
        total_loss=0.0
        for (X, y) in training_data:  # X.shape=torch.Size([1]) ,  y.shape= torch.Size([1])
            
            if X.nelement() != 0:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                output = model.forward(X)  # output.shape = torch.Size([1, 276])
                loss = loss_function(output,y)
                loss.backward()
                optimizer.step()
                total_loss+= loss
              
        print('epoch : {}, Loss : {:.3f}'.format(epoch+1, total_loss.item()))
        loss_list.append(total_loss)
    print("Training finished")
                
train(train_dataset)

plt.figure()
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('loss over epochs')
plt.savefig('loss_over_epochs_skipgram.png')
plt.show()

#%%
#  Cosine Similarity
C = 3  # window size
target = 'believe'
target in vocab    

def word_cosine(model, target, vocab_set, vocab2id):
    model.to('cpu')
    target_embed = model.embeddings(torch.LongTensor([[vocab2id[target]]]))
    target_similar = dict()

    for vocab in vocab_set:
        if vocab != target:
            vocab_embed = model.embeddings(torch.LongTensor([vocab2id[vocab]]))
            similarity = F.cosine_similarity(target_embed.squeeze(dim=0), vocab_embed).item()
            if len(target_similar) < 15:
                target_similar[round(similarity, 6)] = vocab
            elif min(target_similar.keys()) < similarity:
                del target_similar[min(target_similar.keys())]
                target_similar[round(similarity, 6)] = vocab
    
    return sorted(target_similar.items(), reverse=False)    
    
    #%%
#Euclidean Distance

def word_euclidean(model, target, vocab_set, vocab2id):
    model.to('cpu')
    target_embed = model.embeddings(torch.LongTensor([[vocab2id[target]]]))
    target_similar = dict()

    for vocab in vocab_set:
        if vocab != target:
            vocab_embed = model.embeddings(torch.LongTensor([[vocab2id[vocab]]]))
            similarity = torch.dist(target_embed, vocab_embed, 2).item()
            if len(target_similar) < 15:
                target_similar[round(similarity, 6)] = vocab
            elif min(target_similar.keys()) > similarity:
                del target_similar[min(target_similar.keys())]
                target_similar[round(similarity, 6)] = vocab
    
    return sorted(target_similar.items(), reverse=False)

cos=word_cosine(model, target, vocab, word2index)
eucld= word_euclidean(model,target, vocab, word2index)
print('Embed dim : {}'.format(embed_dim))
#%%

print('\nCosine Similarity')
for i, (score, vocab) in enumerate(cos):
    print('Top {} word : {}    score : {}'.format(i+1, vocab, score))

print('\nEuclidean Distance')
for i, (score, vocab) in enumerate(eucld):
    print('Top {} word : {}    score : {}'.format(i+1, vocab, score))

#%%
# visulaize the words
embeddings = model.embeddings.weight.to('cpu').data.numpy()
viz_words = 600
tsne = TSNE()
#I want to visualize first 600 words 
embed_tsne = tsne.fit_transform(embeddings[: viz_words, :])
fig, ax = plt.subplots(figsize=(16, 16))

for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], c='g')
    plt.annotate(index2word[idx], (embed_tsne[idx,0], embed_tsne[idx, 1]), alpha=0.9)

