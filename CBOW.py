import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
CUDA = torch.cuda.is_available()
torch.manual_seed(42)

#%%
#Downloading the data if not available
#!wget -qq https://s3.amazonaws.com/video.udacity-data.com/topher/2018/October/5bbe6499_text8/text8.zip

# Unzip the folder
#with zipfile.ZipFile("text8.zip","r") as zip_ref:
 #   zip_ref.extractall("C:/Users/saifu")

#%%

learning_rate = 0.001
epochs = 25

#reading the text data from the file
with open('C:/Users/saifu/text8') as f:    #open the content of the file 
    text8 = f.read()         # read it
    print("lenght of original dataset:{}".format(len(text8)) )
print("Some sample texts:" ,text8[:300])     #printing the first 500 characters

text= text8.lower().split()
text= text[0:500]

vocab = set(text)  # converting to unique set  ## type(vocab) = set 
word2index = {w:i for i,w in enumerate(vocab)}   # giving index to each of the set by making a dictionary""
index2word = {i:w for i,w in enumerate(vocab)}

class CBOW(nn.Module):
    
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        if CUDA:
            self.embedding = self.embedding.cuda()
        self.hidden = nn.Linear(embedding_size, vocab_size)
        # self.op = F.LogSoftmax()
        
    def forward(self, X):
        embeds = self.embedding(X.long())
        mean_embed = torch.mean(embeds, dim=0).view(1,-1)
        outs = self.hidden(mean_embed)
        log_probs = F.log_softmax(outs) 
        return log_probs
    
def text_to_train(text, context_window):
    '''
    parameters:
    text = list type input
    context_window= int
    Convert text to data for training cbow model
    '''
    
    data = []
    
    for i in range(context_window, len(text) - context_window):
        context = [
            text[i+e] for e in range(-context_window, context_window+1) if i+e != i
        ]
        target = text[i]
        
        data.append((context, target))
    return data

data = text_to_train(text, 2)

print("data samples :", data[:10])


#%%

def words_to_tensor(words: list, w2i: dict, dtype=torch.FloatTensor):
    """ 
    converts a list of words in to correspondinf tensor of indices
    resturns a tensor of word indices
    """
    tensor =  dtype([ w2i[word] for word in words])
    
    if CUDA:
        tensor = tensor.cuda()
    return Variable(tensor)

def get_prediction(context, model):
    """
    

    Parameters
    ----------
    context : list
        DESCRIPTION. context words a list  e.g. ['as', 'the', 'strange', 'beings']
    model : <class '__main__.CBOW'>
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    model.eval()
    context_tensors = words_to_tensor(context, word2index)
    prediction = model(context_tensors)  #  prediction.shape = torch.Size([1, 171])
    # _, index = torch.max(prediction, 1)
    # return index2word[index.data[0]]
    _ , index = torch.max(prediction, 1)  # we are only interested with the index of the maximum probable word
    return index2word[index.item()]

def check_accuracy(model):
    correct = 0
    for context, target in data:
        prediction = get_prediction(context, model)
        if prediction == target:
            correct += 1
    accuracy= (correct/len(data))*100
    return accuracy

#%%
    
## Training
model = CBOW(len(vocab), 100)

if CUDA:
    model = model.cuda()

loss_func = torch.nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
losses = []
accuracy_list=[]

for epoch in range(epochs):
    total_loss = 0
    
    for context, target in data:  # len(context)=4,  len(target)=1
        
        # context= ['anarchism', 'originated', 'a', 'term'] ,  target= tensor([157]
        
        ids = words_to_tensor(context,word2index)   # e.g. ids=tensor([ 58.,44.,163.,121.],   ids.shape= torch.Size([4])
        target = words_to_tensor([target], word2index, dtype=torch.LongTensor)        # target.shape = torch.Size([1])
        model.zero_grad()
        output = model(ids)   #  output.shape torch.Size([1,171],  len(vocab)=171 )        
        loss = loss_func(output, target)   # output.shape= torch.Size([1, 171]), target.shape   torch.Size([1])
        loss.backward()
        optimizer.step()        
        total_loss += loss  
        
    losses.append(total_loss)    
    print("Epoch {} training Loss: {:.2f} at  " .format( epoch, total_loss.item()))     
    if epoch% 2 == 0:
        accuracy = check_accuracy(model)
        accuracy_list.append(accuracy)
        print("\n   Accuracy at epoch {} is {:.2f}% \n ".format(epoch, accuracy))

#  Plot the curves
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title('loss over epochs')
plt.savefig('loss_over_epochs_cbow.png')
plt.show()


plt.figure()
plt.plot(accuracy_list)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title('Accuracy over epochs')
plt.savefig('Accuracy_over_epochs_cbow.png')
plt.show()

#%%    

# measuring the Euclidean distance
C = 3  # window size
target = 'most'
target in vocab    

def word_euclidean(model, target, vocab_set, vocab2id):
    
    model.to('cpu')
    target_embed = model.embedding(torch.LongTensor([[vocab2id[target]]]))
    target_similar = dict()

    for vocab in vocab_set:
        if vocab != target:
            vocab_embed = model.embedding(torch.LongTensor([[vocab2id[vocab]]]))
            similarity = torch.dist(target_embed, vocab_embed, 2).item()
            if len(target_similar) < 10:
                target_similar[round(similarity, 6)] = vocab
            elif min(target_similar.keys()) > similarity:
                del target_similar[min(target_similar.keys())]
                target_similar[round(similarity, 6)] = vocab
    
    return sorted(target_similar.items(), reverse=False)

# measuring the cosine similarity
def word_cosine(model, target, vocab_set, vocab2id):
    
    model.to('cpu')
    target_embed = model.embedding(torch.LongTensor([[vocab2id[target]]]))
    target_similar = dict()

    for vocab in vocab_set:
        if vocab != target:
            vocab_embed = model.embedding(torch.LongTensor([vocab2id[vocab]]))
            similarity = F.cosine_similarity(target_embed.squeeze(dim=0), vocab_embed).item()
            if len(target_similar) < 10:
                target_similar[round(similarity, 6)] = vocab
            elif min(target_similar.keys()) < similarity:
                del target_similar[min(target_similar.keys())]
                target_similar[round(similarity, 6)] = vocab
    
    return sorted(target_similar.items(), reverse=False)    
    
euc= word_euclidean(model,target, vocab, word2index)
cos = word_cosine(model,target, vocab, word2index)
    
print('\nEuclidean Distance')
for i, (score, vocab) in enumerate(euc):
    print('Top {} word : {}    score : {}'.format(i+1, vocab, score))
print('\nCosine Similarity') 
for i, (score, vocab) in enumerate(cos):
    print('Top {} word : {}    score : {}'.format(i+1, vocab, score)) 
    
    
#%%
# TSNE
embeddings = model.embedding.weight.to('cpu').data.numpy()
viz_words = 600
tsne = TSNE()
#visualize first 600 words 
embed_tsne = tsne.fit_transform(embeddings[: viz_words, :])
fig, ax = plt.subplots(figsize=(16, 16))

for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color='g')
    plt.annotate(index2word[idx], (embed_tsne[idx,0], embed_tsne[idx, 1]), alpha=0.9)