# In terms of language processing , Bigram model is the simplest possible network.
''' A bigram language model is a simple statistical tool in natural language processing that predicts
    the next item (word or character)in a sequence based solely on the single item that immediately precedes it'''

import torch
import torch.nn as nn
from torch.nn import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'; # for utilizing nvidia GPU if available

import dataset , tokenizer
import matplotlib.pyplot as plt




class BigramLanguageModel(nn.Module):

  # We will have a prediction matrix of vocabulary_size * vocabulary_size where Each row represents the current token of all the possible tokens and each row contains the possibilities of all the tokens to be the next token
  # this matrix will be initialized randomly and then will be Trained to predict the next token based on the current token , thats called a bigram model

  def __init__(self, vocab_size):
    super().__init__()
    # each token directly reads off the logits for the next token from the matrix
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # will be learned

  def forward(self, idx, targets = None):
    # idx and targets are both (B,T) tensor of integers (Batch,Time)
    logits = self.token_embedding_table(idx) # (B,T,C) tensor where B = Batch (in our case 4) , T = Time (in our case 8) , C = Channel (vocab size)

    # Reshaping to be accepted by crossEntropy
    # The cross entropy loss expects the input to be of shape (Batch * Time, Vocab_size) and targets of shape (Batch * Time)
    B,T,C = logits.shape
    logits = logits.view(B*T , C)  # (4 * 8 , 65) -> (32,65) , B,T,C is the shape of raw tensor passed into the forward func

    # We do so here because we dont really need the context here , we just need the current token and the next token , without caring for the contextual information.

    if targets == None:
      loss=None
    else:
      targets = targets.view(B*T) # (32)
      loss = F.cross_entropy(logits , targets) # Cross Entropy Loss is used to calculate the loss between the predicted logits and the actual targets

    # To get it back to original shape so we can access the last token of each batch
    logits = logits.view(B,T,C)
    return logits , loss





  def generate(self , idx , max_new_tokens = 40):
    # idx is (B,T)
    for _ in range(max_new_tokens):
      # get the predictions
      logits , loss = self.forward(idx)

      # focus only the last token
      logits = logits[:,-1,:]

      # get the probabilities using softmax
      probs = torch.softmax(logits, dim = -1)

      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)

      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim = 1) # (B,1)

    return idx

      # given context tokens (B,T) -> we are sampling the next token (B,1) and then append it to the running sequence, which will be used as context in the next iteration.
      # WE ARE essentially just predicting the next token based on the current token , and then append it to the running sequence, which will be used as context in the next iteration.




model = BigramLanguageModel(len(tokenizer.vocab))
model = model.to(device)
logits , loss =model.forward(dataset.x,dataset.y) # (B,T) every single input is going to refer to the embedding table (vocab_size * vocab_size)

print(logits.shape , loss) #(B,T,C) -> A 3D tensor


idx = torch.zeros((1,1) , dtype=torch.long) # cresating a 1X1 Dim tensor that is 0 as the first input and will kick off the generation

print(tokenizer.decode(model.generate(idx , max_new_tokens=100).tolist()[0])) # tolist will convert the tensor to list but it will be a list of lists , so we take [0] to get the list of integers

# NOW we have created a full forward pass model , we need to train this model now.

# we will use AdamW optimizer and not SGD since its a much more advanced and popular optimizer.
optimizer = torch.optim.AdamW(model.parameters() , lr = 0.01)
losses = []
for _ in range(1000):
  xb , yb = dataset.get_batch('train')
  logits , loss = model.forward(xb,yb)
  optimizer.zero_grad(set_to_none = True) # reseting the gradient from the previous iteration
  loss.backward() # backpropagation
  optimizer.step() # gradient decent

  losses.append(loss.item())


plt.plot(losses)
plt.show()

print(tokenizer.decode(model.generate(idx , max_new_tokens=1000).tolist()[0]))

# This is the simplest possible language model

