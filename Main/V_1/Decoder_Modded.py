import torch , dataset , tokenizer
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu';

import matplotlib.pyplot as plt

vocab_size = len(tokenizer.vocabulary);
num_embeddings = 32;
num_heads = 4;
head_size = num_embeddings//num_heads;




class DecoderLM(nn.Module):

  def __init__(self):
    super().__init__()

    self.token_embedding_matrix = nn.Embedding(vocab_size , num_embeddings)
    self.lm_head = nn.Linear(num_embeddings , vocab_size)
    self.position_embedding_matrix = nn.Embedding(dataset.context_size , num_embeddings)

    # We pass the input through multiple Multi head attention layers , creating a deeper network
    self.blocks = nn.Sequential(
      AttentionBlock( num_embeddings , num_heads = 4 ),
      AttentionBlock( num_embeddings , num_heads = 4 ),
      AttentionBlock( num_embeddings , num_heads = 4 ),
      nn.LayerNorm(num_embeddings) # One layer norm should be at the end of the transformer before the final linear layer
    )

    # Issue with this is , as the network gets deeper , it loses more and more information about the INPUTS provided and Deeper networks struggle with optimization issues!
    # A simple way to counter this is to add a Residual connection - Adding Inputs of the layer to its Output



  def forward(self , idx , targets = None):

    B , T = idx.shape ;

    token_embedding = self.token_embedding_matrix(idx) # (B , T , C)
    positional_embedding = self.position_embedding_matrix(torch.arange(T , device = device)) # (T,C)
    x = token_embedding + positional_embedding # B,T,C
    x = self.blocks(x) # B,T,C # Transformer blocks (WITH PRE-NORM) + layer norm in the end
    # x = self.self_attention_head(x)
    # # Attention Happened but it did not give us enough information to PREDICT the next token , it just mapped the relevance of tokens to each other , based on this mapping , we can Use a Neural network to find patterns in the data and predict the next token.
    # x = self.feed_forward(x) # x.shape --> (B , T , C)
    # Now we have enough idea to predict the next token based on the previous tokens and define the logits
    logits =  self.lm_head(x) # (B , T , Vocab_size) # final linear layer





    if targets is None:
      loss = None;

    else :
      B,T,C = logits.shape
      logits = logits.view(B*T , C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits , targets);

      # converting logits back to their oiginal dims
      logits = logits.view(B,T,C)

    return logits , loss;

  def generate(self, idx , max_new_tokens):


    for _ in range(max_new_tokens):
      # We have to make sure that idx should not be more than context size , so crop the context
      idx_cropped = idx[: , -dataset.context_size :]
      logits , loss = self(idx_cropped);
      logits = logits[: , -1 , :];
      probs = F.softmax(logits , dim = -1);
      idx_next = torch.multinomial(probs , num_samples=1)
      idx = torch.cat((idx , idx_next) , dim = 1)

    return idx



class AttentionHead(nn.Module):
  # One head of self attention
  def __init__(self , head_size):
    super().__init__()
    self.key = nn.Linear(num_embeddings , head_size)
    self.query = nn.Linear(num_embeddings , head_size)
    self.value = nn.Linear(num_embeddings , head_size)
    self.register_buffer('tril' , torch.tril(torch.ones(dataset.context_size , dataset.context_size))) # Not a model parameter just a naming convention

  def forward(self , x):
    B , T , C = x.shape
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)


    weights = q @ k.transpose(-2, -1) / (k.shape[-1]**0.5)
    weights = weights.masked_fill(self.tril[:T,:T] == 0 , float('-inf'))
    weights = F.softmax(weights , dim= -1)

    output = weights@v

    return output


class MultiHeadSelfAttention(nn.Module):
  def __init__(self , num_heads , head_size):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)]) # Running all heads in parellel into a list and concatinating over the channel dim (deviding channels amond heads to capture different properties)
    self.projection = nn.Linear(num_embeddings , num_embeddings)


  def forward(self , x):
    output = torch.cat([h(x) for h in self.heads], dim = -1) # concatinating 4*8 = 32 dim back , BUT this info is all seperately stored by different dimensions , not combined or interrelated
    output = self.projection(output) # Pass through a linear layer to combine the information , finding relations in the info gathered by different heads
    return output


# Basic neural network
class FeedForward(nn.Module):
  def __init__(self,num_embeddings):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(num_embeddings , num_embeddings*4), # expanding the dimension (To capture more )
      nn.ReLU(),
      nn.Linear(num_embeddings*4 , num_embeddings), # contracting the dimension back AND this also works as the projection matrix

    )

  def forward(self,x):
    return self.net(x)

class AttentionBlock(nn.Module):
  def __init__(self , num_embeddings , num_heads):
    super().__init__()
    self.sa = MultiHeadSelfAttention(num_heads , head_size) # Self attention is applied and here we add a feed forward network
    self.ffwd = FeedForward(num_embeddings)
    self.attention_layer_norm = nn.LayerNorm(num_embeddings)
    self.feedforward_layer_norm = nn.LayerNorm(num_embeddings)


  def forward(self , x):
    # Attention with residual connections
    # (adding nurmalization before processing)

    x = x + self.sa(self.attention_layer_norm(x))


    # Feed Forward with residual connections
    x = x + self.ffwd(self.feedforward_layer_norm(x))
    return x

# ==================================================================================================================================

# Note - we can define hyperparameters together

model = DecoderLM()
model = model.to(device)
logits , loss =model.forward(dataset.x,dataset.y) # (B,T) every single input is going to refer to the embedding table (vocab_size * vocab_size)




idx = torch.zeros((1,1) , dtype=torch.long) # cresating a 1X1 Dim tensor that is 0 as the first input and will kick off the generation




optimizer = torch.optim.AdamW(model.parameters() , lr = 1e-3)

for _ in range(3000):
  xb , yb = dataset.retrive_batch('train')
  xt , yt = dataset.retrive_batch('test')
  logits , loss = model.forward(xb,yb)
  _ , loss2 = model.forward(xt,yt)
  optimizer.zero_grad(set_to_none = True)
  loss.backward()
  optimizer.step()
  print(loss.item() , loss2.item())





print(tokenizer.decode(model.generate(idx , max_new_tokens=1000).tolist()[0]))

# Using Multi Head Attention - Applying multiple attention in parellel and concatinating the result

# We first perform self attention and then pass it though a feed forward network to find deeper patterns in the embeddings so we can make better predictions

# More recently the attention is slightly modified and Residual connections and Layer Norm are Added Before attention block processing (pre-norm formulation)



# NOTE - Original transformer was built for Machine Translation and used Decoder-Encoder Architecture , but our model here is a type of Decoder-only transformer that generates text based on data.

# Tweek Metaparameters to produce good outputs