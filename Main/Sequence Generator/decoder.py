import torch , dataset , tokenizer
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu';

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


    self.blocks = nn.Sequential(
      AttentionBlock( num_embeddings , num_heads = 4 ),
      AttentionBlock( num_embeddings , num_heads = 4 ),
      AttentionBlock( num_embeddings , num_heads = 4 ),
      nn.LayerNorm(num_embeddings)
    )




  def forward(self , idx , targets = None):

    B , T = idx.shape ;

    token_embedding = self.token_embedding_matrix(idx)
    positional_embedding = self.position_embedding_matrix(torch.arange(T , device = device))
    x = token_embedding + positional_embedding
    x = self.blocks(x)

    logits =  self.lm_head(x)





    if targets is None:
      loss = None;

    else :
      B,T,C = logits.shape
      logits = logits.view(B*T , C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits , targets);


      logits = logits.view(B,T,C)

    return logits , loss;

  def generate(self, idx , max_new_tokens):


    for _ in range(max_new_tokens):

      idx_cropped = idx[: , -dataset.context_size :]
      logits , loss = self(idx_cropped);
      logits = logits[: , -1 , :];
      probs = F.softmax(logits , dim = -1);
      idx_next = torch.multinomial(probs , num_samples=1)
      idx = torch.cat((idx , idx_next) , dim = 1)

    return idx



class AttentionHead(nn.Module):

  def __init__(self , head_size):
    super().__init__()
    self.key = nn.Linear(num_embeddings , head_size)
    self.query = nn.Linear(num_embeddings , head_size)
    self.value = nn.Linear(num_embeddings , head_size)
    self.register_buffer('tril' , torch.tril(torch.ones(dataset.context_size , dataset.context_size)))

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
    self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
    self.projection = nn.Linear(num_embeddings , num_embeddings)

  def forward(self , x):
    output = torch.cat([h(x) for h in self.heads], dim = -1)
    output = self.projection(output)
    return output



class FeedForward(nn.Module):
  def __init__(self,num_embeddings):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(num_embeddings , num_embeddings*4),
      nn.ReLU(),
      nn.Linear(num_embeddings*4 , num_embeddings),
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

    x = x + self.sa(self.attention_layer_norm(x))
    x = x + self.ffwd(self.feedforward_layer_norm(x))
    return x

# ==================================================================================================================================


model = DecoderLM()
model = model.to(device)
logits , loss =model.forward(dataset.x,dataset.y)
idx = torch.zeros((1,1) , dtype=torch.long)

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
