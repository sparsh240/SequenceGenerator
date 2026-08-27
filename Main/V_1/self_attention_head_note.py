# Performing masked self attention for a single head
import torch
import torch.nn as nn
B,T,C = 4,8,32
dummy_data = torch.randint(0,10,(B,T,C), dtype=torch.float)

# attention_wt_matrix = torch.tril(torch.ones((T,T)))
# attention_wt_matrix = torch.masked_fill(attention_wt_matrix, attention_wt_matrix == 0 , float('-inf'))
# attention_wt_matrix = torch.softmax(attention_wt_matrix , dim = 1)
# modifiet_wt_output = attention_wt_matrix @ dummy_data
# print(modifiet_wt_output[0])

# SELF ATTENTION

# This method takes one assumption that all the previous tokens are equally relevant to the current token , which is not true
# The relevance of the other tokens to current token depends on the tokens themselves and varies depending on the input.

# This is solved by Query,Key,Value vectors
# Query vector - What is the current token looking for ,
# Key Vector - What information does the current token contain
# Value Vector - What information of the current token is ACTUALLY communicated to the other tokens (Key vector finds Finds likelihood wrt Query ,
# Value vector has the info which is shared to other tokens Once likelihood is established (so all the info is not shared and only relevant info is shared))

# We take query vector of the current token and take dot product with the key vectors of all previous tokens , the higher the dot product ,
# the more relevant the token is to the current token , since similar alignment of the query vector and key vector means higher likelihood in
# what the current token is looking for and what other token can share. (think of them as vectors in a vector space with same origin)

B,T,C = 4,8,32
head_size = 16

dummy_data = torch.randn(B,T,C,dtype=torch.float)

# NOW we use Q,K,V to set relevance of tokens wrt current token
# when we initialize the token weights to 0 , then eventually every token gets equal priority eventually.

# Self Attention - Communication between tokens
# SELF ATTENTION HEAD


head_size = 16
# Relevance
Query = nn.Linear(C , head_size , bias=False)
Key = nn.Linear(C , head_size , bias=False)

# Retrival
Value = nn.Linear(C , head_size , bias=False) # C -> 16

q = Query(dummy_data)  # B,T,16
k = Key(dummy_data) # B,T,16


# No communication yet , Query and Key are just two liner layers, q and k are just transformation of C to 16 dimension

communicated_wts = (q@k.transpose(-1,-2))/ (T**0.5)   # THE -1 AND -2 ARE TO preserve batch dim , we transpose the T and 16(after linear layer) dims resulting in (B,T,T)  Note to self: Look at online sources to understand Channels more deeply
# Attention
print(communicated_wts.shape) # Relevance (B,T,T) -> (4,8,8)

# Masked self attention because we can't see future tokens
lower_traingular_matrix = torch.tril(torch.ones(T,T))
communicated_wts = torch.masked_fill(communicated_wts, lower_traingular_matrix == 0 , float('-inf'))
communicated_wts = torch.softmax(communicated_wts, dim = -1)
print(communicated_wts[0])

v = Value(dummy_data) # B,T,16 -> (4,8,16)
fetched_relevant_info = communicated_wts @ v # (4,8,8) @ (4,8,16) -> (4,8,16)
print(fetched_relevant_info.shape)

# Attention is just a communication mechanism , the vectors Do not have any positional information , we have to explicitly encode them positionally
# Note to self : Try looking at attention wrt weighted graphs
# In attention its just a set of vectors anywhere in space , THEY communicate with each other , if we want to add a position we have to add it to the vectors -> positional encoding

# Elements across the Batch dimension NEVER talk to each other and get processed independently and parellelly

# In decoder Masking is needed but in Encoder , masking is not needed and future tokens can communicate with each other , for tasks like Translation and Sentiment analysis
# In such cases encoder block is used , while GPT is a type of Decoder Only Transformer .

# Cross attention is not used here since there is No encoder with decoder . since we do not take any reference memory or context in any encoder block in the decoder.

# SCALED DOT PRODUCT ATTENTION , we also need to scale the communicated wts by taking the sqrt of head size otherwise the variance will explode (grow ) and result in VERY small values after softmax


