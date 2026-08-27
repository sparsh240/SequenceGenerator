from tokenizer import encode
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'; # for utilizing nvidia GPU if available


# Tokenize the dataset
with open("./Data/data.txt","r",encoding='utf-8') as f:
    raw_text = f.read()

# Encoding the dataset
tokenized_data = encode(raw_text)
tokens = torch.tensor(tokenized_data,dtype = torch.long)# long , needed in F.cross_entropy

# Train , Validation split

split = int(0.9 * len(tokenized_data)) # 90% train, 10% val (convert to int if float)
train_data = tokens[:split]
val_data = tokens[split:]

print(f"Training tokens: {len(train_data)}")
print(f"Validation tokens: {len(val_data)}")


# Training on chunks of dataset
max_context_length = 8; # means the length of context will not go beyond 8 tokens
print(train_data[:max_context_length+1]) # we took 9 tokens from the training data (that contain a total of 8 examples in 1 chunk)

# when we take a chunk out , we dont just have a single sample, since it is sequencial data , every next token is predicted off the previous token
# for ex: chunk(1,2,3,4,5,6,7,8,9) then sample1 = ([1] -> 2) here context is token 1 and target is token 2, sample2 = ([1,2] -> 3) here context is token 1,2 and target is token 3, sample3 = ([1,2,3] -> 4) here context is token 1,2,3 and target is token 4 ...


# THIS is done to make the transformer be used to see context as little as 1 to max_context_length or block_size (in this case 8 tokens)
# for ex: sample1 = ([1] -> 2), sample2 = ([1,2] -> 3), ..., sample8 = ([1,2,3,4,5,6,7,8] -> 9)

# we call this the "time" dimension

# second dimension we need to deal with is batch dimension (we just stack the chunks together) - only for efficiency to Utilize GPUs parellel processing power , processing each chunk parellelly , saving time.


batch_size = 32 # how many chunks we want to process in parallel (in one go)
context_length = 8 # how many max tokens to consider as context (same as block_size)

# Batches
def get_batch(mode):
    data = train_data if mode == "train" else val_data

    # Generating Random Starting positions for each batch
    ix = torch.randint(len(data) - context_length, (batch_size,))

    # generating the batch
    x = torch.stack([data[i:i + context_length] for i in ix]) # stacks the contexts row wise
    y = torch.stack([data[i+1:i+context_length + 1] for i in ix]) # stacks the targets row wise (target for every ending index in context)

    # using CUDA
    x = x.to(device)
    y = y.to(device)

    return x,y # y will be used for backprop and loss calculations



x,y = get_batch("train")
print("x shape",x.shape)
print("y shape",y.shape)
print("contexts:\n",x)
print("targets:\n",y)

# for ex:
print(f"input:[{x[0,0]} , {x[0][1]}]")
print(f"target: {y[0][1]}")