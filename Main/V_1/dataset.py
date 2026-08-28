from tokenizer import encode
import torch

device = "cuda" if torch.cuda.is_available() else "cpu";
batch_size = 4;
context_size = 8;

with open("./Data/data.txt","r") as f:
  raw_data = f.read();

tokenized_data = torch.tensor(encode(raw_data),dtype=torch.long);
dataset_length = len(tokenized_data)


train_data = tokenized_data[: int(0.85 *dataset_length)];
val_data = tokenized_data[int(0.85 *dataset_length):];

def retrive_batch(mode):
  data = train_data if mode == "train" else val_data;
  starting_points = torch.randint(len(data) - context_size , (batch_size,))

  x = torch.stack([data[ i : i + context_size ] for i in starting_points]).to(device)
  y = torch.stack([data[ i + 1 : i + context_size + 1] for i in starting_points]).to(device)

  return x , y

x,y = retrive_batch('train');




