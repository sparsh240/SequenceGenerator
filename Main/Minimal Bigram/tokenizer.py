
# Accessing Raw Data
with open('./Data/data.txt','r', encoding='utf-8') as f: #Note: Current working dir is project home
    raw_text = f.read()
    print("Read Input Data")

# Vocabulary
# sorted() automatically converts set into list and sorts the letters according to their encoding value (ASCII)
# By using set() we get the unique characters from the text
vocab = sorted(set(raw_text))
print('Vocab Size: ', len(vocab))



# Small model CAN be a charecter level language model , converting indevidual charecters as integers.
# Larger models use word or sub-word tokenization for better efficiency.

# Tokenizer - here We create a simple charecter level tokenizer where each charecter maps to an integer(corresponding index in vocabulary)

# Encoder maps characters to integers
# Decoder maps integers to characters

# Tokenizer

# enumerate returns index and value for each item in the ITERABLE
stoi = {ch:i for i, ch in enumerate(vocab)} # string to integer look up table
itos = {i:ch for i, ch in enumerate(vocab)} # integer to string look up table

encode = lambda s: [stoi[c] for c in s] # string input
decode = lambda l: "".join([itos[i] for i in l]) # list input


print(encode("Hello"))
print(decode(encode("Hello")))




