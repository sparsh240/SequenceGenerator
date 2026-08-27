with open("./Data/data.txt" , "r") as f:
  string_text = f.read()

vocabulary = sorted(list(set(string_text)))

str_to_int = {char : integer for integer , char in enumerate(vocabulary)}
int_to_str = {integer : char for integer , char in enumerate(vocabulary)}

encode = lambda text : [str_to_int[char] for char in text]
decode = lambda integers : "".join([int_to_str[integer] for integer in integers])
