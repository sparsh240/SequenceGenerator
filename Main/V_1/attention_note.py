# We do not want the tokens ahead to be seen by the previous tokens
# we also want the tokens to be Aware of the previous tokens AND themselves but not the tokens ahead

# Simplest way to do so is to average the value of a token AND all the tokens before it.
# this destroys the information of order or position of the token in the sequence
import torch
# Dummy data
B,T,C = 4,8,2
dummy_data = torch.randint(0,10,(B,T,C) , dtype=torch.float)

# 1

print(dummy_data, sep="\n")

x_attention_wts = torch.zeros((B,T,C)) #
for b in range(B):
  for t in range(T):
    prev_vals = dummy_data[ b , 0:t+1 ].float() # means till token t of batch b ie from 0 to t of batch b
    x_attention_wts[b,t] = torch.mean(prev_vals , dim=0) # this makes Each token talk to its previous tokens by averaging the value of itself with its corresponding previous tokens



# This is inefficient and can be efficient with matrix multiplication
# torch.tril(matrix) returns just the lower triangular matrix

# if we take dot product of a matrix with a lower triangular matrix of 1s , we get the desired effect (since the 0s block off the previous elements)

# 2

avg_calculation_matrix = torch.tril(torch.ones((T,T) , dtype=torch.float))

# below we multiply a 2D matrix with a 3D matrix as pytorch adds the third dim to the 2D matrix
x_attention_wts = avg_calculation_matrix@dummy_data # matrix multiplication - will sum the current token and its previous token , but not give average
x_attention_wts /= torch.sum(avg_calculation_matrix , dim =1 , keepdim=True) # in 1 dim means in rows  , we devide with the avg row wise
# we csn fo this above too

print(x_attention_wts)


# With Softmax - Softmax converts an array to a bunch of probabilities where each value is between 0 and 1 and the sum of all values is 1
# Slight changes

''' BEST WAY FOR GETTING SELF ATTENTION AVG '''
# 3

calc_matrix = torch.tril(torch.ones(T,T)) # Initializeing to 0s now but will be input dependent
calc_matrix = torch.masked_fill(calc_matrix, calc_matrix == 0 , float('-inf')) # filling the upper trianguar matrix with -inf (- infinity)
calc_matrix = torch.softmax(calc_matrix , dim = 1) # dim=1 means ROW WISE
modifiet_wt_output = calc_matrix @ dummy_data



# Long story short , we can do weighted aggrigations of past elements via matrix multiplications of a lower triangular way , where the non
# -lower trianguar matrix is filled with zeros so as to not let the previous elements see the future elements

# This piece of code only introduces Maskeing and a Method of making the previous tokens affect the current token.
# But this method still has a major draw back , it treats all the previous tokens equally which is not true.
