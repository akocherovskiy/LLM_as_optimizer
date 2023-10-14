import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",
    device_map=device,
    torch_dtype=torch.bfloat16)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

x = np.arange(0, 6, 0.5) # create true values for x
y = 3*x + np.random.randint(-1, 2, 12) # create true values for y + noise

# initialize random weights for the linear function y = w*x + b - equation for our line
# during optimization, we will change the weights w, b, calculate the resulting "y" and compare them with the true values "y"
w = np.random.uniform(-5, 5) 
b = np.random.uniform(-5, 5)

def loss_calc(y, w, x, b):
    return ((y - w*x + b)**2).mean() # mean squared error loss function

loss = loss_calc(y, w, x, b)

d = {'loss': [loss], 'w': [w], 'b': [b]}
loss_list = [loss] # collect all losses for plotting at the end

df = pd.DataFrame(data=d) # dataset to store all the proposed weights (w, b) and calculated loss
df.sort_values(by=['loss'], ascending=False, inplace=True)

def is_number_isdigit(s): # function for parsing str response from LLM
    n1 = s[0].replace('.','',1).replace('-','',1).strip().isdigit()
    n2 = s[1].replace('.','',1).replace('-','',1).strip().isdigit()
    return n1 * n2

def check_last_solutions(loss_list, last_nums): # function that stops optimization when the last 4 values of the loss function < 1
    if len(loss_list) >= last_nums:
        last = loss_list[-last_nums:]
        return all(num < 1 for num in last)

def create_prompt(num_sol): # create prompt
    meta_prompt_start = f'''Now you will help me minimize a function with two input variables w, b. I have some (w, b) pairs and the function values at those points.
The pairs are arranged in descending order based on their function values, where lower values are better.\n\n'''

    solutions = ''
    if num_sol > len(df.loss):
        num_sol = len(df.loss)

    for i in range(num_sol):
        solutions += f'''input:\nw={df.w.iloc[-num_sol + i]:.3f}, b={df.b.iloc[-num_sol + i]:.3f}\nvalue:\n{df.loss.iloc[-num_sol + i]:.3f}\n\n''' 
    
    meta_prompt_end = f'''Give me a new (w, b) pair that is different from all pairs above, and has a function value lower than
any of the above. Do not write code. The output must end with a pair [w, b], where w and b are numerical values.

w, b ='''

    return meta_prompt_start + solutions + meta_prompt_end

num_solutions = 10 # number of observations to feed into the prompt

for i in range(500):
    
    text = create_prompt(num_solutions)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    model.to(device)

    generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=15,
            temperature=0.8,
            do_sample=True,
            pad_token_id=50256
            )

    output = tokenizer.batch_decode(generated_ids)[0]
    response = output.split("w, b =")[1].strip()
    
    if "\n" in response:
        response = response.split("\n")[0].strip()

    if "," in response:
        numbers = response.split(',')
    
    if is_number_isdigit(numbers):
        w, b = float(numbers[0].strip()), float(numbers[1].strip())
        loss = loss_calc(y, w, x, b)
        loss_list.append(loss)
        new_row = {'loss': loss, 'w': w, 'b': b}
        new_row_df = pd.DataFrame(new_row, index=[0])
        df = pd.concat([df, new_row_df], ignore_index=True)
        df.sort_values(by='loss', ascending=False, inplace=True)

    if i % 20 == 0:
        print(f'{w=} {b=} loss={loss:.3f}')

    if check_last_solutions(loss_list, 3):
        break

print(*loss_list[-15:], sep='\n')
