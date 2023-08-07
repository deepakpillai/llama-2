# llama 2

Objective:- Creating a quantized version of llama 2 LLM to run on the local system 

Result:- If you have a GPU, this quantized version works. I have reduced the precision to float16 and use of load_in_4bits=True. The load_in_4bits from BitsAndBytes helps to store the model weights in 4bits instead of 32bits which can significantly reduce the model's memory footprint. 


Intel chipsets is not supporting half-precision floating-point number format. Hence we can't use float16 precision. This resulting a need for a larger ram to run the model compared to the GPU version

Use GPU.py, If you want to run your llama 2 llm in a low GPU config

Use main.py, if you wanna run your llama 2 llm in CPU. (You should need a very large RAM)
