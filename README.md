# llama 2

Objective:- Creating a quantized version of llama 2 LLM to run on local system 

Result:- If you have a GPU, this quantized veteion works. I have reduced the position to float16 and use 4bits. 
the same technique wont work in the case of CPU as intel cihopset wont support half-precision floating-point number format

Use GPU.py, If you wanna run your llama 2 llm in a low GPU config

Use main.py, if you wanna run your llama 2 llm in CPU. (You should need a very large RAM)
