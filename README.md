# llama 2

Objective:- Creating a quantized version of llama 2 LLM to run on the local system 

![llama](https://venturebeat.com/wp-content/uploads/2023/07/cfr0z3n_vector_art_cybernetic_llama_wearing_sunglasses_synthwav_d3f82260-2c47-4abd-9599-b91751711f5b.png?fit=1456%2C816&strip=all){:width="100"}

Result:- If you have a GPU, this quantized version works. I have reduced the precision to float16 and use of load_in_4bits=True. The load_in_4bits from BitsAndBytes helps to store the model weights in 4bits instead of 32bits which can significantly reduce the model's memory footprint. 


Intel chipsets is not supporting half-precision floating-point number format. Hence we can't use float16 precision. This resulting a need for a larger ram to run the model compared to the GPU version

> Use GPU.py, If you want to run your llama 2 llm in a low GPU config

> Use main.py, if you wanna run your llama 2 llm in CPU. (You should need a very large RAM)


Make sure to install the below dependencies before running the code
```sh
transformers==4.31.0
accelerate==0.21.0 
einops==0.6.1
xformers==0.0.20 
bitsandbytes==0.41.0
accelerate
```
