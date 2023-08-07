# !pip install -qU \
#     transformers==4.31.0 \
#     accelerate==0.21.0 \
#     einops==0.6.1 \
#     langchain==0.0.240 \
#     xformers==0.0.20 \
#     bitsandbytes==0.41.0
# !pip install accelerate

############################################
# RUN LLAMA ON CPU
############################################

import torch
import transformers
from transformers import AutoTokenizer
from torch import bfloat16, cuda

device = 'cpu'
model_name = "meta-llama/Llama-2-7b-chat-hf"
hf_auth_key = "<Your token comes here>"
torch.set_default_dtype(bfloat16)
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
    llm_int8_enable_fp32_cpu_offload=True,
    llm_int8_has_fp16_weight=True
)

model_config = transformers.AutoConfig.from_pretrained(model_name, use_auth_token=hf_auth_key)
model_config.torch_type = bfloat16
device_map = {
  "transformer.word_embeddings": "cpu",
  "transformer.word_embeddings_layernorm": "cpu",
  "lm_head": "cpu",
  "transformer.h": "cpu",
  "transformer.ln_f": "cpu",
  "model.embed_tokens": "cpu",
  "model.layers":"cpu",
  "model.norm":"cpu"
}
model = model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map=device_map,
    use_auth_token=hf_auth_key,
    torch_dtype=bfloat16
)
model.eval()
print(f"Model running in {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name,
    use_auth_token=hf_auth_key
)
# # Convert the model to float32
# model.half()
generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=0.0,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # mex number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
    torch_dtype=bfloat16
)
res = generate_text("Explain to me the difference between nuclear fission and fusion.")
print(res[0]["generated_text"])
