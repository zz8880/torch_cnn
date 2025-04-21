from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B"
cache_dir = "/home/virsh/ai/llm"

model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)