from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 设置具体LLM位置
model_dir = "/home/virsh/ai/git/pytorch/hello_world/AI_MODEL/Qwen2.5-1.5B"

# 家在模型和分词器
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 使用家在的模型和分瓷器创建生成文本的 pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# 生成文本
output = generator("你好，你是谁", max_new_tokens=50, num_return_sequences=1, truncation=True)


print(output)