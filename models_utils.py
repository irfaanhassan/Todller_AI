from transformers import AutoTokenizer, AutoModelForCausalLM

# Load GPT-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def explain_animal(animal_name):
    prompt = f"Explain what a {animal_name} is in simple, cheerful sentences for a 5-year-old kid."

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return explanation
