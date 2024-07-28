--- 
<div align="center">    
  
# `KG2H`: Enhancing Knowledge-Augmented Prompting for Question Answering   via Knowledge Graph-to-Text Optimization  

</div>

This paper is being reviewed. We will release it and the code after it is reviewed. Some of our models and data are available now via Huggingface.


# Models and Datasets
## Models
Our trained models and experimental data are available on the huggingface hub.
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
hint_generator = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(hint_generator.config._name_or_path)
  
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'  
```
## Datasets
```python
from datasets import load_dataset
dataset = load_dataset(DATASET_NAME)
```
# Instruction Templates
```python
def hint_paragraph_prompting(question, linearized_triple):
    res = (
        f'### Please create a short hint paragraph to answer the question reorganizing the triple information, step by step.\n'
        f'### Question: {question}\n'
        f'### Triple Information: {linearized_triple}\n'
        f'### Hint Paragraph:'
    )
    return res

def answer_prompting(question, hint):
    hint = hint.replace('\n', ' ')
    res = (
        f'### Below are the facts that might be relevant to answer the question. Please provide a short answer(1-3 words in English) to the following question.\n'
        f'### Facts: {hint.strip()}\n'
        f'### Question: {question.strip()}\n'
        f'### Answer:'
    )
    return res
```
