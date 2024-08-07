--- 
<div align="center">    
  
# `KG2H`: Enhancing Knowledge-Augmented Prompting for Question Answering via Optimizing Knowledge Graph Description 

</div>

This paper is currently undergoing a review process. It will be released and the code will be made available once the review has been completed. Some of our models and data are available for use now via [Huggingface](https://huggingface.co/collections/CoShin/kg2h-66a88f1959d861260025241d).



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
