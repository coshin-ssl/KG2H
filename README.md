--- 
<div align="center">    
  
# `KG2H`: Enhancing Knowledge-Augmented Prompting for Question Answering   via Knowledge Graph-to-Text Optimization  

</div>

# Models and Datas
Our trained models and experimental data are available on the huggingface hub.

<pre>
<code>
from transformers import AutoModelForCausalLM, AutoTokenizer
hint_generator = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(hint_generator.config._name_or_path)
  
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'  

</code>
</pre>

# Template

<pre>
<code>
  
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

</code>
</pre>
