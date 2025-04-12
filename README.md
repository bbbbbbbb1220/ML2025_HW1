# ML2025_HW1

# ML2025 Homework 1: Retrieval-Augmented Generation with Agents

This project is the first homework for NTU's "Machine Learning 2025" course. The goal is to implement a Retrieval-Augmented Generation (RAG) system using the quantized **Meta-LLaMA 3.1 8B** model along with LLM-based agents for solving open-domain question-answering tasks.

---

## Environment Setup

### 1. Install Dependencies
```bash
python3 -m pip install --no-cache-dir llama-cpp-python==0.3.4 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
python3 -m pip install googlesearch-python bs4 charset-normalizer requests-html lxml_html_clean
```

### 2. Download the Model & Dataset
```python
from pathlib import Path
if not Path('./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf').exists():
    !wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
if not Path('./public.txt').exists():
    !wget https://www.csie.ntu.edu.tw/~ulin/public.txt
if not Path('./private.txt').exists():
    !wget https://www.csie.ntu.edu.tw/~ulin/private.txt
```

### 3. Check GPU Runtime
```python
import torch
if not torch.cuda.is_available():
    raise Exception('You are not using the GPU runtime. Change it first or you will suffer from the super slow inference speed!')
else:
    print('You are good to go!')
```

---

## Model & Inference Utility

```python
from llama_cpp import Llama

llama3 = Llama(
    "./Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    verbose=False,
    n_gpu_layers=-1,
    n_ctx=16384,
)

def generate_response(_model: Llama, _messages: str) -> str:
    _output = _model.create_chat_completion(
        _messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=512,
        temperature=0,
        repeat_penalty=2.0,
    )["choices"][0]["message"]["content"]
    return _output
```

---

## Search Tool (Google)

```python
from typing import List
from googlesearch import search as _search
from bs4 import BeautifulSoup
from charset_normalizer import detect
import asyncio
from requests_html import AsyncHTMLSession
import urllib3
urllib3.disable_warnings()

async def worker(s:AsyncHTMLSession, url:str):
    try:
        header_response = await asyncio.wait_for(s.head(url, verify=False), timeout=10)
        if 'text/html' not in header_response.headers.get('Content-Type', ''):
            return None
        r = await asyncio.wait_for(s.get(url, verify=False), timeout=10)
        return r.text
    except:
        return None

async def get_htmls(urls):
    session = AsyncHTMLSession()
    tasks = (worker(session, url) for url in urls)
    return await asyncio.gather(*tasks)

async def search(keyword: str, n_results: int=3) -> List[str]:
    keyword = keyword[:100]
    results = list(_search(keyword, n_results * 2, lang="zh", unique=True))
    results = await get_htmls(results)
    results = [x for x in results if x is not None]
    results = [BeautifulSoup(x, 'html.parser') for x in results]
    results = [''.join(x.get_text().split()) for x in results if detect(x.encode()).get('encoding') == 'utf-8']
    return results[:n_results]
```

---

## LLM Agent Interface

```python
class LLMAgent():
    def __init__(self, role_description: str, task_description: str, llm:str="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"):
        self.role_description = role_description
        self.task_description = task_description
        self.llm = llm

    def inference(self, message:str) -> str:
        if self.llm == 'bartowski/Meta-Llama-3.1-8B-Instruct-GGUF':
            messages = [
                {"role": "system", "content": f"{self.role_description}"},
                {"role": "user", "content": f"{self.task_description}\n{message}"},
            ]
            return generate_response(llama3, messages)
        else:
            return ""
```

---

## Agent Setup (Example)

```python
question_extraction_agent = LLMAgent(
    role_description="You are a task simplification assistant that helps reduce verbosity in user queries.",
    task_description="Extract the core question and remove any irrelevant background or unnecessary wording."
)

keyword_extraction_agent = LLMAgent(
    role_description="You are an expert in search engine optimization and keyword extraction.",
    task_description="Please extract the most suitable Traditional Chinese keywords from the question below for Google search."
)

qa_agent = LLMAgent(
    role_description="You are LLaMA-3.1-8B, an assistant trained to answer questions in Traditional Chinese only.",
    task_description="Please answer the following question:"
)
```

---

## RAG Pipeline (Outline)

1. Use `question_extraction_agent` to clarify the question.
2. Use `keyword_extraction_agent` to extract search queries.
3. Retrieve related contents using `search()`.
4. Feed the original question and search result context to `qa_agent`.
5. Return the answer.

---

## Example Test

```python
test_question='請問誰是 Taylor Swift？'

messages = [
    {"role": "system", "content": "你是 LLaMA-3.1-8B，是用來回答問題的 AI。使用中文時只會使用繁體中文來回問題。"},
    {"role": "user", "content": test_question},
]

print(generate_response(llama3, messages))
```

---

##  Project Structure

```
.
├── Meta-Llama-3.1-8B-Instruct-Q8_0.gguf     # LLM model weights
├── public.txt / private.txt                # Sample questions
├── [your_code.py / notebook.ipynb]         # Implementation
├── README.md                               # Project documentation
```


