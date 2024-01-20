> 本文将会介绍大模型部署工具vLLM及其使用方法。

欢迎关注我的公众号**NLP奇幻之旅**，原创技术文章第一时间推送。

<center>
    <img src="https://s2.loli.net/2023/09/07/BFUl9i4872wWATx.jpg" style="width:200px;">
</center>

欢迎关注我的知识星球“**自然语言处理奇幻之旅**”，笔者正在努力构建自己的技术社区。

<center>
    <img src="https://s2.loli.net/2023/09/07/bYtEecQBfjRlUd1.jpg" style="width:200px;">
</center>

## 介绍与安装

`vLLM`是伯克利大学LMSYS组织开源的大语言模型高速推理框架，旨在极大地提升实时场景下的语言模型服务的吞吐与内存使用效率。`vLLM`是一个快速且易于使用的库，用于 LLM 推理和服务，可以和HuggingFace 无缝集成。vLLM利用了全新的注意力算法「PagedAttention」，有效地管理注意力键和值。

![](https://blog.vllm.ai/assets/logos/vllm-logo-text-light.png)

在吞吐量方面，vLLM的性能比HuggingFace Transformers(HF)高出 24 倍，文本生成推理（TGI）高出3.5倍。

![](https://blog.vllm.ai/assets/figures/perf_a100_n1_light.png)

安装命令：

> pip3 install vllm

本文使用的Python第三方模块的版本如下：

```bash
vllm==0.2.7
transformers==4.36.2
requests==2.31.0
gradio==4.14.0
```

## vLLM初步使用

### 线下批量推理

线下批量推理：为输入的prompts列表，使用vLLM生成答案

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from vllm import LLM, SamplingParams

llm = LLM('/data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf')
```

    INFO 01-18 08:13:26 llm_engine.py:70] Initializing an LLM engine with config: model='/data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf', tokenizer='/data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=None, enforce_eager=False, seed=0)
    INFO 01-18 08:13:37 llm_engine.py:275] # GPU blocks: 3418, # CPU blocks: 327
    INFO 01-18 08:13:39 model_runner.py:501] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
    INFO 01-18 08:13:39 model_runner.py:505] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode.
    INFO 01-18 08:13:44 model_runner.py:547] Graph capturing finished in 5 secs.



```python
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```


```python
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

    Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 11.76it/s]

    Prompt: 'Hello, my name is', Generated text: " Sherry and I'm a stay at home mom of three beautiful children."
    Prompt: 'The president of the United States is', Generated text: ' one of the most powerful people in the world, and yet, many people do'
    Prompt: 'The capital of France is', Generated text: ' Paris. This is a fact that is well known to most people, but there'
    Prompt: 'The future of AI is', Generated text: ' likely to be shaped by a combination of technological advancements and soci'

### API Server服务

vLLM可以部署为API服务，web框架使用`FastAPI`。API服务使用`AsyncLLMEngine`类来支持异步调用。

使用命令 `python -m vllm.entrypoints.api_server --help` 可查看支持的脚本参数。

API服务启动命令:

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.api_server --model /data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf
```

输入:

```bash
curl http://localhost:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
    }'
```

输出:

```
{
    "text": [
        "San Francisco is a city of neighborhoods, each with its own unique character and charm. Here are",
        "San Francisco is a city in California that is known for its iconic landmarks, vibrant",
        "San Francisco is a city of neighborhoods, each with its own unique character and charm. From the",
        "San Francisco is a city in California that is known for its vibrant culture, diverse neighborhoods"
    ]
}
```

### OpenAI风格的API服务

启动命令：

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server --model /data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf --served-model-name llama-2-13b-chat-hf
```

还可指定对话模板（chat-template）。

**查看模型**

```bash
curl http://localhost:8000/v1/models
```

输出：

```
{
  "object": "list",
  "data": [
    {
      "id": "llama-2-13b-chat-hf",
      "object": "model",
      "created": 1705568412,
      "owned_by": "vllm",
      "root": "llama-2-13b-chat-hf",
      "parent": null,
      "permission": [
        {
          "id": "modelperm-d7ca4aa0eee44eb4a50e37eba06e520d",
          "object": "model_permission",
          "created": 1705568412,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

**text completion**

输入：

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-2-13b-chat-hf",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq .
```

输出：

```
{
  "id": "cmpl-d1ba6b9f1551443e87d80258a3bedad1",
  "object": "text_completion",
  "created": 19687093,
  "model": "llama-2-13b-chat-hf",
  "choices": [
    {
      "index": 0,
      "text": " city that is known for its v",
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 12,
    "completion_tokens": 7
  }
}
```

**chat completion**

输入：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-2-13b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }' | jq .
```

输出：

```
{
  "id": "cmpl-94fc8bc170be4c29982a08aa6f01e298",
  "object": "chat.completion",
  "created": 19687353,
  "model": "llama-2-13b-chat-hf",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "  Hello! I'm happy to help! The Washington Nationals won the World Series in 2020. They defeated the Houston Astros in Game 7 of the series, which was played on October 30, 2020."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 40,
    "total_tokens": 95,
    "completion_tokens": 55
  }
}
```


## vLLM实战

### 大模型简单问答

vLLM暂不支持同时部署多个大模型，因此，笔者采用`一次部署一个模型，部署多次`的方法来实现部署多个大模型，这里采用`llama-2-13b-chat-hf`和`Baichuan2-13B-Chat`.

模型部署的命令如下：

```bash
CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 50072 --model /data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf --served-model-name llama-2-13b-chat-hf

CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 50073 --model /data-ai/model/baichuan2/Baichuan2-13B-Chat --served-model-name Baichuan2-13B-Chat --trust-remote-code --chat-template /data-ai/usr/code/template_baichuan.jinja
```

其中，`template_baichuan.jinja`（对话模板）采用vLLM在github官方网站中的examples文件夹下的同名文件。

使用Gradio来构建页面，主要实现大模型问答功能，Python代码如下：

```python
# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: gradio_for_llm.py
# @time: 2024/1/19 13:30
import gradio as gr
import requests

models = ['llama-2-13b-chat-hf', 'Baichuan2-13B-Chat']


def completion(question):
    model_url_dict = {models[0]: "http://localhost:50072/v1/chat/completions",
                      models[1]: "http://localhost:50073/v1/chat/completions",
                      }
    answers = []
    for model in models:
        headers = {'Content-Type': 'application/json'}

        json_data = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': question
                },
            ],
        }

        response = requests.post(model_url_dict[model], headers=headers, json=json_data)
        answer = response.json()["choices"][0]["message"]["content"]
        answers.append(answer)
    return answers


demo = gr.Interface(
    fn=completion,
    inputs=gr.Textbox(lines=5, placeholder="input your question", label="question"),
    outputs=[gr.Textbox(lines=5, placeholder="answer", label=models[0]),
             gr.Textbox(lines=5, placeholder="answer", label=models[1])]
)

demo.launch(server_name='0.0.0.0', share=True)
```

演示例子：

![大模型可视化问答](https://s2.loli.net/2024/01/19/zKZRNLJbhoCdwap.png)

### 大模型输出TPS

衡量大模型部署工具的指标之一为TPS（Token Per Second），即每秒模型输出的token数量。

我们以`llama-2-13b-chat-hf`，测试数据集参考网站中的问题集：[https://modal.com/docs/examples/vllm_inference](https://modal.com/docs/examples/vllm_inference) ，一共59个问题。

Python代码如下：

```python
# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: gradio_for_throughput.py
# @time: 2024/1/19 16:05
import gradio as gr
import requests
import time

questions = [
        # Coding questions
        "Implement a Python function to compute the Fibonacci numbers.",
        "Write a Rust function that performs binary exponentiation.",
        "How do I allocate memory in C?",
        "What are the differences between Javascript and Python?",
        "How do I find invalid indices in Postgres?",
        "How can you implement a LRU (Least Recently Used) cache in Python?",
        "What approach would you use to detect and prevent race conditions in a multithreaded application?",
        "Can you explain how a decision tree algorithm works in machine learning?",
        "How would you design a simple key-value store database from scratch?",
        "How do you handle deadlock situations in concurrent programming?",
        "What is the logic behind the A* search algorithm, and where is it used?",
        "How can you design an efficient autocomplete system?",
        "What approach would you take to design a secure session management system in a web application?",
        "How would you handle collision in a hash table?",
        "How can you implement a load balancer for a distributed system?",
        # Literature
        "What is the fable involving a fox and grapes?",
        "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
        "Who does Harry turn into a balloon?",
        "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
        "Describe a day in the life of a secret agent who's also a full-time parent.",
        "Create a story about a detective who can communicate with animals.",
        "What is the most unusual thing about living in a city floating in the clouds?",
        "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
        "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
        "Tell a story about a musician who discovers that their music has magical powers.",
        "In a world where people age backwards, describe the life of a 5-year-old man.",
        "Create a tale about a painter whose artwork comes to life every night.",
        "What happens when a poet's verses start to predict future events?",
        "Imagine a world where books can talk. How does a librarian handle them?",
        "Tell a story about an astronaut who discovered a planet populated by plants.",
        "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
        "Write a tale about a chef whose food can evoke memories from the eater's past.",
        # History
        "What were the major contributing factors to the fall of the Roman Empire?",
        "How did the invention of the printing press revolutionize European society?",
        "What are the effects of quantitative easing?",
        "How did the Greek philosophers influence economic thought in the ancient world?",
        "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
        "How did decolonization in the 20th century change the geopolitical map?",
        "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
        # Thoughtfulness
        "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
        "In a dystopian future where water is the most valuable commodity, how would society function?",
        "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
        "What could be the potential implications of contact with an advanced alien civilization?",
        # Math
        "What is the product of 9 and 8?",
        "If a train travels 120 kilometers in 2 hours, what is its average speed?",
        "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
        "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
        "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
        "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
        # Facts
        "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
        "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
        "What was Project A119 and what were its objectives?",
        "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
        "What is the 'Emu War' that took place in Australia in the 1930s?",
        "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
        "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
        "What are 'zombie stars' in the context of astronomy?",
        "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
        "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
    ]


def chat_completion(question):
    url = "http://localhost:50072/v1/chat/completions"

    headers = {'Content-Type': 'application/json'}

    json_data = {
        'model': "llama-2-13b-chat-hf",
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': question
            },
        ],
    }

    response = requests.post(url, headers=headers, json=json_data)
    answer = response.json()["choices"][0]["message"]["content"]
    output_tokens = response.json()["usage"]["completion_tokens"]
    return answer, output_tokens


def slowly_reverse(texts, progress=gr.Progress()):
    total_token_cnt = 0
    progress(0, desc="starting...")
    q_list = texts.split('\n')
    s_time = time.time()
    data_list = []
    for q in progress.tqdm(q_list, desc=f"generating..."):
        answer, output_token = chat_completion(q)
        total_token_cnt += output_token
        data_list.append([q, answer[:50], total_token_cnt/(time.time() - s_time)])
        print(f"{total_token_cnt/(time.time() - s_time)} TPS")

    return data_list


demo = gr.Interface(
    fn=slowly_reverse,
    # 自定义输入框
    inputs=gr.Textbox(value='\n'.join(questions), label="questions"),
    # 设置输出组件
    outputs=gr.DataFrame(label='Table', headers=['question', 'answer', 'TPS'], interactive=True, wrap=True)
)

demo.queue().launch(server_name='0.0.0.0', share=True)

```

输出的TPS统计如下：

![vLLM部署大模型的吞吐量的简单实验](https://s2.loli.net/2024/01/19/7a9Um6EzKXhtkpS.png)

本次实验共耗时约639秒，最终的TPS为49.4。

以上仅是TPS指标的一个演示例子，事实上，vLLM部署LLAMA-2模型的TPS应该远高于这个数值，这与我们使用vLLM的方式有关，比如GPU数量，worker数量，客户端请求方式等，这些都是影响因素，待笔者后续更新。


## 总结

本文介绍了大模型部署工具vLLM，并给出了其三种不同的部署方式，在文章最后，介绍了笔者对于vLLM的实战。后续，笔者将会对vLLM的推理效率进行深入的实验。

感谢阅读~

## 参考文献

1. vLLM documentation: [https://docs.vllm.ai/en/latest/index.html](https://docs.vllm.ai/en/latest/index.html)
2. VLLM推理流程梳理: [https://blog.csdn.net/just_sort/article/details/132115735](https://blog.csdn.net/just_sort/article/details/132115735)
3. vllm github: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
4. modal vllm inference: [https://modal.com/docs/examples/vllm_inference](https://modal.com/docs/examples/vllm_inference)