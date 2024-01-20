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
