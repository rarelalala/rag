import json
import time
import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

import chatmodel

st.set_page_config(page_title="LLM-RAG-WEB")
st.title("LLM-RAG-WEB")


@st.cache_resource
def init_model():
    model_path = "../models/Baichuan2-7B-Chat"
    print('model_path=', model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    return model, tokenizer


def clear_model_history():
    del st.session_state.messages
    # st.session_state.history1 = [st.session_state.history1[0]]  # 保留初始记录
    # placeholder.empty()


def clear_chat_history():
    del st.session_state.messages
    # st.session_state.history2 = [st.session_state.history2[0]]


def init_model_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是AI助手，很高兴为您服务🥰")

    if "messages1" in st.session_state:
        for message in st.session_state.messages1:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages1 = []

    return st.session_state.messages1


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是AI助手，很高兴为您服务🥰")

    if "messages2" in st.session_state:
        for message in st.session_state.messages2:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages2 = []

    return st.session_state.messages2


def main():
    # 初始化模型
    model, tokenizer = init_model()

    # 创建侧边栏布局
    sidebar_selection = st.sidebar.selectbox("选择对话类型", ("模型对话", "文件对话"))

    if sidebar_selection == "模型对话":
        messages1 = init_model_history()
        # print("history1:", st.session_state.history1)
        if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            messages1.append({"role": "user", "content": prompt})
            print(f"[user] {prompt}", flush=True)
            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                # st.session_state.history1.append(["Human", prompt])
                # st.session_state.history1.append(["Assistant", None])
                # print("history1:", st.session_state.history1)
                start = time.time()

                for response in model.chat(tokenizer, messages1, stream=True):
                    placeholder.markdown(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

                end = time.time()
                cost = end - start
                length = len(response)
                print(f"{length / cost}tokens/s")
                # print(prompt,response)
                # st.session_state.history1[-1][1] = response

            messages1.append({"role": "assistant", "content": response})
            print(json.dumps(messages1, ensure_ascii=False), flush=True)

            st.button("清空对话", on_click=clear_model_history)

    elif sidebar_selection == "文件对话":
        messages2 = init_chat_history()
        # print("history2:", st.session_state.history2)

        if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            sim_result = chatmodel.get_docs(prompt)
            new_prompt = f"""你是一个智能助手，需要根据下述信息以条理清晰的方式回应查询。考虑到你的主要服务对象是老年群体，请确保回答内容易于理解且通俗易懂。
    已知信息：  
    {sim_result} \n
    问题：
    {prompt}  \n"""
            # new_prompt =f"""基于以下已知信息，简洁和专业的来回答用户的问题。

            #                 已知内容:
            #                 {sim_result}
            #                 问题:{prompt}"""
            messages2.append({"role": "user", "content": new_prompt})
            print(f"[user] {new_prompt}", flush=True)
            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                # st.session_state.history2.append(["Human", new_prompt])
                # st.session_state.history2.append(["Assistant", None])
                # print("history2:", st.session_state.history2)
                start = time.time()

                for response in model.chat(tokenizer, messages2, stream=True):
                    placeholder.markdown(response)
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                end = time.time()

                cost = end - start
                length = len(response)
                print(f"{length / cost}tokens/s")
                # print(prompt,response[(len(prompt)+1):])
                # st.session_state.history2[-1][1] = response

            messages2.append({"role": "assistant", "content": response})
            print(json.dumps(messages2, ensure_ascii=False), flush=True)

            st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
