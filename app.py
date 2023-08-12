from langchain.llms import CTransformers
import streamlit as st
from streamlit.components.v1 import html
import streamlit.components.v1 as components
from streamlit_chat import message
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import StreamlitCallbackHandler

st.title("LLAMA2-GPT")

history=[]


# prompt special tokens
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# Default Sys Prompt
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."



with st.sidebar:
    model_name=st.selectbox("Select Model :-",['Llama 7B','Llama 13B'])
    temperature=st.slider("Temperature :-",0.0,1.0,0.1)
    max_token=st.slider("Max token :-",0,2000,1024)
    top_p=st.slider("top_p :-",0.0,1.0,0.95)
    top_k=st.slider("top_k :- ",0,100,50)
    DEFAULT_SYSTEM_PROMPT=st.text_area("System Prompt :-",f"{DEFAULT_SYSTEM_PROMPT}",height=400)

# Load the selected model
if model_name=="Llama 7B":
    print("Llama 7B model Loading")
    model_path='models/llama-2-7b-chat.ggmlv3.q8_0.bin'
else:
    print("Llama 13B model Loading")
    model_path="models/llama-2-13b-chat.ggmlv3.q2_K.bin"


# create the custom prompt
def get_prompt(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> str:
    texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    for user_input, response in chat_history:
        texts.append(f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ")
    texts.append(f"{message.strip()} [/INST]")
    return "".join(texts)

## Load the Local Llama 2 model
def llama_model(model_path,max_new_tokens=1024,temperature=0.7,top_p=0.95,top_k=50):
    llm = CTransformers(
        model = model_path, 
        model_type="llama",
        max_new_tokens =max_new_tokens,
        temperature = temperature,
        top_p=top_p,
        top_k=top_k,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )
    return llm

print(f"{model_name} Model Loading start")
model=llama_model(model_path=model_path,max_new_tokens=max_token,temperature=temperature)
print(f"{model_name}Load Model Successfully.")



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Send a message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            final_prompt=get_prompt(prompt,history,DEFAULT_SYSTEM_PROMPT)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in model.predict(final_prompt):
                full_response += response
                message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )