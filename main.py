"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os

from langchain.chains import ConversationChain
from langchain.llms import OpenAI


st.set_page_config(page_title="ChatURL", page_icon=":robot:")
st.header("▶️ ChatURL")
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


api_token = st.text_input('OpenAI API Token',type="password")
st.session_state['authorized'] = st.button('authorize')

if st.session_state['authorized']:
    if api_token:
        os.environ['OPENAI_API_KEY'] = api_token
        st.write('authorized.')
    else:
        st.write('Please input a valid API token.')
else:
    st.write('Waiting for API token...')

def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain

if os.environ['OPENAI_API_KEY']!="":
    try:
        chain = load_chain()
    except Exception as e:
        st.write("error loading data: " + str(e))
else:
    st.write("waiting for api token...")

from llama_index import download_loader

BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

loader = BeautifulSoupWebReader()

url = st.text_input("URL 🔗")

import webshot

# webshot.config.screenshot_path = '/path/to/save/screenshot'
ask_button = False

def get_text(prompt):
    input_text = st.text_input("You: ", prompt, key="input")
    return input_text


user_input = get_text("この記事の要点を3つにまとめてください")
additional_prompt = "あなたは優秀な解説者です。丁寧かつフレンドリー、わかりやすい言葉で受け答えしてください。回答は相手の言葉と同じものでお願いします。"

from llama_index import GPTPineconeIndex, ServiceContext, PromptHelper

max_input_size = 3000
num_output = 1000
chunk_size_limit = 1000
max_chunk_overlap = 20

prompt_helper = PromptHelper(
    max_input_size=max_input_size,
    num_output=num_output,
    max_chunk_overlap=max_chunk_overlap,
    chunk_size_limit=chunk_size_limit
)

service_context = ServiceContext.from_defaults(prompt_helper=prompt_helper)

import pinecone
pinecone.delete_index("chaturl")
# Pinecone
api_key = os.environ['PINECONE_API_KEY']
pinecone.init(api_key=api_key, environment="us-east1-gcp")
pinecone.create_index("chaturl", dimension=1536, metric="euclidean", pod_type="p1")
pinecone_index = pinecone.Index("chaturl")

if url:
    # img = webshot.url(url)
    # st.image(img)
    documents = loader.load_data(urls=[url])
    ask_button = st.button('ask')
    index = GPTPineconeIndex.from_documents(
        documents, 
        pinecone_index=pinecone_index,
        service_context=service_context)
else:
    st.write('please paste url') 


if ask_button:
    with st.spinner('typing...'):
        output = index.query(additional_prompt + user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output.response)
else:
    pass


if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        try:
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        except:
            pass
