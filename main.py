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
submit_button = st.button('authorize')

if submit_button:
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

from llama_index import download_loader,GPTSimpleVectorIndex

BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

loader = BeautifulSoupWebReader()

url = st.text_input("URL 🔗")

import webshot

# webshot.config.screenshot_path = '/path/to/save/screenshot'


if url:
    # img = webshot.url(url)
    # st.image(img)
    documents = loader.load_data(urls=[url])
else:
    st.write('please paste url') 


def get_text(prompt):
    input_text = st.text_input("You: ", prompt, key="input")
    return input_text


load_button = st.button('read')
user_input = get_text("この記事の要点を3つにまとめてください")


if load_button:
    try:
        ask_button = st.button('ask')
        index = GPTSimpleVectorIndex.from_documents(documents)
        
    except Exception as e:
        st.write("error reading the context: "+ str(e))
else:
    st.write("waiting for the website")
    index = ''


if ask_button:
    with st.spinner('typing...'):
        output = index.query(user_input)
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
