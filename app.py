import streamlit as st
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

from PyPDF2 import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
import base64
from htmlTemplates import expander_css, css, bot_template, user_template


def main():
    st.set_page_config(
    page_title="Basic Chatbot",
    page_icon=':coffee:',
    layout='wide'
    )
    st.title("Basic Chatbot")
    load_dotenv(override=True)
    # Access the OpenAI API key
    openai_key= os.environ["OPENAI_API_KEY"]
    # Initialize the OpenAI client with the API key
    client = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_key)
    chain = ConversationChain(llm=client, verbose=False)
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = client
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "num_page" not in st.session_state:
        st.session_state.num_page = 0
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your questions here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        result = chain.run({"input": prompt})
        response = result

        # Display the assistant's response in the chat
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
if __name__ == '__main__':
    main()
