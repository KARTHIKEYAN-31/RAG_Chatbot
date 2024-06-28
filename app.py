import os
import streamlit as st
import function as func





@st.experimental_fragment
# @st.cache_data
def upload_file():
    file = st.file_uploader("Upload a file to Chat with", type=["csv", "txt", "pdf"])
    if file is not None:
        db, k = func.get_db(file)
        st.session_state.db = db
        st.session_state.k = k
        st.toast("File uploaded successfully")

@st.experimental_fragment
def init_chat():
    cont = st.container(height=380)
    with cont:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    if prompt := st.chat_input("Come on lets Chat!"):
        cont.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = func.chat_with_doc(prompt, st.session_state.db, st.session_state.k)
        with cont.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})




st.set_page_config(
    page_title="Chat-Bot",
    page_icon="🤖",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "db" not in st.session_state:
    st.session_state.db = None
if "k" not in st.session_state:
    st.session_state.k = 1

c1,c2 = st.columns((4,1))
with c1:
    st.title("Chat with Data")
with c2:
    if st.button("Clear Chat"):
        st.session_state.messages = []

st.sidebar.header("File Manager")

with st.sidebar.expander("File Uploader", expanded=False, icon="📁"):
    upload_file()

if st.session_state.db != None:
    init_chat()
