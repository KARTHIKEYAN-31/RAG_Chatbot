import streamlit as st 
import app_functrions as func



@st.experimental_fragment
def clear_chat():
    st.session_state.messages = []

@st.experimental_fragment
def init_chat():
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = func.call_chat_api(prompt)
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    # return 

st.set_page_config(
    page_title="Chat-Bot",
    page_icon="🤖",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_name" not in st.session_state:
    st.session_state.file_name = ""

 

st.title("Chat with Data")

st.sidebar.header("File Manager")

if st.sidebar.button("Clear Chat"):
    clear_chat()   



file = st.sidebar.file_uploader("Upload a file to Chat with", type=["csv", "txt", "pdf"])


if file is not None:
    api_output = func.call_file_api(file)
    
    st.sidebar.write(api_output["status"])
    st.session_state.file_name = api_output["file_name"]

    if api_output["status"] == "Success":

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Come on lets Chat!"):
            init_chat()


        
    


