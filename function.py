from huggingface_hub import HfApi
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
import tempfile
from langchain_community.vectorstores import SKLearnVectorStore
import pandas as pd
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import tempfile
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader
import re




HF_key = os.environ.get("HF_TOKEN")
HFapi = HfApi(HF_key)


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)
persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")


def get_text_from_pdf(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    num = len(pages)
    for page in pages:
        if page.page_content.count('/g') > 3:
            page.page_content = decode(page.page_content)
    return pages, num


def get_text_from_csv(file, key_column):
    df = pd.read_csv(file)
    loader = DataFrameLoader(data_frame=df, page_content_column=key_column)
    pages = loader.load()
    num = len(pages)
    return pages, num


def get_text_from_txt(file):
    text_documents = TextLoader(file).load()
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(text_documents)
    num = len(text_chunks)
    return text_chunks, num

def get_temp_file_path(file):
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file.name)
    with open(path, "wb") as f:
        f.write(file.read())
    return path

def cidToChar(cidx):
    return chr(int(re.findall(r'\/g(\d+)',cidx)[0]) + 29)

def decode(sentence):
  sen = ''
  for x in sentence.split('\n'):
    if x != '' and x != '/g3':         # merely to compact the output
      abc = re.findall(r'\/g\d+',x)
      if len(abc) > 0:
          for cid in abc: x=x.replace(cid, cidToChar(cid))
      sen += repr(x).strip("'")

  return re.sub(r'\s+', ' ', sen)

def get_db(file):
    file_path = get_temp_file_path(file)
    file_extension = os.path.splitext(file.name)[1]
    if file_extension == ".pdf":
        doc, num = get_text_from_pdf(file_path)
        db = SKLearnVectorStore.from_documents(
                documents=doc,
                embedding=embeddings,
                persist_path=persist_path,  
                serializer="parquet",
            )
        return db, num
    elif file_extension == ".txt":
        doc, num = get_text_from_txt(file_path)
        db = SKLearnVectorStore.from_documents(
                documents=doc,
                embedding=embeddings,
                persist_path=persist_path,  # persist_path and serializer are optional
            )
        return db, num
    elif file_extension == ".csv":
        doc, num = get_text_from_csv(file_path)
        db = SKLearnVectorStore.from_documents(
                documents=doc,
                embedding=embeddings,
                persist_path=persist_path,  # persist_path and serializer are optional
            )
        return db, num
    else:
        return None
    



def get_llm_chain(llm, db, k):
    if k > 4:
        k = 4

    # set the prompt templte
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. Give the answer in the form of markdown.
    Context: {context}
    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # create memory and llm-chain
    memory = ConversationBufferMemory(
        memory_key="chat_history", output_key="answer", return_messages=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        db.as_retriever(search_type="similarity_score_threshold",
            search_kwargs={"k": k,
                            'score_threshold': 0.3}),
        return_source_documents=True,
        memory=memory,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    return qa_chain

def get_source(response):
    ans = response['answer']
    source = ""
    try:
        if os.path.basename(response['source_documents'][0]['metadata']['source']) != "":
            src = os.path.basename(response['source_documents'][0]['metadata']['source'])
            source += "\n Document Name: " +src + "  Page No.: " + str(response['source_documents'][0]['metadata']['page']+1) 
        
            reply = ans + '\n\n' + 'Source: ' +  source
            return reply
    
    except:
        return ans + '\n\n' + "Thanks for asking!" + '\n\n' + "No Source Document Found!"

def chat_with_doc(prompt, db, k):
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    qa_chain = get_llm_chain(llm, db, k)
    result = qa_chain.invoke({"question": prompt, "chat_history": []})

    return get_source(result)