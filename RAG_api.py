from huggingface_hub import HfApi
import os
import api_functions as func
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma import Chroma


HF_key = os.environ.get("HF_TOKEN")
HFapi = HfApi(HF_key)


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)
db = ''

def upload_file(file):

    #store file temporarily
    file_path = func.get_temp_file_path(file)
    file_extension = os.path.splitext(file_path)[1]

    # db.delete(filter={})
    global db 
    error = 0

    if file_extension == ".pdf":
        db = Chroma.from_documents(func.get_text_from_pdf(file_path), embeddings)
    elif file_extension == ".txt":
        db = Chroma.from_documents(func.get_text_from_txt(file_path), embeddings)
    elif file_extension == ".csv":
        db = Chroma.from_documents(func.get_text_from_csv(file_path), embeddings)
    else:
        error = 1
    
    if error == 0:
        return {"status": "Success", "file_name": file.name}
    else:
        return {"status": "File type not supported"}


def chat_with_doc(query, file_name = "Temp"):
    global db
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    # db = HanaDB(embedding=embeddings, connection=conn, table_name="MAV_SAP_RAG")
    qa_chain = func.get_llm_chain(llm, db, file_name)
    # result = qa_chain({"question": query})
    result = qa_chain.invoke({"question": query, "chat_history": []})
    # answer = func.extract_between_colon_and_period(result['answer'])
    print(result)
    return result
    # return {"Message": answer}

 
