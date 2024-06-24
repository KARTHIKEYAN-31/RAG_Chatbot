import os
import RAG_api as api


def read_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

def call_file_api(input_data):
    response = api.upload_file(input_data)
    return response

def call_chat_api(query, file_name = None):
    if file_name == None:
        response= api.chat_with_doc(query)
    else:
        response = api.chat_with_doc(query, file_name)
    return get_source(response['answer'])
    # return response



def get_source(ans):
    # ans = response['answer']
    source = ""
    try:
        if os.path.basename(response['source_documents'][0]['metadata']['source']) != "":
            src = os.path.basename(response['source_documents'][0]['metadata']['source'])
            source += "\n Document Name: " +src + "  Page No.: " + str(response['source_documents'][0]['metadata']['page']+1) 
        
            reply = ans + '\n\n' + 'Source: ' +  source
            return reply
    
    except:
        return "Sorry There is no relevent Source Document!" + '\n\n' + "Thanks for asking!"