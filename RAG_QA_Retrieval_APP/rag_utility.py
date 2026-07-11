import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


#Load environment variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

# Loading the embedding model
embedding = HuggingFaceEmbeddings()

# Load the llama model from groq
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0
)

def process_document_to_chroma_db(file_name):
    #Load the pdf using UnstructuredPDFLoader
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()
    
    #split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 200
    )
    texts = text_splitter.split_documents(documents)
    # store the document chunks in chroma vector database
    vectordb = Chroma.from_documents(
        documents = texts,
        embedding = embedding,
        persist_directory = f"{working_dir}/doc_vectorstore"
    )
    return 0


def answer_question(user_question):
    # Load the persistent Chroma Vector DataBase
    vectordb = Chroma(
        embedding_function = embedding,
        persist_directory = f"{working_dir}/doc_vectorstore"
    )
    # create a retriever for document search
    retriever = vectordb.as_retriever()
    
    #create a RetrievalQA for chain to answer user questions along Llama-3.3-70B
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
    )
    response = qa_chain.invoke({"query": user_question})

    answer = response["result"]
    sources = response["source_documents"]

    return answer, sources
