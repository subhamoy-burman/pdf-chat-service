import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai  import AzureChatOpenAI, AzureOpenAIEmbeddings, OpenAI 
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub


load_dotenv()

if __name__ == '__main__':
    print('Hello world')
    pdf_path = r'C:\Users\sburman\Projects\Py-PdfReader\2210_03629v3.pdf'
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator= "\n")

    docs = text_splitter.split_documents(documents = documents)

    # Use Azure OpenAI embeddings
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-ada-002",
        azure_deployment="subhamoy-text-embeddings"
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    # azure_endpoint="https://<your-endpoint>.openai.azure.com/", If not provided, will read env variable AZURE_OPENAI_ENDPOINT
    # api_key=... # Can provide an API key directly. If missing read env variable AZURE_OPENAI_API_KEY
    # openai_api_version=..., # If not provided, will read env variable AZURE_OPENAI_API_VERSION
    )
    
    llm = AzureChatOpenAI(
        temperature=0,
        stop=["\nObservation"],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_api_version="2024-08-01-preview",  # Specify API version
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        azure_deployment="gpt-4o",
        model="gpt-4o"
    )

    vectorStore = FAISS.from_documents(docs, embeddings)
    vectorStore.save_local("faiss_index_react")

    new_vectorStore = FAISS.load_local(
        "faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(new_vectorStore.as_retriever(), combine_docs_chain)

    response = retrieval_chain.invoke({"input":"Give me the gist of ReAct Agent in 5 lines"})

    print(response["answer"])


