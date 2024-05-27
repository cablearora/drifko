from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import BaseRetriever
from utils import MEMORY
from langchain_pinecone import PineconeVectorStore
from prompt import user_prompt




#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
embeddings = download_hugging_face_embeddings()



LLM = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature= 0.1,
    repetition_penalty = 1.03,
    max_new_tokens = 1024,
    top_k = 30,
    huggingfacehub_api_token = "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")


def configure_retriever():
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='AIzaSyAt5b8fk6CKsLjyW_SqxrnGq28dgEdOYJU')
    # alternatively: 
    index_name = "drifko-legal-chatbot"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key= "5beeb765-a71b-4dfe-b42b-8dc4ca3f14e0")
    retriever = docsearch.as_retriever(search_kwargs = {'k':2})
    return retriever


def configure_chain(retriever: BaseRetriever, prompt: str):
    params = dict(
        llm=LLM,
        retriever=retriever,
        memory=MEMORY,
        verbose=True,
        max_tokens_limit=4000, 
        )
    
    return ConversationalRetrievalChain.from_llm(**params)


def configure_retrieval_chain(prompt: str):
    retriever = configure_retriever()
    chain = configure_chain(retriever=retriever,  prompt=user_prompt)
    return chain
