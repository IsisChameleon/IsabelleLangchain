import streamlit as st

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from chromadb.config import Settings
from chromadb import Client

load_dotenv()

collection_name = 'NDIS_PDFPLUMBER_1_TEXTS_1024_128'

def setup_chain_and_prompts(temperature):

    llm = ChatOpenAI(temperature=temperature, model='gpt-3.5-turbo-16k')

    template = """
    You are a helpful, polite and well-mannered bot, a specialist in the NDIS Price Guide.
     to help providers to invoice for their services or the items they have sold to the participant.
    I will share a provider's query with you. You will try to understand whether it is a product they sell or a service they perform for a participant. 
    Upon receiving the user query and the price guide context, your aim is to:
    - select for them the approriate item code from the price guide
    - determine the maximum price they can charge for the good or service
    - more generally, by advising them following recommendations set up in the price guide for that particular service if any
    
    When replying, you will follow ALL of the rules below:

    1/ If some information is missing to determine what item code to use, please ask that information to the user
    2/ If there is more than one item code matching the given criteria, determine what makes the difference between one item code and another and ask that question to the user
    3/ If you otherwise don't have enough information to answer the user query, don't invent anything and say you don't know

    Provider query:
    {query}

    Here are the relevant extracts from the price guide:
    {price_guide_context}

    Please write the most informative answer to the provider query:
    """

    prompt = PromptTemplate(
        input_variables=['query', 'price_guide_context'],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def get_chroma_collection(collection_name):
    client_settings = Settings(
            chroma_api_impl="rest",
            chroma_server_host="host.docker.internal",  # when you run this inside a devcontainer you need to explicitely say host.docker.internal to signify "devcontainer host localhost"
            chroma_server_http_port="8000"
        )
    chromaClient = Client(client_settings)
    coll = chromaClient.get_collection(name=collection_name, embedding_function=OpenAIEmbeddings().embed_documents)
    return coll

def similarity_search(query, coll, n_results=10):
    results = coll.query(query_texts=[query], n_results=n_results)
    metadatas = [ met for met in results['metadatas'][0]]
    docs = [ doc for doc in results['documents'][0]]
    return { 'documents': docs, 'metadatas': metadatas}

def get_query_response(chain, query, n_results=10):
    similar_docs = similarity_search(
        query, 
        get_chroma_collection(collection_name), 
        n_results=n_results)
    response = chain.run(query=query, price_guide_context=similar_docs['documents'])
    return response

def main():
    st.set_page_config(
        page_title="NDIS Provider invoicing helper bot", page_icon=":sun:")

    st.header("Invoicing query :sunflower:")
    temperature = st.sidebar.slider('Temperature', 0.0, 1.0, 0.5)
    query = st.text_area("Please enter your query related to invoicing. Don't forget to provide the location, time and description of your service/item")

    if query:

        chain = setup_chain_and_prompts(temperature)
        st.write("Retrieving price guide information...")

        result = get_query_response(chain, query)

        st.info(result)


if __name__ == '__main__':
    main()

