from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from chromadb.config import Settings
from chromadb import Client
from chromadb.api import API

class ChromaRetriever:
    
    def __init__(self,  client_settings=None):
        localhost_client_settings = Settings(
            chroma_api_impl="rest",
            chroma_server_host="host.docker.internal",  # when you run this inside a devcontainer you need to explicitely say host.docker.internal to signify "devcontainer host localhost"
            chroma_server_http_port="8000"
        )
        
        self.client_settings: Settings = client_settings or localhost_client_settings
        self.client: API = Client(self.client_settings)
        
    def fromExistingCollection(self, collection_name: str, k: int=5):
        
        # check collection exists
        if (self.client.get_collection(name=collection_name, embedding_function=OpenAIEmbeddings()) is None):
            raise ValueError(f'Chroma collection: {collection_name} does not exist')
        
        print('Chroma collection name:', collection_name)
        self.chromaDb = Chroma(
            collection_name, 
            embedding_function=OpenAIEmbeddings(),
            client_settings=self.client_settings)
        self.retriever = self.chromaDb.as_retriever()

        self.retriever.search_kwargs["distance_metric"] = "cos"
        self.retriever.search_kwargs["fetch_k"] =k
        self.retriever.search_kwargs["maximal_marginal_relevance"] = True
        self.retriever.search_kwargs["k"] = k  
        return self.retriever
        
        '''
        Note about maximal relevance search:
        
        .../site-packages/langchain/vectorstores/base.py
        
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        ''' 