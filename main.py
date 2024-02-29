from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# Create a vector store
vectorstore = FAISS.from_texts(["Virat scored 100 against SL", "Bumra took 5 wickets"], embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
# Create a ChatOpenAI model
model = AzureChatOpenAI(
    deployment_name="paste deployment name here",
    model_name="paste model name here",
)
# Create the ConversationalRetrievalChain from the model and retriever
chain = ConversationalRetrievalChain.from_llm(model, retriever)
from fastapi import FastAPI
from langserve import add_routes
from my_package.chain import chain
app = FastAPI(title="Retrieval App")
# Add the LangServe routes to the FastAPI app
add_routes(app, chain)
if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app
    uvicorn.run(app, host="localhost", port=8000)