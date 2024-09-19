from fastapi import FastAPI
from langserve import add_routes

from app.llm_service import translate_model_chain

server = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)
chain = translate_model_chain()
add_routes(
    server,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(server, host="localhost", port=8000)
