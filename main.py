from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main import process_query


class QueryModel(BaseModel):
    query: str


app = FastAPI(title="WB API", version="1.0", root_path="/api")


@app.post("/generate/")
async def generate_response(query: QueryModel):
    try:
        result = process_query(f"query: {query.query}")
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
