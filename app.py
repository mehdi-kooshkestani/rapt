from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import tempfile
import os
# import numpy as np
from RAPTOR import load_file_semantic_chunking_and_embedding_saving, query_chunks

app = FastAPI()

# 🔸 Global retriever variable to store after upload
retriever_25 = None
retriever_sum= None
cuda_avaibality= None
paragraphs= None
chunks_texts=None
initial_and_summarized_chunks = None
all_chunks_number = None

@app.post("/upload")
async def upload_docx(file: UploadFile = File(...)):
    global retriever_25  # reference the global variable
    global retriever_sum
    global cuda_avaibality
    global paragraphs
    global chunks_texts
    global initial_and_summarized_chunks
    global all_chunks_number  
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        retriever_25, retriever_sum, cuda_avaibality, paragraphs, chunks_texts, initial_and_summarized_chunks, all_chunks_number = load_file_semantic_chunking_and_embedding_saving(tmp_path)
        return {
            "status": "file uploaded, divided by paragraphs, semantically chunked, converted to embeddings, embeddings saved to chroma",
            "retrievers returned": "retriever BM25, semantic retriever", "cuda status": cuda_avaibality, "paragraphs":paragraphs, 
            "initial chunks text list":chunks_texts, "number of initial chunks":len(chunks_texts),
            "initial and summarized chunks": initial_and_summarized_chunks,
            "total chunks number": all_chunks_number
        }
    finally:
        os.remove(tmp_path)  # Clean up


@app.post("/query")
async def ask_query(query: str = Form(...)):
    if retriever_25 is None:
        return JSONResponse(status_code=400, content={"error": "No file uploaded yet. Please upload a file first."})
    if retriever_sum is None:
        return JSONResponse(status_code=400, content={"error": "No file uploaded yet. Please upload a file first."})    

    results = query_chunks(query, retriever_25, retriever_sum)  # Pass retriever to query function
    return {"results": results}
