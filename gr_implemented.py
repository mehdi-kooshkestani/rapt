import gradio as gr
import tempfile
import os
import RAPTOR

retriever_25 = None
retriever_sum = None
cuda_avaibality = None
paragraphs = None
chunks_texts = None
initial_and_summarized_chunks = None
all_chunks_number = None

def upload_fn(file_obj, api_key):
    global retriever_25, retriever_sum, cuda_avaibality
    global paragraphs, chunks_texts, initial_and_summarized_chunks, all_chunks_number

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_obj.read())
        tmp_path = tmp.name

    RAPTOR.OPENAI_API_KEY = api_key
    try:
        retriever_25, retriever_sum, cuda_avaibality, paragraphs, chunks_texts, initial_and_summarized_chunks, all_chunks_number = RAPTOR.load_file_semantic_chunking_and_embedding_saving(tmp_path)
        # return f"✅ File processed. CUDA: {cuda_avaibality}, Chunks: {len(chunks_texts)}"
        return {
            "status": "file uploaded, divided by paragraphs, semantically chunked, converted to embeddings, embeddings saved to chroma",
            "retrievers returned": "retriever BM25, semantic retriever", "cuda status": cuda_avaibality, "paragraphs":paragraphs, 
            "initial chunks text list":chunks_texts, "number of initial chunks":len(chunks_texts),
            "initial and summarized chunks": initial_and_summarized_chunks,
            "total chunks number": all_chunks_number
        }
    except Exception as e:
        return f"❌ Error: {str(e)}"
    finally:
        os.remove(tmp_path)

def query_fn(query):
    if not retriever_25 or not retriever_sum:
        return "⚠️ Please upload a file first."
    results = RAPTOR.query_chunks(query, retriever_25, retriever_sum, model_name="gpt-3.5-turbo-instruct", context_window=4096)
    # return results
    return {"results": results}

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Retrieves related texts to query from the file by three methods!")
    
    with gr.Row():
        file_input = gr.File(file_types=[".docx"], label="Upload a .docx file")
        api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
        upload_btn = gr.Button("Process Document")

    upload_status = gr.Textbox(label="Status")

    upload_btn.click(fn=upload_fn, inputs=[file_input, api_key_input], outputs=upload_status)

    query_input = gr.Textbox(label="Your query here")
    query_btn = gr.Button("Ask")
    response_output = gr.Textbox(label="Response")

    query_btn.click(fn=query_fn, inputs=query_input, outputs=response_output)

demo.launch()
