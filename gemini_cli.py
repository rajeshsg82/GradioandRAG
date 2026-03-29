import os
import requests
import pdfplumber
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
load_dotenv()

PDF_FOLDER = 'pdfs'  # Folder containing PDFs
CHROMA_DB_DIR = 'chroma_db'  # Directory for Chroma DB

def get_gemini_api_key():
    print("Retrieving Gemini API key from environment variable...")
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError('GEMINI_API_KEY environment variable not set.')
    return api_key

def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from PDF: {pdf_path}...")
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

def chunk_text(text, chunk_size=500):
    print(f"Chunking text into pieces of size {chunk_size}...")
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_gemini_embedding(text, api_key):
    print(f"Getting embedding for text chunk: {text[:50]}...")  # Debug print
    #url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent'
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2-preview:embedContent'
    headers = {'Content-Type': 'application/json'}
    payload = {'content': {'parts': [{'text': text}]}}
    params = {'key': api_key}
    response = requests.post(url, headers=headers, params=params, json=payload)
    print(f"Embedding API response status: {response.status_code}")  # Debug print
    #print(f"Embedding API response body: {response.text}")  # Debug print
    if response.status_code == 200:
        data = response.json()
        try:
            return data['embedding']['values']
        except (KeyError, IndexError):
            return None
    else:
        return None

def index_pdfs(api_key):
    print('Indexing PDFs...')
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection('pdf_chunks')
    metadatas = collection.get().get('metadatas')
    indexed_files = set([m['source'] for m in metadatas]) if metadatas else set()
    for fname in os.listdir(PDF_FOLDER):
        if fname.endswith('.pdf') and fname not in indexed_files:
            pdf_path = os.path.join(PDF_FOLDER, fname)
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                emb = get_gemini_embedding(chunk, api_key)
                print(f"Chunk {i} embedding: {emb[:5]}...")  # Debug print
                if emb:
                    collection.add(
                        ids=[f"{fname}_{i}"],
                        documents=[chunk],
                        embeddings=[emb],
                        metadatas=[{'source': fname, 'chunk': i}]
                    )
                else:
                    print(f"Failed to get embedding for chunk {i} of file {fname}.")    
    print('Indexing complete.')

def search_vector_db(query, api_key, top_k=5):
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection('pdf_chunks')
    print
    query_emb = get_gemini_embedding(query, api_key)
    if not query_emb:
        return []
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    print(f"Search results: {results}")  # Debug print
    return [doc for doc in results['documents'][0]]

def ask_gemini_with_context(question, context, api_key):
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
    headers = {'Content-Type': 'application/json'}
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    print(f"Sending prompt to Gemini API: {prompt[:200]}...")  # Debug print
    payload = {
        'contents': [{
            'parts': [{
                'text': prompt
            }]
        }]
    }
    params = {'key': api_key}
    response = requests.post(url, headers=headers, params=params, json=payload)
    print(f"Gemini API response status: {response.status_code}")  # Debug print
    print(f"Gemini API response body: {response.text}")  # Debug print
    if response.status_code == 200:
        data = response.json()
        try:
            return data['candidates'][0]['content']['parts'][0]['text']
        except (KeyError, IndexError):
            return 'No answer found.'
    else:
        return f'Error: {response.status_code} - {response.text}'

def gradio_chatbot_interface():
    import gradio as gr
    api_key = get_gemini_api_key()
    index_pdfs(api_key)  # Always index on startup
    def chat_fn(message, history):
        context_chunks = search_vector_db(message, api_key)
        context = '\n'.join(context_chunks)
        answer = ask_gemini_with_context(message, context, api_key)
        print(f"Gemini API response: {answer}")
        return answer  # Only return the answer, no citations
    gr.ChatInterface(
        fn=chat_fn,
        title="Gemini 2.5 Flash Chatbot based on RAG and ChromaDB",
        description="Ask questions to Gemini LLM (2.5 Flash) with PDF retrieval via ChromaDB."
    ).launch()

def main():
    api_key = get_gemini_api_key()
    index_pdfs(api_key)
    print('Indexing complete. You can now run the Gradio chatbot interface with:')
    print('python gemini_cli.py gradio')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and (sys.argv[1] == '--gradio' or sys.argv[1] == 'gradio'):
        gradio_chatbot_interface()
    else:
        main()