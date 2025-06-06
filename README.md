# AgenticAI2.0Assignment
# Product Information Retrieval with LangChain

This project demonstrates how to use LangChain with Google Generative AI to retrieve structured product information in JSON format. The application uses a Pydantic model to validate the output and ensures the response adheres to the specified format.

## Project Structure

- **`assignment.py`**: The main script that defines the logic for querying product information using LangChain and Google Generative AI.
- **`.env`**: A file to store environment variables such as API keys.
- **`requirements.txt`**: A file listing all the dependencies required to run the project.

## Prerequisites

1. Python 3.8 or higher installed on your system.
2. A valid Google API key for accessing Google Generative AI.
3. Install dependencies listed in `requirements.txt`.

## Setup Instructions

1. Clone the repository or download the project files.
2. Create a `.env` file in the project directory and add the following:

3. Install the required dependencies:
```bash
4. python assignment.py




## README for 2nd Assignment
# AgenticAI2.0Assignment

This project demonstrates the use of **LangChain**, **Pinecone**, **FAISS**, **BM25**, and **OpenSearch** for document retrieval, reranking, and evaluation. It integrates various vector stores, embeddings, and retrieval techniques to process and analyze documents, providing insights into their relevance to specific queries.

## Features

- **PDF Document Processing**: Load and split PDF documents into smaller chunks for efficient processing.
- **Embeddings**: Generate embeddings using the `HuggingFace` model (`intfloat/e5-base-v2`).
- **Vector Stores**:
  - **Pinecone**: Cloud-based vector database for dense vector search.
  - **FAISS**: Local vector store for approximate nearest neighbor (ANN) search.
  - **OpenSearch**: KNN-based vector search for approximate vector retrieval.
- **BM25 Reranking**: Rerank retrieved documents using BM25 scoring for better relevance.
- **Evaluation Metrics**:
  - Precision@k
  - Jaccard Similarity
  - Average Similarity
  - nDCG (Normalized Discounted Cumulative Gain)

## Project Structure

- **`assignment_010625.ipynb`**: Notebook implementing document retrieval, reranking, and evaluation.
- **`assignment_240525.py`**: Script for structured product information retrieval using LangChain and Google Generative AI.
- **`.env`**: File to store environment variables such as API keys.
- **`requirements.txt`**: File listing all the dependencies required to run the project.
- **`faiss_index/`**: Directory to store FAISS index files.
- **`pinecone_reranked_results.txt`**: File to store reranked results from Pinecone.
- **`faiss_reranked_results.txt`**: File to store reranked results from FAISS.

## Prerequisites

1. Python 3.8 or higher installed on your system.
2. Install dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up the following environment variables in a `.env` file:
   ```env
   FILE_PATH=<path_to_pdf_file>
   GROQ_API_KEY=<your_groq_api_key>
   PINECONE_API_KEY=<your_pinecone_api_key>
   GOOGLE_API_KEY=<your_google_api_key>
   ```

## Setup Instructions

1. Clone the repository or download the project files.
2. Create a `.env` file in the project directory and add the required environment variables.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook `assignment_010625.ipynb` or the script `assignment_240525.py` as needed.

## Workflow

### Document Retrieval and Reranking (Notebook: `assignment_010625.ipynb`)

1. **Load PDF Documents**: Use `_load_pdf()` to load and process PDF files.
2. **Split Documents**: Use `_split_documents()` to split documents into smaller chunks.
3. **Generate Embeddings**: Use `HuggingFaceEmbeddings` to generate embeddings for the split documents.
4. **Save Embeddings**:
   - Save to **Pinecone** using `_save_embeddings_to_pinecone()`.
   - Save to **FAISS** using `_save_embedding_to_faiss()`.
   - Save to **OpenSearch** using `_save_embeddings_to_opensearch()`.
5. **Retrieve Documents**:
   - Retrieve from Pinecone using `retrieve_documents_from_pinecone()`.
   - Retrieve from FAISS using `vectorstore.as_retriever()`.
   - Retrieve from OpenSearch using `retrieve_documents()`.
6. **Rerank Documents**: Use `bm25_rerank()` to rerank retrieved documents based on BM25 scores.
7. **Evaluate Results**: Compare retrieval results using metrics like Precision@k, Jaccard Similarity, and Average Similarity.

### Product Information Retrieval (Script: `assignment_240525.py`)

1. Use LangChain with Google Generative AI to retrieve structured product information in JSON format.
2. Validate the output using a Pydantic model.
3. Example query: `"Tell me about the product 'iPhone 14 Pro Max'"`.

## OpenSearch Integration

OpenSearch is used for approximate vector retrieval with KNN-based indexing. The following steps are performed:

1. **Save Embeddings**: Use `_save_embeddings_to_opensearch()` to save document embeddings to an OpenSearch index.
2. **Retrieve Documents**: Use `retrieve_documents()` to retrieve documents from the OpenSearch index.
3. **Index Configuration**: Ensure the OpenSearch index is configured with the correct KNN vector mapping:
   ```json
   {
       "settings": {
           "index": {
               "knn": true
           }
       },
       "mappings": {
           "properties": {
               "embedding": {
                   "type": "knn_vector",
                   "dimension": 768
               },
               "content": {
                   "type": "text"
               }
           }
       }
   }
   ```

## Example Usage

### Query: "Explainable AI for Computer Vision?"

1. **Retrieve and Rerank**:
   - Retrieve documents from Pinecone, FAISS, and OpenSearch.
   - Rerank results using BM25.
2. **Save Results**:
   - Save Pinecone results to `pinecone_reranked_results.txt`.
   - Save FAISS results to `faiss_reranked_results.txt`.

### Output:
- **Pinecone Reranked Results**:
  ```
  Document 1: <First 300 characters of the document>... (BM25 Score: 12.34)
  Document 2: <First 300 characters of the document>... (BM25 Score: 10.56)
  ```
- **FAISS Reranked Results**:
  ```
  Document 1: <First 300 characters of the document>... (BM25 Score: 11.78)
  Document 2: <First 300 characters of the document>... (BM25 Score: 9.45)
  ```

## Evaluation Metrics

- **Precision@k**: Measures the fraction of relevant documents retrieved.
- **Jaccard Similarity**: Measures the overlap between Pinecone and FAISS results.
- **Average Similarity**: Computes the average cosine similarity between embeddings.
- **nDCG@k**: Evaluates the ranking quality of retrieved documents.

## References

- [LangChain Documentation](https://docs.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenSearch Documentation](https://opensearch.org/docs/latest/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)
