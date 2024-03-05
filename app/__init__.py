from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
#========================================================================================================================================================================
# Initialize Flask app
app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
#========================================================================================================================================================================
# Load documents and create the database for similarity search
def load_documents():
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = JSONLoader(file_path="./menu.json", jq_schema=".data[]", text_content=False)
    documents = loader.load()
    db = Chroma.from_documents(documents, embedding_function)
    return db
#========================================================================================================================================================================
# Load the documents into the vector store
db = load_documents()
#========================================================================================================================================================================
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.json
        query = data['query']
        results = db.similarity_search(query)
        # Transform results into a serializable format
        serialized_results = [result.page_content for result in results]
        return jsonify(serialized_results), 200
    except KeyError:
        return jsonify({"error": "Query not provided."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
#========================================================================================================================================================================
