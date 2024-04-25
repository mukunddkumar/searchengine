import sqlite3 as sql
import pandas as pd
from tqdm.auto import tqdm
import regex as re
from transformers import BertModel, BertTokenizer

# Establishing Connection and Data Preparation
connection = sql.connect("eng_subtitles_database.db")  # Adjust database name if necessary
query = 'SELECT * FROM zipfiles'  # Adjust table name if necessary
subtitles_df = pd.read_sql_query(query, connection)
connection.close()

# Custom Cleaning Function
def clean_text(text):
    text = re.sub("\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}"," ",  text)
    text = re.sub(r'\n?\d+\r', "", text)
    text = re.sub('\r|\n', "", text)
    text = re.sub('<i>|</i>', "", text)
    text = re.sub("(?:www\.)osdb\.link\/[\w\d]+|www\.OpenSubtitles\.org|osdb\.link\/ext|api\.OpenSubtitles\.org|OpenSubtitles\.com", " ",text)
    return text.lower()

subtitles_df['cleaned_content'] = subtitles_df['content'].progress_apply(clean_text)

# Semantic Chunking for Data Segmentation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def semantic_chunking(document, similarity_threshold=0.9):
    sentences = document.split('.')
    chunks = []
    current_chunk = sentences[0]
    for i in range(1, len(sentences)):
        similarity_score = 0.5  # Placeholder for similarity score calculation
        if similarity_score >= similarity_threshold:
            current_chunk += '.' + sentences[i]
        else:
            chunks.append(current_chunk)
            current_chunk = sentences[i]
    chunks.append(current_chunk)
    return chunks

subtitles_df['chunks'] = subtitles_df['cleaned_content'].progress_apply(lambda x: semantic_chunking(x))

# Indexing and Embedding Generation
def generate_embeddings(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
    return embeddings

subtitles_df['embeddings'] = subtitles_df['chunks'].progress_apply(lambda x: generate_embeddings(x))

# Streamlit Implementation for User Interaction
import chromadb
import streamlit as st

client = chromadb.PersistentClient(path="subtitle_search_db")
collection = client.get_collection(name="semantic_search_collection")
collection_name = client.get_collection(name="subtitle_file_names")

st.title("Subtitle Search Engine")

search_query = st.text_input("Enter a dialogue to search")

if st.button("Search"):
    # Placeholder for search functionality
    st.write("Search results will be displayed here")
