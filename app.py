# Agentic RAG System - Build 30
# Created: 2026-06-30

import streamlit as st
import PyPDF2
import json
import requests
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from litellm import completion
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time
from dotenv import load_dotenv

load_dotenv()

if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None


def get_all_urls(base_url):
    urls = set()
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                url = link["href"]
                full_url = urljoin(base_url, url)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    urls.add(
                        parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
                    )
    except Exception as e:
        st.error(f"An error occurred while crawling {base_url}: {e}")
    return urls


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())

            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

            text = " ".join(chunk for chunk in chunks if chunk)

            return text
        else:
            st.warning(
                f"Failed to fetch content from {url}: Status code {response.status_code}"
            )
            return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None


def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception as e:
        st.error(f"Error: Failed to fetch URL {url}. Exception: {e}")
        return None


def get_embeddings(texts, model="text-embedding-3-small", api_key=None):
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"input": texts, "model": model}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["data"]
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None


def process_uploaded_pdfs(uploaded_files):
    pdf_list = []
    for uploaded_file in uploaded_files:
        content = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                content += page.extract_text()
            pdf_list.append({"content": content, "filename": uploaded_file.name})
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    return pdf_list


