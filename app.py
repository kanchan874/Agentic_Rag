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
from fastembed import TextEmbedding

load_dotenv()

if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def get_all_urls(base_url):
    urls = set()
    try:
        response = requests.get(base_url, headers=DEFAULT_HEADERS, timeout=15)
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
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
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


@st.cache_resource
def get_embedding_model():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


def get_embeddings(texts):
    model = get_embedding_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = list(model.embed(texts))
    return [emb.tolist() for emb in embeddings]


def process_uploaded_pdfs(uploaded_files):
    pdf_list = []
    for uploaded_file in uploaded_files:
        content = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                content += page.extract_text() or ""
            pdf_list.append({"content": content, "filename": uploaded_file.name})
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    return pdf_list


def process_and_index_documents(
    uploaded_files, web_urls=None, chunk_size=300, crawl_website=False
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50,
    )

    all_chunks = []
    doc_metadata = []

    if uploaded_files:
        all_documents = process_uploaded_pdfs(uploaded_files)
        for doc in all_documents:
            chunks = text_splitter.split_text(doc["content"])
            all_chunks.extend(chunks)
            for _ in chunks:
                doc_metadata.append(
                    {"filename": doc["filename"], "source": "pdf_dataset"}
                )

    if web_urls:
        urls = [url.strip() for url in web_urls.split(",") if url.strip()]

        if crawl_website:
            all_urls = set()
            progress_bar = st.progress(0)
            progress_text = st.empty()

            for i, base_url in enumerate(urls):
                progress_text.text(f"Crawling website: {base_url}")
                site_urls = get_all_urls(base_url)
                all_urls.update(site_urls)
                progress_bar.progress((i + 1) / len(urls))

            urls = list(all_urls)
            progress_text.text(f"Found {len(urls)} unique URLs")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, url in enumerate(urls):
            progress_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
            content = fetch_url_content(url)

            if content:
                chunks = text_splitter.split_text(content)
                all_chunks.extend(chunks)
                for _ in chunks:
                    doc_metadata.append({"url": url, "source": "web_content"})

            progress_bar.progress((i + 1) / len(urls))
            time.sleep(0.1)

        progress_text.empty()
        progress_bar.empty()

    if not all_chunks:
        st.error("No content to process. Please provide valid PDFs or web URLs.")
        return None, None

    with st.spinner("Generating local embeddings (100% Free via FastEmbed)..."):
        embeddings = get_embeddings(all_chunks)

    client = QdrantClient("http://localhost:6333")
    collection_name = "agent_rag_index"
    VECTOR_SIZE = 384  # Dimension of BAAI/bge-small-en-v1.5

    with st.spinner("Indexing vectors in Qdrant..."):
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        ids = list(range(len(all_chunks)))
        payload = [
            {"content": chunk, "metadata": metadata}
            for chunk, metadata in zip(all_chunks, doc_metadata)
        ]

        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=payload,
            ids=ids,
            batch_size=256,
        )

    st.success(
        f"Successfully indexed {len(all_chunks)} chunks from {len(set(m['source'] for m in doc_metadata))} sources in Qdrant!"
    )
    return client, collection_name


def answer_question(question, client, collection_name, model="gemini/gemini-1.5-flash", api_key=None, top_k=3):
    if not question.strip():
        st.warning("Please enter a question.")
        return

    def search(text: str):
        query_embedding = get_embeddings(text)[0]
        return client.search(
            collection_name=collection_name, query_vector=query_embedding, limit=top_k
        )

    def format_docs(docs):
        return "\n\n".join(doc.payload["content"] for doc in docs)

    decision_system_prompt = """You are an agent deciding whether a given context answers a user question.
    Return 1 if the context directly contains enough information to answer the question.
    If not return 0.
    Context: {context}
    """

    system_prompt = """You are an expert in answering questions. Provide answers based **exclusively** on the given context. 

        **Rules:**
        1. If the question cannot be answered using the context, respond only with: "I don't know."
        2. Do **not** infer, assume, or add information not explicitly provided in the context.
        3. Your answers must be:
        - **Concise**: Avoid unnecessary details.
        - **Informative**: Focus on actionable and precise responses.
        4. Format your response in **Markdown**.

        **Context:** {context}
    """

    user_prompt = """
    Question: {question}
    Answer:"""

    with st.spinner("Searching for relevant information..."):
        results = search(question)
        context = format_docs(results)

        try:
            response = completion(
                model=model,
                messages=[
                    {
                        "content": decision_system_prompt.format(context=context),
                        "role": "system",
                    },
                    {"content": user_prompt.format(question=question), "role": "user"},
                ],
                max_tokens=50,
                api_key=api_key,
            )
            has_answer = response.choices[0].message.content.strip()

            if "1" in has_answer:
                st.info("Found relevant information in the indexed documents!")
                response = completion(
                    model=model,
                    messages=[
                        {
                            "content": system_prompt.format(context=context),
                            "role": "system",
                        },
                        {"content": user_prompt.format(question=question), "role": "user"},
                    ],
                    max_tokens=1000,
                    api_key=api_key,
                )
                st.markdown(response.choices[0].message.content)
            else:
                st.info("No direct answer in indexed documents. Searching online via DuckDuckGo...")
                results = DDGS().text(question, max_results=5)
                context = "\n\n".join(doc["body"] for doc in results)
                st.info("Generating response from online sources...")
                response = completion(
                    model=model,
                    messages=[
                        {
                            "content": system_prompt.format(context=context),
                            "role": "system",
                        },
                        {"content": user_prompt.format(question=question), "role": "user"},
                    ],
                    max_tokens=1000,
                    api_key=api_key,
                )
                st.markdown(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error querying model ({model}): {e}")


st.title("RAG System with PDF and Website Crawling Support")

col1, col2 = st.columns([1, 2])
with col1:
    llm_choice = st.selectbox(
        "Select LLM Provider:",
        ["Google Gemini (Free)", "OpenAI"],
        index=0,
    )

with col2:
    if llm_choice == "Google Gemini (Free)":
        default_gemini = os.getenv("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Enter Google Gemini API Key:",
            value=st.session_state.get("gemini_api_key", default_gemini),
            type="password",
            help="Free key available at https://aistudio.google.com/app/apikey",
        )
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
        active_model = "gemini/gemini-1.5-flash"
        active_key = st.session_state.get("gemini_api_key", default_gemini)
        st.caption("✨ Tip: Gemini API is 100% Free. Get key at [Google AI Studio](https://aistudio.google.com/app/apikey)")
    else:
        default_openai = os.getenv("OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            "Enter OpenAI API Key:",
            value=st.session_state.get("openai_api_key", default_openai),
            type="password",
        )
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
        active_model = "gpt-4o-mini"
        active_key = st.session_state.get("openai_api_key", default_openai)

uploaded_files = st.file_uploader(
    "Upload PDF files:", accept_multiple_files=True, type=["pdf"]
)

st.subheader("Website Input")
web_urls = st.text_input(
    "Enter website URLs (comma-separated):", placeholder="https://example.com"
)
crawl_website = st.checkbox(
    "Crawl entire website(s)",
    help="Enable this to extract content from all pages of the specified website(s)",
)

if st.button("Process and Index Documents"):
    st.session_state.client, st.session_state.collection_name = (
        process_and_index_documents(
            uploaded_files, web_urls, crawl_website=crawl_website
        )
    )

if st.session_state.client and st.session_state.collection_name:
    question = st.text_input("Ask a question about the documents:")
    if st.button("Get Answer"):
        if not active_key:
            st.error(f"Please enter your {llm_choice} API key above to generate the answer.")
        else:
            answer_question(
                question,
                st.session_state.client,
                st.session_state.collection_name,
                model=active_model,
                api_key=active_key,
            )
elif uploaded_files or web_urls:
    st.warning("Please click 'Process and Index Documents' first.")
else:
    st.info("Upload PDFs or provide web URLs to get started.")
