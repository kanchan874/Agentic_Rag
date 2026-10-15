import streamlit as st
try:
    import pypdf as PyPDF2
except ImportError:
    try:
        import PyPDF2
    except ImportError:
        PyPDF2 = None
import json
import requests
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
import litellm
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

# Page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Cognitive RAG Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# Prevent LiteLLM telemetry and retries hanging
os.environ["LITELLM_TELEMETRY"] = "False"
litellm.telemetry = False
litellm.num_retries = 0

def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret from environment variables or Streamlit secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

@st.cache_resource
def get_qdrant_client():
    """
    Initializes a resilient, singleton Qdrant client:
    1. Checks for cloud Qdrant via QDRANT_URL and QDRANT_API_KEY (from env or st.secrets).
    2. Checks for local Qdrant server (http://localhost:6333) used during local development.
    3. Falls back to embedded on-disk Qdrant storage.
    4. If the directory is locked (e.g. concurrent session on Streamlit Cloud), falls back to in-memory.
    """
    qdrant_url = get_secret("QDRANT_URL", "")
    qdrant_api_key = get_secret("QDRANT_API_KEY", "")

    if qdrant_url:
        try:
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, timeout=4.0)
            client.get_collections()
            return client
        except Exception:
            pass

    # Check local running daemon (localhost:6333)
    try:
        client = QdrantClient("http://localhost:6333", timeout=1.0)
        client.get_collections()
        return client
    except Exception:
        pass

    # Embedded local disk storage
    try:
        storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")
        os.makedirs(storage_path, exist_ok=True)
        return QdrantClient(path=storage_path)
    except Exception:
        # If disk storage is already locked or inaccessible, gracefully fall back to in-memory
        return QdrantClient(location=":memory:")

# Session State Initialization
if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "indexed_count" not in st.session_state:
    st.session_state.indexed_count = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# Auto-reconnect to existing indexed Qdrant collection on page reload
if st.session_state.client is None or st.session_state.collection_name is None:
    try:
        init_client = get_qdrant_client()
        existing_cols = [c.name for c in init_client.get_collections().collections]
        if "agent_rag_index" in existing_cols:
            col_info = init_client.get_collection("agent_rag_index")
            if col_info.points_count > 0:
                st.session_state.client = init_client
                st.session_state.collection_name = "agent_rag_index"
                st.session_state.indexed_count = col_info.points_count
    except Exception:
        pass

# Refined Executive UI Styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Top Header Styling */
        .workspace-header {
            margin-bottom: 1.25rem;
            padding-bottom: 0.85rem;
            border-bottom: 1px solid rgba(226, 232, 240, 0.2);
        }
        .workspace-title {
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin: 0;
            color: inherit;
        }
        .workspace-subtitle {
            font-size: 0.88rem;
            color: #64748b;
            margin-top: 0.2rem;
        }

        /* Status Pills */
        .status-pill-active {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 500;
            background-color: rgba(16, 185, 129, 0.1);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }
        .status-pill-neutral {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 500;
            background-color: rgba(148, 163, 184, 0.1);
            color: #94a3b8;
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        /* Metric Chips */
        .metric-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            background: rgba(241, 245, 249, 0.05);
            border: 1px solid rgba(226, 232, 240, 0.15);
            margin-right: 8px;
            margin-bottom: 8px;
        }

        /* Source Citation Box */
        .source-card {
            padding: 10px 14px;
            border-radius: 6px;
            border-left: 3px solid #3b82f6;
            background: rgba(241, 245, 249, 0.03);
            border-top: 1px solid rgba(226, 232, 240, 0.08);
            border-right: 1px solid rgba(226, 232, 240, 0.08);
            border-bottom: 1px solid rgba(226, 232, 240, 0.08);
            margin-bottom: 8px;
            font-size: 0.86rem;
        }

        /* Clean tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            border-bottom: 1px solid rgba(226, 232, 240, 0.15);
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 6px 6px 0 0;
            font-weight: 500;
            font-size: 0.88rem;
        }

        /* Sidebar Clean Polish */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(226, 232, 240, 0.12);
        }
        .sidebar-section-header {
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            margin-top: 1.25rem;
            margin-bottom: 0.5rem;
        }

        /* Code block styling */
        pre {
            border-radius: 6px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        st.error(f"Error crawling {base_url}: {e}")
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
            return " ".join(chunk for chunk in chunks if chunk)
        else:
            return None
    except Exception:
        return None


def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception:
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
    if PyPDF2 is None:
        st.error("No PDF extraction library (pypdf or PyPDF2) found. Please ensure `pypdf` is installed.")
        return pdf_list
    for uploaded_file in uploaded_files:
        content = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                content += page.extract_text() or ""
            pdf_list.append({"content": content, "filename": uploaded_file.name})
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
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
                progress_text.text(f"Scanning target: {base_url}")
                site_urls = get_all_urls(base_url)
                all_urls.update(site_urls)
                progress_bar.progress((i + 1) / len(urls))

            urls = list(all_urls)
            progress_text.text(f"Discovered {len(urls)} target URLs")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, url in enumerate(urls):
            progress_text.text(f"Ingesting URL {i+1}/{len(urls)}: {url}")
            content = fetch_url_content(url)

            if content:
                chunks = text_splitter.split_text(content)
                all_chunks.extend(chunks)
                for _ in chunks:
                    doc_metadata.append({"url": url, "source": "web_content"})

            progress_bar.progress((i + 1) / len(urls))
            time.sleep(0.05)

        progress_text.empty()
        progress_bar.empty()

    if not all_chunks:
        st.error("No valid content found to index.")
        return None, None

    with st.spinner("Generating dense vectors with FastEmbed..."):
        embeddings = get_embeddings(all_chunks)

    client = get_qdrant_client()
    collection_name = "agent_rag_index"
    VECTOR_SIZE = 384

    with st.spinner("Writing chunks into Qdrant vector database..."):
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

    st.session_state.indexed_count = len(all_chunks)
    st.success(
        f"Indexed {len(all_chunks)} chunks across {len(set(m.get('filename') or m.get('url') for m in doc_metadata))} sources."
    )
    return client, collection_name


@st.cache_data(ttl=1800, show_spinner=False)
def get_available_gemini_models(api_key: str):
    """Detect available models for the provided Gemini API key in prioritized order."""
    preferred_order = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.7-flash",
    ]
    try:
        r = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=4,
        )
        if r.status_code == 200:
            available = {
                m["name"].replace("models/", "")
                for m in r.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            }
            ordered = [m for m in preferred_order if m in available]
            if ordered:
                return ordered
    except Exception:
        pass
    return ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-flash-latest"]


def safe_ddg_search(query: str, max_results: int = 4):
    """Resilient DuckDuckGo search with strict timeout to avoid interface blocking."""
    try:
        ddgs = DDGS(timeout=4)
        return list(ddgs.text(query, max_results=max_results))
    except Exception:
        return []


def stream_llm(messages, provider: str, api_key: str, max_tokens: int = 1500):
    """Streams response tokens from Groq, Gemini, or OpenAI."""
    if "Groq" in provider:
        groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for g_model in groq_models:
            payload = {
                "model": g_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
            }
            try:
                with requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=25,
                ) as resp:
                    if resp.status_code == 200:
                        yielded_any = False
                        for raw_line in resp.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8", errors="replace")
                            if line.startswith("data: "):
                                chunk_str = line[6:].strip()
                                if chunk_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(chunk_str)
                                    delta = data["choices"][0]["delta"].get("content", "")
                                    if delta:
                                        yield delta
                                        yielded_any = True
                                except Exception:
                                    continue
                        if yielded_any:
                            return
                    elif resp.status_code in (401, 403):
                        err_msg = resp.json().get("error", {}).get("message", "Invalid Groq API key.")
                        raise Exception(f"Groq API Error ({resp.status_code}): {err_msg}")
                    else:
                        continue
            except Exception as e:
                if "Groq API Error" in str(e):
                    raise e
                continue

        # Fallback to direct completion if streaming interrupted
        yield call_llm(messages, provider, api_key, max_tokens)
        return

    elif "Gemini" in provider:
        candidate_models = get_available_gemini_models(api_key)
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = {"parts": [{"text": msg["content"]}]}
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        body = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system_instruction:
            body["systemInstruction"] = system_instruction

        headers = {"Content-Type": "application/json"}

        for target_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:streamGenerateContent?key={api_key}&alt=sse"
            try:
                with requests.post(url, headers=headers, json=body, stream=True, timeout=25) as resp:
                    if resp.status_code == 200:
                        yielded_any = False
                        for raw_line in resp.iter_lines():
                            if not raw_line:
                                continue
                            line = raw_line.decode("utf-8", errors="replace")
                            if line.startswith("data: "):
                                raw_json = line[6:].strip()
                                if raw_json:
                                    data = json.loads(raw_json)
                                    candidates = data.get("candidates", [])
                                    if candidates:
                                        parts = candidates[0].get("content", {}).get("parts", [])
                                        for part in parts:
                                            chunk_text = part.get("text", "")
                                            if chunk_text:
                                                yield chunk_text
                                                yielded_any = True
                        if yielded_any:
                            return
                    elif resp.status_code in (404, 503):
                        continue
                    elif resp.status_code in (401, 403):
                        try:
                            err_msg = resp.json().get("error", {}).get("message", resp.text)
                        except Exception:
                            err_msg = resp.text
                        raise Exception(f"Gemini API Error ({resp.status_code}): {err_msg}")
                    elif resp.status_code == 400:
                        err_msg = resp.json().get("error", {}).get("message", resp.text)
                        raise Exception(f"Gemini API Error (400): {err_msg}")
            except Exception as e:
                if "Gemini API Error" in str(e):
                    raise e
                continue

        yield call_llm(messages, provider, api_key, max_tokens)
        return

    else:
        # OpenAI
        try:
            response = completion(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                api_key=api_key,
                stream=True,
                num_retries=0,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        except Exception as e:
            err_str = str(e)
            if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str:
                raise Exception(
                    "OpenAI Quota Exhausted: Your OpenAI account has 0 credits remaining. "
                    "Switch to 'Groq (Ultra-Fast)' or 'Google Gemini (Free)' in the sidebar."
                )
            raise e


def call_llm(messages, provider: str, api_key: str, max_tokens: int = 1000) -> str:
    """Non-streaming fallback caller."""
    if "Groq" in provider:
        groq_models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        for g_model in groq_models:
            payload = {
                "model": g_model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=25,
                )
                if r.status_code == 200:
                    data = r.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                elif r.status_code in (401, 403):
                    err_msg = r.json().get("error", {}).get("message", "Invalid Groq API key.")
                    raise Exception(f"Groq API Error ({r.status_code}): {err_msg}")
            except Exception as e:
                if "Groq API Error" in str(e):
                    raise e
                continue
        raise Exception("Groq inference failed on all candidate models.")

    elif "Gemini" in provider:
        candidate_models = get_available_gemini_models(api_key)
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = {"parts": [{"text": msg["content"]}]}
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        body = {
            "contents": contents,
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system_instruction:
            body["systemInstruction"] = system_instruction

        headers = {"Content-Type": "application/json"}
        last_resp_text = ""

        for target_model in candidate_models:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"
            response = requests.post(url, headers=headers, json=body, timeout=25)

            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "")
                return ""
            elif response.status_code in (404, 503):
                last_resp_text = response.text
                continue
            elif response.status_code in (401, 403):
                try:
                    err_msg = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                raise Exception(f"Gemini API Error ({response.status_code}): {err_msg}")
            else:
                last_resp_text = response.text
                continue

        raise Exception(f"Gemini API could not complete request: {last_resp_text}")

    else:
        # OpenAI
        try:
            res = completion(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                api_key=api_key,
                num_retries=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "insufficient_quota" in err_str or "credit_balance_exhausted" in err_str:
                raise Exception(
                    "OpenAI Quota Exhausted: Your OpenAI account has 0 credits remaining. "
                    "Switch to 'Groq (Ultra-Fast)' or 'Google Gemini (Free)' in the sidebar."
                )
            raise e


def answer_question(
    question,
    client,
    collection_name,
    provider="Groq (Ultra-Fast)",
    api_key=None,
    top_k=4,
    enable_web_fallback=True,
):
    if not question.strip():
        st.warning("Please provide a question.")
        return

    def search(text: str):
        query_embedding = get_embeddings(text)[0]
        return client.search(
            collection_name=collection_name, query_vector=query_embedding, limit=top_k
        )

    def format_docs(docs):
        return "\n\n".join(
            f"[Source: {doc.payload.get('metadata', {}).get('filename') or doc.payload.get('metadata', {}).get('url', 'Document')}]\n{doc.payload.get('content', '')}"
            for doc in docs
        )

    t_start = time.time()
    with st.spinner("Executing dense vector retrieval..."):
        results = search(question)
    t_search = time.time() - t_start

    top_score = results[0].score if results else 0.0
    context = format_docs(results) if results else ""

    # Confidence Threshold for direct context matching
    has_high_relevance = top_score >= 0.40

    if has_high_relevance:
        source_names = list(
            set(
                doc.payload.get("metadata", {}).get("filename")
                or doc.payload.get("metadata", {}).get("url")
                or "Document"
                for doc in results
            )
        )

        # Performance and Match Summary Chips
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <span class="metric-chip"><strong>Confidence</strong>: {top_score:.1%}</span>
                <span class="metric-chip"><strong>Retrieval Latency</strong>: {t_search:.3f}s</span>
                <span class="metric-chip"><strong>Sources</strong>: {', '.join(source_names)}</span>
                <span class="metric-chip"><strong>Provider</strong>: {provider.split('(')[0].strip()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Attributed Context Chunks", expanded=False):
            for i, doc in enumerate(results):
                src = doc.payload.get("metadata", {}).get(
                    "filename", doc.payload.get("metadata", {}).get("url", "Context")
                )
                st.markdown(
                    f"""
                    <div class="source-card">
                        <strong>Chunk {i+1}</strong> &bull; Score: <code>{doc.score:.2f}</code> &bull; Source: <code>{src}</code><br/>
                        <span style="color: #64748b;">{doc.payload.get("content", "")}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        system_prompt = f"""You are an executive enterprise assistant providing grounded answers based on indexed organizational documents.
Answer the question accurately, concisely, and cleanly based on the provided context.

Context:
{context}

Formatting Rules:
- Structure your response using clear markdown paragraphs and bullet points.
- If specific figures, platforms, or dates are mentioned, highlight them accurately.
"""
        user_prompt = f"Question: {question}\nAnswer:"

        st.markdown("### Synthesized Response")
        t_gen_start = time.time()
        try:
            stream = stream_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                provider=provider,
                api_key=api_key,
                max_tokens=1500,
            )
            st.write_stream(stream)
            t_gen = time.time() - t_gen_start
            st.caption(
                f"Generation completed in {t_gen:.2f}s (Total roundtrip: {time.time()-t_start:.2f}s)"
            )
        except Exception as e:
            st.error(f"Inference Error: {e}")

    else:
        # Document similarity below threshold -> Web fallback
        if enable_web_fallback:
            st.info(
                f"Internal relevance below threshold ({top_score:.1%}). Broadening search to external web sources..."
            )
            t_ddg_start = time.time()
            web_results = safe_ddg_search(question, max_results=4)
            t_ddg = time.time() - t_ddg_start

            if web_results:
                web_context = "\n\n".join(
                    f"[Title: {r.get('title', '')} | URL: {r.get('href', '')}]\n{r.get('body', '')}"
                    for r in web_results
                )
                st.markdown(
                    f"""
                    <div style="margin-bottom: 0.75rem;">
                        <span class="metric-chip"><strong>Web Retrieval</strong>: {t_ddg:.2f}s</span>
                        <span class="metric-chip"><strong>Results Discovered</strong>: {len(web_results)}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Web Source Citations", expanded=False):
                    for r in web_results:
                        st.markdown(
                            f"- [{r.get('title', 'Reference')}]({r.get('href', '#')}): {r.get('body', '')[:140]}..."
                        )
                effective_context = web_context
                source_desc = "Live Web Search"
            else:
                effective_context = context
                source_desc = "Best Available Document Match"

            web_system_prompt = f"""You are a professional research assistant. Synthesize a structured, accurate answer to the inquiry using the reference context below.

Context ({source_desc}):
{effective_context}
"""
            user_prompt = f"Question: {question}\nAnswer:"

            st.markdown("### Synthesized Response")
            t_gen_start = time.time()
            try:
                stream = stream_llm(
                    messages=[
                        {"role": "system", "content": web_system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    provider=provider,
                    api_key=api_key,
                    max_tokens=1500,
                )
                st.write_stream(stream)
                t_gen = time.time() - t_gen_start
                st.caption(
                    f"Generation completed in {t_gen:.2f}s (Total: {time.time()-t_start:.2f}s)"
                )
            except Exception as e:
                st.error(f"Inference Error: {e}")
        else:
            st.warning(
                f"No direct match found in internal documents (Top score: {top_score:.1%}). External web search is disabled."
            )


# ==============================================================================
# SIDEBAR CONFIGURATION (Executive Panel)
# ==============================================================================
with st.sidebar:
    st.markdown("### System Configuration")
    st.caption("Retrieval Engine & LLM Settings")

    st.markdown('<div class="sidebar-section-header">Inference Provider</div>', unsafe_allow_html=True)
    llm_choice = st.selectbox(
        "Select LLM Provider",
        [
            "Groq (Ultra-Fast)",
            "Google Gemini (Free)",
            "OpenAI (GPT-4o mini)",
        ],
        index=0,
        label_visibility="collapsed",
    )

    if "Groq" in llm_choice:
        default_groq = get_secret("GROQ_API_KEY", "")
        api_key_input = st.text_input(
            "Groq API Key",
            value=st.session_state.get("groq_api_key", default_groq),
            type="password",
            placeholder="gsk_...",
            help="High-speed inference powered by Groq LPU",
        )
        if api_key_input:
            st.session_state.groq_api_key = api_key_input
        active_key = st.session_state.get("groq_api_key", default_groq)
        if active_key:
            st.markdown('<span class="status-pill-active">&bull; Groq Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill-neutral">Key Required</span>', unsafe_allow_html=True)

    elif "Gemini" in llm_choice:
        default_gemini = get_secret("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Google Gemini API Key",
            value=st.session_state.get("gemini_api_key", default_gemini),
            type="password",
            placeholder="AQ... or AIzaSy...",
            help="Google AI Studio key",
        )
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
        active_key = st.session_state.get("gemini_api_key", default_gemini)
        if active_key:
            st.markdown('<span class="status-pill-active">&bull; Gemini Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill-neutral">Key Required</span>', unsafe_allow_html=True)

    else:
        default_openai = get_secret("OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", default_openai),
            type="password",
            placeholder="sk-...",
        )
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
        active_key = st.session_state.get("openai_api_key", default_openai)
        if active_key:
            st.markdown('<span class="status-pill-active">&bull; OpenAI Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill-neutral">Key Required</span>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-header">Knowledge Base Index</div>', unsafe_allow_html=True)
    if st.session_state.client and st.session_state.collection_name:
        st.markdown(
            f"""
            <div style="font-size: 0.84rem; line-height: 1.6;">
                <strong>Database:</strong> Qdrant Local<br/>
                <strong>Collection:</strong> <code>{st.session_state.collection_name}</code><br/>
                <strong>Indexed Chunks:</strong> <code>{st.session_state.indexed_count}</code><br/>
                <strong>Embeddings:</strong> FastEmbed BGE-small
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<span class="status-pill-active" style="margin-top: 6px;">&bull; Vector DB Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size: 0.84rem; color: #94a3b8;">No documents currently indexed. Use the Ingestion tab to add sources.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-section-header">Search Controls</div>', unsafe_allow_html=True)
    enable_web_fallback = st.checkbox("Web Search Fallback", value=True, help="Perform external DuckDuckGo search if vector similarity is low.")
    top_k_chunks = st.slider("Top Chunks Retrieved", min_value=2, max_value=8, value=4)


# ==============================================================================
# MAIN WORKSPACE
# ==============================================================================
st.markdown(
    """
    <div class="workspace-header">
        <h1 class="workspace-title">Cognitive RAG Workspace</h1>
        <div class="workspace-subtitle">Precision document ingestion, dense vector retrieval, and agentic multi-model synthesis.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_query, tab_ingest = st.tabs(["Search & Synthesis", "Document Ingestion & Index"])

# ------------------------------------------------------------------------------
# TAB 1: Search & Synthesis
# ------------------------------------------------------------------------------
with tab_query:
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "Search Query",
            placeholder="Ask anything about the indexed documents (e.g. What is the proposed platform architecture?)",
            label_visibility="collapsed",
            key="user_query_input",
        )
    with col_btn:
        search_clicked = st.button("Synthesize", type="primary", use_container_width=True)

    # Suggestion Buttons
    if st.session_state.client and st.session_state.collection_name and not question:
        st.caption("Suggested queries:")
        s_c1, s_c2, _ = st.columns([1.5, 1.5, 3])
        with s_c1:
            if st.button("Objective of platform", key="btn_sug1"):
                question = "What is the objective of the proposed solution?"
                search_clicked = True
        with s_c2:
            if st.button("Existing vs Proposed", key="btn_sug2"):
                question = "Compare existing platform vs proposed solution"
                search_clicked = True

    if search_clicked:
        if not st.session_state.client or not st.session_state.collection_name:
            st.warning("Knowledge base is empty. Please index documents in the 'Document Ingestion' tab first.")
        elif not active_key:
            st.error(f"Please configure your {llm_choice.split('(')[0].strip()} API key in the left sidebar.")
        else:
            st.session_state.last_query = question
            answer_question(
                question=question,
                client=st.session_state.client,
                collection_name=st.session_state.collection_name,
                provider=llm_choice,
                api_key=active_key,
                top_k=top_k_chunks,
                enable_web_fallback=enable_web_fallback,
            )

# ------------------------------------------------------------------------------
# TAB 2: Document Ingestion & Indexing
# ------------------------------------------------------------------------------
with tab_ingest:
    st.markdown("#### Ingest Organizational Sources")
    st.caption("Upload PDF documentation or enter domain URLs to split, embed, and index into the local vector database.")

    col_pdf, col_web = st.columns(2)

    with col_pdf:
        st.markdown("##### PDF Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"],
            label_visibility="collapsed",
        )

    with col_web:
        st.markdown("##### Web Ingestion")
        web_urls = st.text_input(
            "Enter website URLs (comma-separated)",
            placeholder="https://example.com, https://docs.example.com",
            label_visibility="collapsed",
        )
        crawl_website = st.checkbox(
            "Crawl all linked pages on domain",
            value=False,
            help="Recursively extract text from all links sharing the root domain.",
        )

    col_chunk, col_action = st.columns([2, 1])
    with col_chunk:
        chunk_size = st.slider("Chunk Size (Characters)", min_value=150, max_value=800, value=300, step=50)

    with col_action:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        index_clicked = st.button("Process & Re-Index", type="primary", use_container_width=True)

    if index_clicked:
        if not uploaded_files and not web_urls:
            st.error("Please provide at least one PDF file or web URL to index.")
        else:
            st.session_state.client, st.session_state.collection_name = (
                process_and_index_documents(
                    uploaded_files=uploaded_files,
                    web_urls=web_urls,
                    chunk_size=chunk_size,
                    crawl_website=crawl_website,
                )
            )
            st.rerun()

st.markdown(
    """
    <div class="subtle-footer">
        Cognitive RAG Workspace &bull; Local Dense Vectors &bull; Multi-Provider Synthesis
    </div>
    """,
    unsafe_allow_html=True,
)
