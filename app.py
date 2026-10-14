import streamlit as st
import PyPDF2
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

load_dotenv()

# Prevent LiteLLM from hanging for 30s on rate limits or sending telemetry
os.environ["LITELLM_TELEMETRY"] = "False"
litellm.telemetry = False
litellm.num_retries = 0

if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

# Auto-reconnect to existing indexed Qdrant collection on page reload
if st.session_state.client is None or st.session_state.collection_name is None:
    try:
        init_client = QdrantClient("http://localhost:6333")
        existing_cols = [c.name for c in init_client.get_collections().collections]
        if "agent_rag_index" in existing_cols:
            col_info = init_client.get_collection("agent_rag_index")
            if col_info.points_count > 0:
                st.session_state.client = init_client
                st.session_state.collection_name = "agent_rag_index"
                st.session_state.indexed_count = col_info.points_count
    except Exception:
        pass

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
    VECTOR_SIZE = 384

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


@st.cache_data(ttl=3600, show_spinner=False)
def get_available_gemini_models(api_key: str):
    """Detect available models for the provided Gemini API key once and cache."""
    try:
        r = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            timeout=3,
        )
        if r.status_code == 200:
            models = [
                m["name"].replace("models/", "")
                for m in r.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            preferred_order = [
                "gemini-3.8-flash",
                "gemini-2.5-flash",
                "gemini-2.0-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-latest",
                "gemini-pro",
            ]
            for pref in preferred_order:
                if pref in models:
                    return pref
            if models:
                return models[0]
    except Exception:
        pass
    return "gemini-3.8-flash"


def safe_ddg_search(query: str, max_results: int = 4):
    """Fast, safe DuckDuckGo search that does not hang or crash on ratelimits."""
    try:
        ddgs = DDGS(timeout=4)
        return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        print(f"DuckDuckGo search notice: {e}")
        return []


def stream_llm(messages, provider: str, api_key: str, max_tokens: int = 1500):
    """Streams response tokens from Gemini or OpenAI for instant UI feedback."""
    if "Gemini" in provider:
        target_model = get_available_gemini_models(api_key)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:streamGenerateContent?key={api_key}&alt=sse"

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
        try:
            with requests.post(url, headers=headers, json=body, stream=True, timeout=25) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines(decode_unicode=True):
                        if line and line.startswith("data: "):
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
                    return
                elif resp.status_code == 400:
                    err_msg = resp.json().get("error", {}).get("message", resp.text)
                    raise Exception(f"Gemini API Error (400): {err_msg}")
                elif resp.status_code in (401, 403):
                    try:
                        err_msg = resp.json().get("error", {}).get("message", resp.text)
                    except Exception:
                        err_msg = resp.text
                    if "denied access" in err_msg.lower() or not api_key.startswith("AIzaSy"):
                        raise Exception(
                            "Invalid Gemini API Key: Standard Google AI Studio keys start with 'AIzaSy'. "
                            f"The provided key was denied by Google ({err_msg}). Get a free key at https://aistudio.google.com/app/apikey"
                        )
                    raise Exception(f"Gemini API Error ({resp.status_code}): {err_msg}")
                else:
                    # Fallback to standard completion
                    yield call_llm(messages, provider, api_key, max_tokens)
                    return
        except Exception as e:
            if "Gemini API Error" in str(e):
                raise e
            # Fallback to direct call_llm if SSE fails
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
                    "Please add billing credits at platform.openai.com/settings/organization/billing "
                    "or switch to 'Google Gemini (Free)' in the dropdown above."
                )
            raise e


def call_llm(messages, provider: str, api_key: str, max_tokens: int = 1000) -> str:
    """Universal non-streaming LLM caller with immediate error handling."""
    if "Gemini" in provider:
        target_model = get_available_gemini_models(api_key)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={api_key}"

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
        response = requests.post(url, headers=headers, json=body, timeout=25)

        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return ""
        else:
            try:
                res = completion(
                    model=f"gemini/{target_model}",
                    messages=messages,
                    max_tokens=max_tokens,
                    api_key=api_key,
                    num_retries=0,
                )
                return res.choices[0].message.content
            except Exception:
                raise Exception(f"Gemini API returned {response.status_code}: {response.text}")
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
                    "Please add billing credits at platform.openai.com/settings/organization/billing "
                    "or switch to 'Google Gemini (Free)' in the dropdown above."
                )
            raise e


def answer_question(question, client, collection_name, provider="Google Gemini (Free)", api_key=None, top_k=4):
    if not question.strip():
        st.warning("Please enter a question.")
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
    with st.spinner("⚡ Searching indexed documents..."):
        results = search(question)
    t_search = time.time() - t_start

    top_score = results[0].score if results else 0.0
    context = format_docs(results) if results else ""

    # Smart High-Confidence Threshold:
    # If top chunk has cosine similarity >= 0.40, the indexed context directly matches the topic!
    # Generates in 1 fast pass without a wasteful extra roundtrip.
    has_high_relevance = top_score >= 0.40

    if has_high_relevance:
        source_names = list(set(
            doc.payload.get("metadata", {}).get("filename") or doc.payload.get("metadata", {}).get("url") or "Document"
            for doc in results
        ))
        st.success(f"🎯 Found relevant content in: **{', '.join(source_names)}** *(Top Score: {top_score:.1%}, Retrieved in {t_search:.2f}s)*")
        
        with st.expander("📄 View Retrieved Document Context"):
            for i, doc in enumerate(results):
                st.markdown(f"**Chunk {i+1}** (Score: `{doc.score:.2f}` | Source: `{doc.payload.get('metadata', {}).get('filename', doc.payload.get('metadata', {}).get('url', 'Unknown'))}`):")
                st.caption(doc.payload.get("content", ""))

        system_prompt = f"""You are an expert AI assistant answering questions based on indexed documents.
Provide an accurate, concise, and structured answer based **exclusively** on the given context.

Rules:
1. If the question cannot be answered using the context, state: "The uploaded documents do not contain enough information to fully answer this." and summarize whatever related points are mentioned.
2. Format your response in clean Markdown with clear headings or bullet points.

Context:
{context}
"""
        user_prompt = f"Question: {question}\nAnswer:"

        st.subheader("💡 Answer:")
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
            st.caption(f"⚡ Search: {t_search:.2f}s | Generation: {t_gen:.2f}s | Total: {time.time()-t_start:.2f}s")
        except Exception as e:
            st.error(f"Error querying model: {e}")

    else:
        # Out of database context -> Fallback to Web Search
        st.info(f"🔍 Low similarity in indexed documents (Top score: {top_score:.1%}). Searching web sources...")
        t_ddg_start = time.time()
        web_results = safe_ddg_search(question, max_results=4)
        t_ddg = time.time() - t_ddg_start

        if web_results:
            web_context = "\n\n".join(
                f"[Title: {r.get('title', '')} | URL: {r.get('href', '')}]\n{r.get('body', '')}"
                for r in web_results
            )
            st.markdown(f"🌐 Found **{len(web_results)}** web results (Search took {t_ddg:.2f}s)")
            with st.expander("🌐 View Web Sources"):
                for r in web_results:
                    st.markdown(f"- [{r.get('title', 'Link')}]({r.get('href', '#')}): {r.get('body', '')[:160]}...")
            effective_context = web_context
            source_desc = "DuckDuckGo Web Search"
        else:
            st.warning("⚠️ Web search provider rate-limited. Falling back to model knowledge with best available context.")
            effective_context = context
            source_desc = "Document matches & General Knowledge"

        web_system_prompt = f"""You are an expert AI assistant. Provide a clear, helpful, and informative answer to the question using the context below.

Context ({source_desc}):
{effective_context}

Format response in clean Markdown.
"""
        user_prompt = f"Question: {question}\nAnswer:"

        st.subheader("💡 Answer:")
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
            st.caption(f"⚡ Search: {t_search + t_ddg:.2f}s | Generation: {t_gen:.2f}s | Total: {time.time()-t_start:.2f}s")
        except Exception as e:
            st.error(f"Error querying model: {e}")


st.title("⚡ Agentic RAG System")
st.caption("High-Performance RAG with PDF & Website Crawling, Local Embeddings, and Real-Time Streaming")

# Display status of vector database and indexed documents
if st.session_state.client and st.session_state.collection_name:
    try:
        current_info = st.session_state.client.get_collection(st.session_state.collection_name)
        count = current_info.points_count
        st.success(f"🟢 **Qdrant Vector Database Connected**: `{st.session_state.collection_name}` has **{count} chunks** ready to query.")
    except Exception:
        pass

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
        active_key = st.session_state.get("gemini_api_key", default_gemini)
        if active_key and not active_key.startswith("AIzaSy"):
            st.warning(
                "⚠️ **Key Format Notice**: Google Gemini API keys start with `AIzaSy...`. "
                "The current key (`AQ...`) is an internal OAuth/session token and is rejected by Google AI. "
                "Please get your free key from [Google AI Studio](https://aistudio.google.com/app/apikey)."
            )
        else:
            st.caption("✨ **Gemini API is 100% Free** (No credit card required). Get key at [Google AI Studio](https://aistudio.google.com/app/apikey)")
    else:
        default_openai = os.getenv("OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            "Enter OpenAI API Key:",
            value=st.session_state.get("openai_api_key", default_openai),
            type="password",
        )
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
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
    st.markdown("---")
    st.subheader("💬 Ask Your Question")
    question = st.text_input("Enter your question about the documents:", placeholder="e.g. What is the proposed solution?")
    if st.button("Get Answer", type="primary"):
        if not active_key:
            st.error(f"Please enter your {llm_choice} API key above to generate the answer.")
        else:
            answer_question(
                question,
                st.session_state.client,
                st.session_state.collection_name,
                provider=llm_choice,
                api_key=active_key,
            )
elif uploaded_files or web_urls:
    st.warning("Please click 'Process and Index Documents' first.")
else:
    st.info("Upload PDFs or provide web URLs to get started.")
