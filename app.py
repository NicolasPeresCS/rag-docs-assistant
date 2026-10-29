"""Interface Streamlit: `streamlit run app.py`."""

import streamlit as st
from pypdf.errors import PyPdfError

from ragdocs.config import get_settings
from ragdocs.llm import LLMError
from ragdocs.loaders import load_directory, load_uploaded
from ragdocs.pipeline import RagPipeline

st.set_page_config(page_title="RAG Docs Assistant", page_icon="📚", layout="wide")
settings = get_settings()


@st.cache_resource
def default_pipeline() -> RagPipeline:
    return RagPipeline.from_documents(load_directory(settings.docs_dir), settings=settings)


with st.sidebar:
    st.header("Documentos")
    files = st.file_uploader("Envie .pdf, .md ou .txt", type=["pdf", "md", "txt"], accept_multiple_files=True)
    if st.button("Indexar meus arquivos", disabled=not files):
        try:
            docs = [d for f in files for d in load_uploaded(f.name, f.getvalue())]
            st.session_state.pipeline = RagPipeline.from_documents(docs, settings=settings)
            st.success(f"{len(st.session_state.pipeline.chunks)} trechos indexados.")
        except (ValueError, OSError, PyPdfError) as exc:  # mostra o erro em vez de quebrar a página
            st.error(f"Não foi possível ler os arquivos: {exc}")
    if st.button("Voltar aos documentos de exemplo"):
        st.session_state.pop("pipeline", None)
    pipeline = st.session_state.get("pipeline") or default_pipeline()
    st.caption(f"Trechos: {len(pipeline.chunks)} · Modo: {pipeline.llm.name}")

st.title("📚 RAG Docs Assistant")
st.write("Pergunte algo sobre os documentos. As respostas citam as fontes e o sistema recusa o que não está neles.")

question = st.chat_input("Ex.: Quantos dias de home office por semana?")
if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        try:
            answer = pipeline.ask(question)
        except LLMError as exc:
            st.error(str(exc))
        else:
            st.write(answer.text)
            for s in answer.sources:
                label = f"[{s.id}] {s.source}" + (f" · p.{s.page}" if s.page else "") + (" ✅ citado" if s.cited else "")
                with st.expander(label):
                    st.caption(f"score {s.score}")
                    st.write(s.snippet + "…")
