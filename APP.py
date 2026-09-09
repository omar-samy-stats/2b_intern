import os
import langchain
# to import environment variables from .env file
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")


if not api_key and "api_key" in st.secrets:
        api_key = st.secrets["api_key"]










from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource    # <-- this decorator caches the result of the function, so we don't reload the model and vectorstore every time

def load_pipeline():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)


    return vectorstore, model

vectorstore, model = load_pipeline()









def build_prompt(query, retrieved_docs, user_name):     # <-- This function constructs a prompt for the model using the query and retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = f"""You are a helpful assistant that answers questions using the provided context in my own project 
            project context. If the answer is not in the context, say: I don't know.
            The user's name is {user_name} — address them by name naturally in your answer.

Context:
{context}

Question: {query}
Answer:"""
    
    return prompt

def get_text(response):         # <-- This function extracts the text content from the model's response, handling different response formats
    if isinstance(response.content, str):
        return response.content
    elif isinstance(response.content, list):
        return "".join(block["text"] for block in response.content if block.get("type") == "text")
    return str(response.content)


def answer_question(query, user_name, k=3):    # <-- this function retrieves relevant documents based on the query, constructs a prompt, and generates an answer using the model
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    prompt = build_prompt(query, retrieved_docs, user_name)
    response = model.invoke(prompt)   # <-- Generating the answer using the model with the constructed prompt
    return get_text(response), retrieved_docs     














# Streamlit UI

if "user_name" not in st.session_state:
    st.session_state.user_name = None

st.title("Ask My Project")

# Ask for the name only if we don't have it yet
if st.session_state.user_name is None:
    name_input = st.text_input("What's your name?")
    if name_input:
        st.session_state.user_name = name_input
        st.rerun()   # refresh the page now that we have the name
else:
    st.write(f"Welcome, **{st.session_state.user_name}**! Ask a question about the project.")

    query = st.text_input("Ask a question:", placeholder="e.g. Why did DBSCAN perform worse than K-Means?")

    if query:
        with st.spinner("Thinking..."):
            answer, sources = answer_question(query, st.session_state.user_name)

        st.write("### Answer")
        st.write(answer)

        st.write("### Sources")
        for doc in sources:
            section = doc.metadata.get("H2") or doc.metadata.get("H3") or "N/A"
            filename = os.path.basename(doc.metadata.get("source", ""))
            st.write(f"- **{filename}** — {section}"###############################################################