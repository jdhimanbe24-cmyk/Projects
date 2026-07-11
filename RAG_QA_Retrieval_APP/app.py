import os

import streamlit as st

from rag_utility import process_document_to_chroma_db, answer_question


# set the working directory
working_dir = os.path.dirname(os.path.abspath((__file__)))

st.title("🦙 Llama-3.3-70B - Document RAG")

# file uploader widget
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    progress = st.progress(0)

    for i, uploaded_file in enumerate(uploaded_files):

        # Define save path
        save_path = os.path.join(working_dir, uploaded_file.name)

        # Save the uploaded file
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the document
        process_document_to_chroma_db(uploaded_file.name)
        # If your function expects the full path instead, use:
        # process_document_to_chroma_db(save_path)

        progress.progress((i + 1) / len(uploaded_files))

    st.success(f"Successfully processed {len(uploaded_files)} PDF(s).")

# text widget to get user input
user_question = st.text_area("Ask your question about the document")

if st.button("Answer"):

    answer, sources = answer_question(user_question)

    st.markdown("### 🦙 Llama-3.3-70B Response")
    st.markdown(answer)

    st.markdown("### 📄 Sources")
    
    source_names = list(
        set(os.path.basename(doc.metadata["source"]) for doc in sources)
    )

    for source in source_names:
        st.write(f"- {source}")
