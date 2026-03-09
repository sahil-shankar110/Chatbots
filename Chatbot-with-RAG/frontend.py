import streamlit as st
from backend import generate_response


st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="centered")
st.title("📚 RAG Chatbot")
st.caption("Ask questions from the PDF document.")



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Chat input
prompt = st.chat_input("Ask something from the PDF...")

if prompt:
    # Show user message
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Generate response
    with st.spinner("🤔 Thinking..."):
        response, sources = generate_response(prompt)

    # Format source pages
    if sources:
        source_text = f"\n\n📄 *Sources: Page(s) {', '.join(map(str, sources))}*"
    else:
        source_text = ""

    full_response = response + source_text

    # Show assistant message
    st.chat_message("assistant").write(full_response)
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })