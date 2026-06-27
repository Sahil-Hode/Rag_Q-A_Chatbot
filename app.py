import streamlit as st
from utils import answer_question

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="🤖", layout="centered")

st.markdown("""
<style>
.stChatMessage { padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Simple PDF RAG Chatbot")
st.write("Ask any question and I'll find the answer in your PDF documents!")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Source Chunks"):
                for i, doc in enumerate(msg["sources"], 1):
                    src = doc.metadata.get('source', 'Unknown').split('/')[-1]
                    st.markdown(f"**Chunk {i} ({src})**\n\n{doc.page_content}")

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                answer, docs = answer_question(prompt)
            
            st.markdown(answer)
            
            if docs:
                with st.expander("View Source Chunks"):
                    for i, doc in enumerate(docs, 1):
                        src = doc.metadata.get('source', 'Unknown').split('/')[-1]
                        st.markdown(f"**Chunk {i} ({src})**\n\n{doc.page_content}")
                        
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "sources": docs
            })
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
