import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="RAG Study Assistant 🤖",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 RAG Study Assistant")
st.caption("Built by Kancheti Dakshayani | AIML Student | NRIIT 🔥")

st.markdown("---")

# ============================================
# SIDEBAR - Add your knowledge
# ============================================
with st.sidebar:
    st.header("📄 Your Knowledge Base")
    st.caption("Paste your notes here!")
    
    default_text = """Binary Search Tree (BST) is a tree where left child is less than parent and right child is greater.
BST supports insertion, deletion, searching and traversal.
Inorder traversal of BST gives sorted output.

Stack is a LIFO data structure - Last In First Out.
Stack operations are Push and Pop.
Stack is used for infix to postfix conversion.

Queue is a FIFO data structure - First In First Out.
Queue operations are Enqueue and Dequeue.
Queue is used in BFS graph traversal.

Linked List elements are connected using pointers.
Singly Linked List has one pointer. Doubly Linked List has two pointers.

Hashing maps data to fixed size table using hash function.
Collision resolution techniques are separate chaining and linear probing.

RAG stands for Retrieval Augmented Generation.
RAG retrieves relevant documents before generating answers.
RAG reduces hallucination and provides accurate responses."""

    knowledge_text = st.text_area(
        "Paste your notes:",
        value=default_text,
        height=300
    )
    
    build_btn = st.button("🚀 Build Knowledge Base", type="primary", use_container_width=True)

# ============================================
# SESSION STATE
# ============================================
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================
# BUILD VECTOR DB
# ============================================
if build_btn and knowledge_text:
    with st.spinner("🔢 Building knowledge base... (first time takes ~1 min)"):
        try:
            # Split text
            splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
            docs = [Document(page_content=knowledge_text)]
            chunks = splitter.split_documents(docs)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Vector DB
            vectordb = Chroma.from_documents(chunks, embeddings)
            st.session_state.vectordb = vectordb
            
            st.success(f"✅ Knowledge base ready! {len(chunks)} chunks created!")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================
# CHAT INTERFACE
# ============================================
st.subheader("💬 Ask Anything!")

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])
        if chat.get("context"):
            with st.expander("📚 Retrieved Context"):
                st.caption(chat["context"])

# Input
question = st.chat_input("Ask a question from your notes...")

if question:
    if st.session_state.vectordb is None:
        st.warning("⚠️ Please build the knowledge base first! Click '🚀 Build Knowledge Base' in sidebar.")
    else:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching your notes..."):
                # Retrieve
                retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": 2})
                relevant_docs = retriever.get_relevant_documents(question)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                
                # Simple answer from context
                answer = f"📚 Based on your notes:\n\n{context}"
                
                st.write(answer)
                
                with st.expander("🔍 See retrieved chunks"):
                    st.caption(context)
        
        # Save to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "context": context
        })

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("🔥 Powered by LangChain + HuggingFace + ChromaDB | 100% Free!")
