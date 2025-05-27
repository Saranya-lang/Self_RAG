import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load vectorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization= True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key="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",  # Set this via env var or directly
    model_name="llama3-8b-8192",  # or "llama3-70b-8192"
)

# Step 1: Initial Answer Prompt
answer_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:""")

# Step 2: Self-Reflection Prompt
reflect_prompt = PromptTemplate.from_template("""
You are a reflective AI.

Here is your previous answer:
"{answer}"

And here is the original context:
"{context}"

Was your answer sufficient? If not, improve it. Otherwise, repeat it.
Improved Answer:""")

# Streamlit UI
st.title("🧠 Self-RAG with Groq")
query = st.text_input("Ask a question:")

if query:
    # Step 1: Retrieve Docs
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 2: Initial Answer
    answer_chain = LLMChain(llm=llm, prompt=answer_prompt)
    initial_answer = answer_chain.run({"context": context, "question": query})

    # Step 3: Reflect and possibly improve
    reflect_chain = LLMChain(llm=llm, prompt=reflect_prompt)
    final_answer = reflect_chain.run({"context": context, "answer": initial_answer})

    # Output
    st.subheader("📝 Final Answer")
    st.write(final_answer)

    with st.expander("📄 Retrieved Documents"):
        for doc in docs:
            st.markdown(doc.page_content)

    with st.expander("🧪 Self-Reflection"):
        st.write("Initial Answer:")
        st.code(initial_answer)
