import fitz
import os
import jieba
import openai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

openai.api_key = st.secrets["OPENAI_API_KEY"]

VECTORSTORE_PATH = "faiss_vectorstore"

prompt_template = """
你是一个基于提供的信息回答问题的AI模型。

过去的对话记录：{previous_chat_history}
问题：{question}

请根据你掌握的信息回答这个问题，但不要严格按照数据集逐字输出。请先思考后再回答，你可以基于数据集提供的内容进行推理，并加入你的理解，但不能偏离数据的核心信息。请用自然、生动的语言回答，确保内容准确且富有逻辑性。
"""

def extract_text_from_pdfs(folder_path):
    text = ""
    for pdf_file in os.listdir(folder_path):
        doc = fitz.open(os.path.join(folder_path, pdf_file))
        for page in doc:
            text += page.get_text()
    return text

def tokenize_chinese_text(text):
    return "".join(jieba.cut(text))

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "。", "！", "？"],
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    tokenized_text = tokenize_chinese_text(text)
    chunks = text_splitter.split_text(tokenized_text)
    return chunks

def get_vectorstore(text_chunks):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embedding)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore

def load_or_create_vectorstore():
    """Check if FAISS vector store exists. If not, create it from PDFs."""
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            VECTORSTORE_PATH, 
            OpenAIEmbeddings(model="text-embedding-ada-002"),
            allow_dangerous_deserialization=True  # Explicitly allow loading
        )
    
    text = extract_text_from_pdfs("pdf_folder")  # Extract once
    chunks = get_text_chunks(text)  # Split once
    return get_vectorstore(chunks)  # Save and return FAISS


def generate_response(question, chat_history, conversation_chain):
    previous_chat_history = "\n".join([f"{message['role']}:{message['content']}" for message in chat_history])
    formatted_prompt = prompt_template.format(previous_chat_history=previous_chat_history, question=question)
    response = conversation_chain.invoke(formatted_prompt)
    return response["answer"]

def get_conversation_memory(vectorstore, model):
    llm = ChatOpenAI(model=model)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever(),
        get_chat_history=lambda h: h,
    )
    return conversation_chain

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon=":books:")
    st.header("RAG Chatbot :books:")

    if st.button("Reset Chat"):
        st.session_state["messages"] = [{"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}]
        st.rerun()

    model = st.sidebar.selectbox("Choose AI Model", ["gpt-4o", "chatgpt-4o-latest"])

    # Load or create vector store (runs only once)
    vectorstore = load_or_create_vectorstore()
    
    # Create conversation chain
    conversation_chain = get_conversation_memory(vectorstore, model)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}]

    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    prompt = st.chat_input("Ask something...")

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = generate_response(prompt, st.session_state.messages, conversation_chain)
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
