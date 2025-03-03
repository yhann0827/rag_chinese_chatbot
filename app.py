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

base_prompt_template = """
你是一位专业且富有洞察力的对话伙伴，擅长基于已有数据进行分析，同时能够结合逻辑推理和自身理解提供深入的见解。你的回答不仅参考数据，还会融入思考，使其更具价值。

对话背景：
过去的对话记录：{previous_chat_history}
当前问题：{question}
你的任务：
基于数据回答，但不局限于数据。 你可以参考已有信息，但需要结合自己的理解进行分析，而非简单复述。
保持专业性，同时富有思考深度。 你的回答应条理清晰、逻辑严密，并能适当引入推理、行业经验或案例来支持观点。
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

def generate_response(question, chat_history, conversation_chain, additional_prompt):
    previous_chat_history = "\n".join([f"{message['role']}:{message['content']}" for message in chat_history])
    full_prompt = base_prompt_template + additional_prompt
    formatted_prompt = full_prompt.format(previous_chat_history=previous_chat_history, question=question)
    response = conversation_chain.invoke(formatted_prompt)
    return response["answer"]

def get_conversation_memory(vectorstore, model, temperature, max_tokens, top_p, frequency_penalty):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty
    )
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

    model = st.sidebar.selectbox("Choose AI Model", ["gpt-4.5-preview", "gpt-4o", "chatgpt-4o-latest"])

    # Sidebar parameters for fine-tuning
    st.sidebar.markdown("### ⚙️Model Parameters")
    temperature = st.sidebar.number_input("Temperature (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    max_tokens = st.sidebar.number_input("Max Tokens (50 - 4096)", min_value=50, max_value=4096, value=500, step=10)
    top_p = st.sidebar.number_input("Top-p (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    frequency_penalty = st.sidebar.number_input("Frequency Penalty (-2.0 to 2.0)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    # Text area for additional instructions
    additional_prompt = st.sidebar.text_area("Additional Instructions (Optional)", "", height=150)
    
    # Load or create vector store (runs only once)
    vectorstore = load_or_create_vectorstore()
    
    # Create conversation chain
    conversation_chain = get_conversation_memory(vectorstore, model, temperature, max_tokens, top_p, frequency_penalty)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"}]

    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    prompt = st.chat_input("Ask something...")

    if prompt:
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = generate_response(prompt, st.session_state.messages, conversation_chain, additional_prompt)
        st.chat_message("assistant").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

