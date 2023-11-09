import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# import requests

# url = "https://api-inference.huggingface.co/pipeline/text2text-generation/google/flan-t5-xxl"
# response = requests.get(url, verify=False)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(
        page_title="GI Accenture - ChatBot",
        page_icon="./static/img/acc-logo.png",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get Help": "https://www.extremelycoolapp.com/help",
            "Report a bug": "https://www.extremelycoolapp.com/bug",
            "About": "# This is a header. This is an *extremely* cool app!",
        },
    )
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Add your logo using st.image
    # st.image("./static/img/accenturelogo.png", width=300)

    # Add the header text using st.title
    st.title("GI Accenture - Chat with PDFs")

    # st.help("this is the help you asked for")
    # st.header("GI Accenture - Chat with PDFs")
    # st.write('### Component Example')
    # st.write('<img src="./static/img/acc-logo.png"/>', unsafe_allow_html=True)

    # Create tabs
    tab_titles = ["About", "Use Chatbot"]
    tab1, tab2 = st.tabs(tab_titles)

    # Add content to each tab

    with tab1:
        with open("./static/about.md", "r") as file:
            about_markdown = file.read()
        st.markdown(about_markdown)

        # st.write("Short Video")

        # video_file = open("./static/vid/video.mp4", "rb")
        # video_bytes = video_file.read()

        # st.video(video_bytes)

    with tab2:
        # st.header("Topic A")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

        with st.sidebar:
            # st.image("./static/img/accenturelogo.png", width=300)
            st.image("./static/img/accenturelogo.png", use_column_width=True)
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'",
                accept_multiple_files=True,
            )
            if st.button(label="Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    raw_text = get_pdf_text(pdf_docs)

                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                st.caption(
                    "Click to initiate document loading. Loading documents without GPUs may result in longer processing times, especially when documents are being embedded concurrently."
                )


if __name__ == "__main__":
    main()
