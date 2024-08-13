import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from PIL import Image
import speech_recognition as sr
from io import StringIO
import requests
from bs4 import BeautifulSoup
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage

# Set up the environment
os.environ['GOOGLE_API_KEY'] = 'your google api key'

# Initialize models
llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

# Set up the Streamlit app
st.set_page_config(page_title="Multi-Function Chat App", page_icon="🌐", layout="wide")

# Sidebar for navigation
st.sidebar.title("🔍 Explore the App")
app_mode = st.sidebar.selectbox("Choose the functionality",
                                ["🈂️ Language Translator",
                                 "📄 PDF Summarizer", 
                                 "🌐 Chat with Website", 
                                 "🖼️ Image Analysis", 
                                 "📄 Chat with PDF"]
                                )

# Define the system template for answering questions
system_template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer."""

# Setup message templates
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

prompt = ChatPromptTemplate.from_messages(messages)

# Functions for PDF handling
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context," and don't provide the wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to process user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("🤖", response["output_text"])

# Function to capture audio and convert it to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError:
            return "Sorry, there was an error with the request."

# Main function to render the Streamlit app
def main():
    st.header("💬 Multi-Function Chat App")

    # Language Translator
    if app_mode == "🈂️ Language Translator":
        translation_template = """you are a helpful assistant,
        translate the following {speech} into {Language}."""
        translation_prompt = PromptTemplate(
            input_variables=["speech", "Language"],
            template=translation_template
        )

        st.title("🌐 Language Translator")
        st.write("Welcome! Translate your speech into the language of your choice. 🌍")
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
        speech = st.text_area("✏️ Input your speech")
        uploaded_file = st.file_uploader("Or upload a PDF file", type="pdf")
        language = st.selectbox("🌍 Select Language", ["Urdu", "Spanish", "French", "German", "Chinese","Hindi"])

        if st.button("Translate"):
            with st.spinner("Translating... 🔄"):
                if uploaded_file is not None:
                    reader = PdfReader(uploaded_file)
                    speech = ""
                    for page in reader.pages:
                        speech += page.extract_text()

                translation_chain = LLMChain(llm=llm, prompt=translation_prompt)
                translated_text = translation_chain.run({"speech": speech, "Language": language})
                st.success("Translation completed! ✅")
                st.write(translated_text)

    # PDF Summarizer
    elif app_mode == "📄 PDF Summarizer":
        summarization_template = """you are a helpful assistant,
        Summarize the following {speech}."""
        summarization_prompt = PromptTemplate(
            input_variables=["speech"],
            template=summarization_template
        )

        st.title("📄 PDF Summarizer")
        st.write("Welcome! Upload a PDF document and get a summary. 📚")
        llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro')
        uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

        if st.button("Summarize"):
            with st.spinner("Reading and summarizing the document... 🔄"):
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)
                summarized_text = summarization_chain.run({"speech": text})
                
                st.success("Summary completed! ✅")
                st.write(summarized_text)

    # Chat with Website
    elif app_mode == "🌐 Chat with Website":
        st.sidebar.title('🌍 Input your website URL')
        st.sidebar.write('***Ask questions below, and receive answers directly from the website.***')

        url = st.sidebar.text_input("🌍 Insert the website URL")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.write("💬 Ask a question via text or speech input:")
        user_question_text = st.chat_input("📝 Ask a Question About PDF Files")
        if st.button("🔊"):
            user_question_speech = speech_to_text()
            if user_question_speech:
                user_question_text = user_question_speech
                    
        if user_question_text and url:
            os.environ['GOOGLE_API_KEY'] = "your google api key"  
            r = requests.get(url)
            soup = BeautifulSoup(r.content, 'html.parser')
            text = soup.get_text(separator='\n')
            text_splitter = CharacterTextSplitter(separator='\n', chunk_size=512, chunk_overlap=100)
            docs = text_splitter.split_text(text)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = FAISS.from_texts(texts=docs, embedding=embeddings)
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})

            llm = ChatGroq(model="llama3-70b-8192", groq_api_key="your groq api key")
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            full_query = {
                "query": user_question_text,
                "chat_history": st.session_state.chat_history
            }

            response = qa.invoke(full_query)
            st.session_state.chat_history.append({"query": user_question_text, "response": response['result']})

        if st.session_state.chat_history:
            for entry in st.session_state.chat_history:
                _, user_col = st.columns([1, 1])
                with user_col:
                    st.markdown(f"<div style='background-color:; padding: 10px; border-radius: 10px;'>{entry['query']}</div>", unsafe_allow_html=True)
                
                bot_col, _ = st.columns([2, 1])
                with bot_col:
                    st.markdown(f"<div style='background-color:; padding: 10px; border-radius: 10px;'>{entry['response']}</div>", unsafe_allow_html=True)

    # Image Analysis
    elif app_mode == "🖼️ Image Analysis":
        st.sidebar.title("🖼️ Upload Your Image Here")
        uploaded_file = st.sidebar.file_uploader("📂 Select an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())

            image = Image.open("temp_image.jpg")
            st.sidebar.image(image, caption='🖼️ Uploaded Image.', width=300)
            st.sidebar.success("✅ Image Uploaded Successfully")

        st.title("💬🖼️ Image Expression Analysis App")

        if uploaded_file is not None:
            llm1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            text = st.text_input("❓ Ask a Question About the Image")
            if st.button("🔮 Predict"):
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": "temp_image.jpg"}
                    ]
                )

                with st.spinner('🔄 Analyzing the image...'):
                    response = llm1.invoke([message])

                st.subheader("🧠 Model Prediction")
                st.write(response.content)
        else:
            st.write("📤 Upload an image in the sidebar to start analysis.")

    # Chat with PDF
    elif app_mode == "📄 Chat with PDF":
        st.header("📄 RAG: Chat with Multiple PDFs 🧠")

        user_question = st.text_input("📝 Ask a Question About PDF Files")
        if st.button("🔊"):
            user_question_speech = speech_to_text()
            if user_question_speech:
                user_question = user_question_speech

        if user_question:
            user_input(user_question)

        with st.sidebar:
            st.title("📄 Menu:")
            pdf_docs = st.file_uploader("📄 Upload PDF Files", accept_multiple_files=True)
            if st.button("🚀 Submit & Process"):
                with st.spinner("🔄 Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("✅ Done")

if __name__ == "__main__":
    main()
