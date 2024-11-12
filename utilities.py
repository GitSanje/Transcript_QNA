from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
import requests
from dotenv import load_dotenv, find_dotenv
import os
from fpdf import FPDF
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, PromptTemplate

from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings, OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Example function that can use any embedding provider
def get_embeddings(provider: str, **kwargs) :
    if provider == "openai":
        return OpenAIEmbeddings(**kwargs)
    elif provider == "cohere":
        return CohereEmbeddings(**kwargs)
    elif provider == "ollama":
        return OllamaEmbeddings(**kwargs)
    elif provider == "gemini":
        os.environ["GOOGLE_API_KEY"] =get_api_key("GEMINI_API_KEY")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def load_env():
    _ = load_dotenv(find_dotenv())
    
def get_api_key(API_KEY):
    load_env()
    api_key = os.getenv(API_KEY)
    return api_key

def extract_video_id(url):
    import re

    # This regex will match the video ID from any YouTube URL (normal, shortened, embedded)
    try:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        return match.group(1) if match else None
    except Exception as e:
        print(e)
        
def validate_video_id( video_id, api_key):
      
        response = requests.get(f"https://www.googleapis.com/youtube/v3/videos?part=id&id={video_id}&key={api_key}")
        if response.json().get("pageInfo", {}).get("totalResults", 0) == 0:
            return False
        return True
        
def get_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_with_timestamps = [
            f"{entry['start']:.2f}s: {entry['text']}" for entry in transcript
        ]
        full_text = "\n".join(transcript_with_timestamps)
        return full_text
    except Exception as e:
        return str(e)
    
    


# Initialize YouTube Data API client
def get_youtube_client(api_key):
    return build("youtube", "v3", developerKey=api_key)

# Function to get video description and channel information
def get_video_details(api_key, video_id):
    # video_id= extract_video_id(url)
    transcript_text = get_transcript(video_id)
    
    youtube = get_youtube_client(api_key)
    
    # Get video details
    video_response = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()
    
    # Extract video description and channel ID
    video_snippet = video_response['items'][0]['snippet']
    video_description = video_snippet.get('description', 'No description available')
    channel_id = video_snippet['channelId']
    
    # Get channel details
    channel_response = youtube.channels().list(
        part="snippet",
        id=channel_id
    ).execute()
    
    # Extract channel "About" description
    channel_snippet = channel_response['items'][0]['snippet']
    channel_about = channel_snippet.get('description', 'No channel about info available')
    country = channel_snippet.get('country', 'No country about info available')
    channel_name = channel_snippet.get('title', 'No channel name available')
    
    

    return video_description, channel_about,country,channel_name,transcript_text
    


def generate_pdf(video_description, channel_about, country, channel_name, transcript_text, url):
    pdf_file = "output.pdf"
    c = canvas.Canvas(pdf_file, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Channel Details")
    
    # Channel Name
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 720, "Channel Name:")
    c.setFont("Helvetica", 12)
    c.drawString(150, 720, channel_name)

    # Video URL
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 700, "Video URL:")
    c.setFont("Helvetica", 12)
    c.drawString(150, 700, url)

    # Channel About
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 680, "Channel About:")
    c.setFont("Helvetica", 12)
    c.drawString(150, 680, channel_about)

    # Video Description
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 660, "Video Description:")
    c.setFont("Helvetica", 12)
    c.drawString(150, 660, video_description)

    # Country
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 640, "Country:")
    c.setFont("Helvetica", 12)
    c.drawString(150, 640, country)

    # Transcript
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, 620, "Transcript:")
    c.setFont("Helvetica", 12)
    c.drawString(150, 620, transcript_text)

    c.save()
    print(f"PDF created successfully as '{pdf_file}'")


    
    
def process_pdf(uploaded_file,message_container,embeddings):
  
    loader = UnstructuredPDFLoader(file_path=uploaded_file)
    with message_container.chat_message("assistant", avatar="🤖"):
        with st.spinner("Extracting text from PDF..."):
            data = loader.load()
            # Split content into chunks for embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

            # Embed chunks into a vector database
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name="local-rag"
            )
    return vector_db


def chat_pdf(prompt,message_container,llm):
    try:
        
        QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""You are an AI language model assistant. Your task is to generate five
                different versions of the given user question to retrieve relevant documents from
                a vector database. By generating multiple perspectives on the user question, your
                goal is to help the user overcome some of the limitations of the distance-based
                similarity search. Provide these alternative questions separated by newlines.
                Original question: {question}""",
                )   
        # RAG prompt
        template = """Answer the question based ONLY on the following context:
                {context}
                Question: {question}
                """
        prompt_template = ChatPromptTemplate.from_template(template)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Retrieve and answer based on vector DB
        retriever = MultiQueryRetriever.from_llm(
                    st.session_state.vector_db.as_retriever(),
                    llm,
                    prompt=QUERY_PROMPT
                )
        
        # Process question with RAG chain
        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
        message_container.chat_message("user", avatar="😎").markdown(prompt)
        # Run the chain and get response
        with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("Model working..."):
                    response = chain.invoke( prompt)
                    st.markdown(response)
        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
                st.error(f"Error: {e}", icon="⛔️")
