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
    
def generate_pdf(video_description, channel_about,country,channel_name,transcript_text,url):
    pdf = FPDF()

    pdf.add_page()


    pdf.set_font("Arial", size=12)
    # Channel Name 
    pdf.set_font("Arial", "B", 12)  # Bold font
    pdf.cell(0, 10, "Channel Name:", ln=True)
    pdf.set_font("Arial", size=12)  # Regular font
    pdf.multi_cell(0, 10, channel_name)
    pdf.ln(5)

    # Add video description
    pdf.set_font("Arial", "B", 12)  # Bold font
    pdf.cell(0, 10, "Video URL:", ln=True)
    pdf.set_font("Arial", size=12)  # Regular font
    pdf.multi_cell(0, 10, url)
    pdf.ln(5)


    # Add channel "about" information
    pdf.set_font("Arial", "B", 12)  # Bold font
    pdf.cell(0, 10, "Channel About:", ln=True)
    pdf.set_font("Arial", size=12)  # Regular font
    pdf.multi_cell(0, 10, channel_about)
    pdf.ln(5)

    # Add video description
    pdf.set_font("Arial", "B", 12)  # Bold font
    pdf.cell(0, 10, "Video Description:", ln=True)
    pdf.set_font("Arial", size=12)  # Regular font
    pdf.multi_cell(0, 10, video_description)
    pdf.ln(5)

    # Add country
    pdf.set_font("Arial", "B", 12)  # Bold font
    pdf.cell(0, 10, "Country:", ln=True)
    pdf.set_font("Arial", size=12)  # Regular font
    pdf.multi_cell(0, 10, country)
    pdf.ln(10)

    # Add transcript
    pdf.set_font("Arial", "B", 12)  # Bold font
    pdf.cell(0, 10, "Transcript:", ln=True)
    pdf.set_font("Arial", size=12)  # Regular font
    pdf.multi_cell(0, 10, transcript_text)

    # Save the PDF
    pdf.output("output.pdf")
    print("PDF created successfully as 'output.pdf'")
    
    
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