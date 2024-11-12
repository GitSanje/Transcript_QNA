import streamlit as st
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader

from utilities import extract_video_id,validate_video_id,get_api_key, get_video_details,generate_pdf

st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)
models = (
    "openai/gpt-4",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-2",
    "anthropic/claude-instant-1",
    "ai21/j1-jumbo",
    "ai21/j2-grande",
    "huggingface/bigscience/bloom",
    "huggingface/google/flan-t5-xxl",
    "azure/gpt-4",
    "azure/gpt-3.5-turbo",
    "gemini/gemini-pro",
    "gemini/gemini-1.5-flash",
    "mistral/mistral-7b",
    "mistral/mistral-instruct-7b",
    "cohere/command-r-plus",
    "cohere/command-xlarge-nightly",
)



def extract_model_names(models_info: list) -> tuple:
    return tuple(model["name"] for model in models_info["models"])


def main():
    
    st.title("🎥 Ask Questions About Any YouTube Video")
    st.subheader("Analyze and interact with video content effortlessly", divider="red", )
    default_model = "gemini-1.5-flash"
    
    st.toggle("Enable cloud Models", key="cloud_toggle")
    models_info = ollama.list()
    available_models = extract_model_names(models_info)
    utube_api_key = get_api_key("YOUTUBE_API_KEY")
    
    if st.session_state.cloud_toggle:
         selected_model = st.selectbox(
            "Pick a model available cloud models ↓", models
        )
         
    else:
        if available_models:
            selected_model = st.selectbox(
                "Pick a model available locally on your system ↓", available_models
            )
        else:
            st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
            if st.button("Go to settings to download a model"):
              st.page_switch("pages/03_⚙️_Settings.py")
    input_box = st.chat_input("Enter the youtube url ...")
    
    if "video_url" not in st.session_state:
        st.session_state.video_url = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "vector_db" not in st.session_state:
         st.session_state.vector_db = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
        
    message_container = st.container(height=500, border=True)
        
    if input_box:
        video_id = extract_video_id(input_box)
        if validate_video_id(video_id, utube_api_key):
            
            video_description, channel_about,country,channel_name,transcript_text = get_video_details(utube_api_key, video_id)
            generate_pdf(video_description, channel_about,country,channel_name,transcript_text, input_box)
            
            loader = UnstructuredPDFLoader(file_path="output.pdf")
            data = loader.load()
            
        else:
            st.error("Video ID not valid or video is not accessible.",)
        
        
      
       
    
    
    st.write(st.session_state)
    
if __name__ == "__main__":
    main()