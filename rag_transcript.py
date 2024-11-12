import streamlit as st


st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    
    st.title("🎥 Ask Questions About Any YouTube Video")
    st.subheader("Analyze and interact with video content effortlessly", divider="red", )
    default_model = "gemini-1.5-flash"
    
    st.toggle("Enable cloude Mode", key="cloud_toggle")
    
    
    
    # st.selectbox()

    
    input_box = st.chat_input(" Enter the youtube url ...")
  
    
if __name__ == "__main__":
    main()