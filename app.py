import streamlit as st
from PIL import Image

def main():
    st.title("Crop Disease Detection")
    st.write("Upload an image of plant leaf")
    
     # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        image= Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        

if __name__ == '__main__':
    main()