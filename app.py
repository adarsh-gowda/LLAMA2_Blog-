from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.llms import ctransformers


# Function to get response from the LLAMA2 model

def getLLAMA2Response(input_text, no_of_words, blog_style):
    prompt = PromptTemplate(
        input_variables=["input_text", "no_of_words", "blog_style"],
        template="Write a {no_of_words} words blog on {input_text} for {blog_style}"
    )

    # Load the model
    model = ctransformers.CTransformers(
        model="TheBloke/Llama-2-7b-chat-GGUF",
        model_type="llama",
        
        device="cuda:0",
        n_threads=4,
        max_new_tokens=300,
        temperature=0.01,
        top_p=0.95,
      
    )
    # Generate the response
    response = model(prompt.format(input_text=input_text, no_of_words=no_of_words, blog_style=blog_style))
    
    print("Response from the model: ", response)

    return response

st.set_page_config(page_title="Generate",
                    page_icon="assets/images/logo.png",
                    layout='centered',
                    initial_sidebar_state='collapsed')

st.header("Generate Blog")

input_text = st.text_input("Enter the blog topic")


## Creating more columns for additional 2 fields

col1, col2 = st.columns([5, 5])

with col1:
    no_of_words = st.text_input("Number of words")

with col2:
    blog_style = st.selectbox("wrinting the blog for", ("Researchers","Data Scientists","common people"),index=0)

submit_button = st.button("Generate Blog")

# final response

if submit_button:
    st.write(getLLAMA2Response(input_text, no_of_words, blog_style))


