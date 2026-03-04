import os,json,traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgen.utils import read_file,get_table_data
import streamlit as st 
from langchain.callbacks import get_openai_callback
from src.mcqgen.Mcqgenerator import evaluate_chain
from src.mcqgen.logger import logging

with open("c:\Users\ssan\Downloads\AI\Response.json","r") as file:
    Response_json=json.load(file)

# to create a web application we use streamlit library
st.title("----MCQ Generator Application-----")
# create a form using st.form
with st.form("user_inputs"):
    # file upload
    uploaded_files=st.file_uploader("Upload a Pdf or a txt file")
    mcq_count=st.number_input("No. of Mcqs", min_value=3, max_value=50)
    subject=st.text_input("Insert Subject", max_Cahrs=30)
    tone=st.text_input("Complexity Level of Questions", max_chars=30,placeholders="simple")
    button=st.form_submit_button("Create MCQ's")
    
    if button and uploaded_files is not None and tone and mcq_count and subject:
        with st.spinner("loading..."):
            try:
                text=read_file(uploaded_files)
                with get_openai_callback as cb:
                    response=evaluate_chain(
                        {
                            "text": text,
                            "number": mcq_count,
                            "subject": subject,
                            "tone": tone,
                            "response_json": json.dumps(Response_json)
                        }
                    )
            except Exception as e:
                traceback.print_exception(type(e),e,e.__traceback__)
                st.error("Error")
            else:
                print(f"Total Tokens:{cb.total_tokens}")
                print(f"Prompt Tokens:{cb.prompt_tokens}")
                print(f"Completion Tokens:{cb.completion_tokens}")
                print(f"Total Cost:{cb.total_cost}")
                if isinstance(response,dict):
                    quiz=response.get("quiz",None)
                    if quiz is not None:
                        table_get_data=get_table_data(quiz)
                        if table_get_data is not None:
                            df=pd.Dataframe(table_get_data)
                            df.index+=1
                            st.table(df)
                            st.text_area(label="Review", value=response["review"])
                        else:
                            st.error("Error in table data")
                else:
                    st.write(response)
                        
