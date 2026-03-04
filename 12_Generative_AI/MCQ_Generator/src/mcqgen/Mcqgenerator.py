import os
import json


from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.llms import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
# from the .env file it reads the key variable
key=os.getenv("OPEN_API_KEY")
llm=ChatOpenAI(open_api_key=key,model="gpt-3.5-turbo",temperature=0.8)

Template=""" here """
quiz_prompt=PromptTemplate(
            input_variables=["text","number","tone","subject","response_json"],
            template=Template
)

quiz_chain=LLMChain(llm=llm,prompt=quiz_prompt,output_key="quiz",verbose=True)

Template2=""" here """
review_prompt=PromptTemplate(
            input_variables=["quiz","subject"],
            template=Template2
)

review_chain=LLMChain(llm=llm,prompt=review_prompt,output_key="review",verbose=True)

evaluate_chain=SequentialChain(chains=[quiz_chain,review_chain],
                               input_variables=["text","number","tone","subject","response_json"],
                               output_variables=["quiz","review"],
                               verbose=True
                               )

