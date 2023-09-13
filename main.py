from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
template = '''You are an experienced python programmer. 
Write a function that implements the concept of {concept}'''
prompt = PromptTemplate(
    input_variables=['concept'],
    template=template
)
chain = LLMChain(llm=chat_model, prompt=prompt)
llm = OpenAI(model_name='text-davinci-003', temperature=0.3, max_tokens=1024)
template2 = '''Given the python function {function}, describe its as detailed as possible'''
prompt2 = PromptTemplate(
    input_variables=['function'],
    template=template2
)

chain2 = LLMChain(llm=llm, prompt=prompt2)
overall_chain = SimpleSequentialChain(chains=[chain, chain2], verbose=True)
output = overall_chain.run('DSA graphs')
print(output)
