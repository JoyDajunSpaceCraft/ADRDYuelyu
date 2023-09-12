from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


template = ""
prompt = PromptTemplate(
    input_variables=["product"],
    template=template,
)
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
res = chain.run(product)
print(res)
