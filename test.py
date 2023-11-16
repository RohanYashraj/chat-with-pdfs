import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""

from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """ Question: {question}

Answer : Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(
    prompt=prompt,
    llm=HuggingFaceHub(
        repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0, "max_length": 64}
    ),
)

question = "What is the capital of France?"

print(llm_chain.run(question))
