## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:
The objective is to create a chatbot that can intelligently respond to queries based on information extracted from a PDF document. By using LangChain, the chatbot will be able to process the content of the PDF and use a language model to provide relevant answers to user queries. The effectiveness of the chatbot will be evaluated by testing it with various questions related to the document.
### DESIGN STEPS:

1.Initialization Input: PDF document path. Output: Document loader, embeddings, vector database, prompt, and chain.

2.Load PDF Content Import and initialize the PyPDFLoader with the provided PDF file path. Extract the content of the PDF into pages.

3.Embed the Document Chunks Import and initialize OpenAIEmbeddings to generate embeddings. Initialize the Chroma vector database with: Persistent storage directory (persist_directory). The embedding function.

4.Define the Language Model Import and initialize ChatOpenAI with: Model name (gpt-4). Temperature (0) for deterministic responses.

5.Create a Retrieval Prompt Define a concise, user-friendly prompt template to: Use context from the document. Limit answers to three sentences. Encourage polite responses with "Thanks for asking!" at the end.

6.Build the Retrieval Chain Initialize the RetrievalQA chain by: Specifying the language model (llm). Linking the retriever (vectordb.as_retriever()). Applying the prompt template. Enabling source document return for transparency.

7.Run the Query Take a query (question) as input. Pass the query to the qa_chain for processing. Retrieve the result and its associated source documents.

8.Output the Result Print the query (question). Print the chatbot’s answer (result["result"]).

### PROGRAM:
```
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
pages = loader.load()

len(pages)

page = pages[0]

print(page.page_content[0:500])

page.metadata

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'

embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name='gpt-4', temperature=0)

# Build prompt
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
question = "What is the topic of this paper?"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

result = qa_chain({"query": question})
print("Question: ", question)
print("Answer: ", result["result"])



```
### OUTPUT:
![Screenshot 2025-04-19 112936](https://github.com/user-attachments/assets/b859581d-9897-4a93-84d5-94e395268f35)


### RESULT:
Prompt: A structured prompt template was designed to pass the document content and user query to the language model.

Model: OpenAI's GPT model was used to process the input data and provide an answer based on the document's content. 

Output Parsing: The model's output is returned as the answer to the query, ensuring that it provides relevant responses based on the content extracted from the PDF.
