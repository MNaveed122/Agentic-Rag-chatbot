#agentic rag
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

llm = ChatGroq(
    model="gemma2-9b-it",
    api_key=os.getenv("GROQ_API_KEY")
)
from typing import Annotated, Literal, Sequence, TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
# Pinecone configuration from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "us-east-1")
#Extract data from the PDF
def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)

    documents = loader.load()

    return documents
    "D:\Edhi_Hospital_Complete_Details_Expanded1.pdf"
    from langchain.document_loaders import PyPDFLoader

def load_pdf(file_path):
    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(file_path)

    # Load the documents
    documents = loader.load()

    return documents

# Specify the path to your PDF file
pdf_file_path = "D:\Edhi_Hospital_Complete_Details_Expanded1.pdf"

# Load the PDF
documents = load_pdf(pdf_file_path)


# Print the loaded documents (optional)
for doc in documents:
    print(doc)
extracted_data = load_pdf(pdf_file_path)
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=100, chunk_overlap=10)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
doc_splits = text_split(extracted_data)
doc_splits[1]
text_chunks = text_split(doc_splits)
print("length of my chunk:", len(text_chunks))
import os
# Set HuggingFace tokens from environment variables
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
from langchain.vectorstores import Pinecone as LangchainPinecone  # Use LangChain's Pinecone integration
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from pinecone import Pinecone # Use the official Pinecone client for index management
import os
# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Set the environment variables to properly configure the Pinecone client
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_API_ENV"] = PINECONE_API_ENV
# Define HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")

# Define the Pinecone index name
index_name = "agent"

# Check if the index exists, and create it if it doesn't
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Dimension of all-MiniLM-L6-v2 embeddings
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(
            cloud="aws",  # Cloud provider
            region="us-east-1"  # Region
        )
    )

# Create a Pinecone vector store from documents using LangChain's Pinecone integration
vectorstore = LangchainPinecone.from_documents(
    documents=text_chunks,  # Your document splits
    embedding=embeddings,
    index_name=index_name,
)

retriever=vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_Edhi_Hospital_Complete_Details_Expanded1",
    """You are an assistant specializing in Edhi Hospital information.

    - **NEVER attempt to answer on your own. Always use the retriever tool.**
    - **ONLY use the retriever tool (`retriever_tool`) when the question relates to Edhi Hospital and doctors information.**
    - If the query is not about Edhi Hospital and doctors, respond normally.

    If you don't find relevant information, say:
    `"I'm sorry, but I couldn't find specific details on that."`
    """
)
tools=[retriever_tool]
retrieve=ToolNode([retriever_tool])
from typing import TypedDict, Sequence
from typing_extensions import Annotated

class BaseMessage:
    def __init__(self, content: str):
        self.content = content

# Defining the `AgentState` TypedDict
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "add_messages"]

# Example usage
message1 = BaseMessage("Hello, world!")
message2 = BaseMessage("How are you?")
state = AgentState(messages=[message1, message2])

print(state)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
def ai_assistant(state:AgentState):
    print("---CALL AGENT---")
    messages = state['messages']

    if len(messages)>1:
        last_message = messages[-1]
        question = last_message.content
        prompt=PromptTemplate(
        template="""You are a helpful assistant whatever question has been asked to find out that in the given question and answer.
                        Here is the question:{question}
                        """,
                        input_variables=["question"]
                        )

        chain = prompt | llm

        response=chain.invoke({"question": question})
        return {"messages": [response]}
    else:
        llm_with_tool = llm.bind_tools(tools)
        response = llm_with_tool.invoke(messages)
        #response=handle_query(messages)
        return {"messages": [response]}
class grade(BaseModel):
    binary_score:str=Field(description="Relevance score 'yes' or 'no'")
def grade_documents(state:AgentState)->Literal["Output_Generator", "Query_Rewriter"]:
    llm_with_structure_op=llm.with_structured_output(grade)

    prompt=PromptTemplate(
        template="""You are a grader deciding if a document is relevant to a user's question.
                    Here is the document: {context}
                    Here is the user's question: {question}
                    If the document talks about or contains information related to the user's question, mark it as relevant.
                    Give a 'yes' or 'no' answer to show if the document is relevant to the question.""",
                    input_variables=["context", "question"]
                    )
    chain = prompt | llm_with_structure_op

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generator" #this should be a node name
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "rewriter" #this should be a node name
hub.pull("rlm/rag-prompt").pretty_print()
def generate(state:AgentState):
    print("---GENERATE---")
    messages = state["messages"]

    question = messages[0].content

    last_message = messages[-1]
    docs = last_message.content

    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = prompt | llm

    response = rag_chain.invoke({"context": docs, "question": question})
    print(f"this is my response:{response}")

    return {"messages": [response]}
from langchain_core.messages import  HumanMessage
def rewrite(state:AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    message = [HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent or meaning.
                    Here is the initial question: {question}
                    Formulate an improved question: """)
       ]
    response = llm.invoke(message)
    return {"messages": [response]}
workflow=StateGraph(AgentState)
workflow.add_node("My_Ai_Assistant",ai_assistant)
workflow.add_node("Vector_Retriever", retrieve)
workflow.add_node("Output_Generator", generate)
workflow.add_node("Query_Rewriter", rewrite)
workflow.add_edge(START,"My_Ai_Assistant")
workflow.add_conditional_edges("My_Ai_Assistant",
                            tools_condition,
                            {"tools": "Vector_Retriever",
                                END: END,})
workflow.add_conditional_edges("Vector_Retriever",
                            grade_documents,
                            {"generator": "Output_Generator",
                            "rewriter": "Query_Rewriter"
                            }
                            )
workflow.add_edge("Output_Generator", END)
workflow.add_edge("Query_Rewriter", "My_Ai_Assistant")
app=workflow.compile()
from IPython.display import Image, display

try:
    display(Image(app.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
app.invoke({"messages":[""]})

def answer_with_rag(user_input):
    # This will invoke your RAG workflow and return the answer as a string
    result = app.invoke({"messages": [user_input]})
    # Extract the answer from the result (adjust as needed for your workflow's output)
    messages = result.get("messages", [])
    if messages:
        return str(messages[-1])
    return "Sorry, I couldn't find an answer."
 