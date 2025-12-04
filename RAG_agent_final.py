"""
Here we will be building a RAG agent
Packages: GoogleGenerativeAIEmbeddings,
Process:
First we load the pdf loader, and text splitter to split the document into pages.
These pages are then splitted and stored in a vector storage
Then we create a retriever using all the components made so far: embedding model, pages_splitted
All that and we start with the agent code:
Create a tool that retrieves the importnat parts from the docs
make the nodes and finally make the graphs


MAJOR LEARNING: Please be mindful of what are you are importing, first check and verify it, then continue coding.
I spent hours debugging similar looking error in the [SytemMessage] + message handling due to wrong imports :)
"""

import os
from typing import Sequence
from typing import TypedDict
from typing import Annotated
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI  # Chat model
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Embedding model
from langchain_community.document_loaders import PyPDFLoader  # PDF loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.constants import START, END
from operator import add as add_messages  # be very important with this imports and imports in general!!
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool



load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load the text
loader = PyPDFLoader("2303.07839v1.pdf")  # Put the pdf path
pages = loader.load()  # create the pdf loader object

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # create the text splitter object
pages_splitted = text_splitter.split_documents(pages)  # apply splitting to the pages

# Vector store
## Create a persist dir for ease
try:
    if os.path.exists("vectors"):
        print("Vector directory already exists.")
    else:
        os.makedirs("vectors")
        print("Vector directory created.")
except Exception as e:
    print(f"Error creating vector directory: {e}")

try:
    # try loading existing vector store
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name="gemini_embedding_collection",
        persist_directory="vectors"
    )
except Exception as e:
    print(f"Error loading existing vector store: {e}. Trying to make one ...")
    try:
        vectorstore = Chroma.from_documents(
            documents=pages_splitted,
            embedding=embedding_model,
            collection_name="gemini_embedding_collection",
            persist_directory="vectors"
        )
        print("Vector store created successfully.")
    except Exception as e:
        print(f"Error in creating vector store: {e}")

# Retrieval
retriever =vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Agent code starts here:
# NOTE: understand this part to understand the processing of the query and integration
# with the agent workflow

# Tool
@tool
def retrieval_tool(query: str)-> str:
    """This tool searches and finds the information about ChatGPT prompt pattern use"""
    response = retriever.invoke(query)
    if not response:
        return "No relevant information found."
    ## if response is found we process it
    results = []
    ## loop throught the numbered doc chunks and add to the results
    for i, doc in enumerate(response):
        results.append(f"Document {i+1}:\n{doc.page_content}\n")
    ## now from the results makea single string to pas to the llm
    text = "\n\n".join(results)
    return text

tools_list = [retrieval_tool]
llm = llm.bind_tools(tools_list)  # re init needed after binding the tools

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]  # add_messages is a reducer function that is exe each time the stae is updated to adjsut the new and the old states

# Nodes
def decision(state: AgentState)-> str:
    """Decide whether to continue or end based on the last message tool call"""
    message = state["messages"]
    last_message = message[-1]
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

"""
Understand how to write system prompts:
- General behavior
- Tool and description
- Think, act, observe cycle
"""
sys_prompt = """
You are a AI assistant that helps user to read research papers.
You have access to these tools:
retrieval_tool: a tool that retrieves relevant information from the research paper about ChatGPT prompt patterns.
Think: understand the user query and ask follow up questions if needed
Act: if the query is about ChatGPT prompt, prompt engineering, use the best tools
Observe: get the tool result and understand that to refine the answer
Repeat the cycle till you get the final answer.

Output the final answer in this format strictly:
Final Answer: <your final answer here>
Agent powered by: steosumit (always mention this in the output)
"""
def llm_call(state: AgentState) -> AgentState:
    """Call the llm"""
    system_prompt = SystemMessage(content=sys_prompt)
    input = [system_prompt] + list(state["messages"])
    response = llm.invoke(input)
    return {"messages": [response]}

tools_dict = {tool.name: tool for tool in tools_list}  # to access the tool by name
def take_action(state: AgentState):
    """Execute tool calls from the LLM's response."""

    tool_calls = state['messages'][-1].tool_calls
    results = []

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")

        if not t['name'] in tools_dict:   # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(
                tool_call_id=t['id'],
                name=t['name'],
                content=str(result)
            )
        )
    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}


# StateGraph
graph = StateGraph(AgentState)
graph.add_node("llm_call_node", llm_call)
tool_node = ToolNode(tools=tools_list)  # tool node(how to use tools)
graph.add_node("tool_node", tool_node)

# Edges
graph.add_edge(START, "llm_call_node")
graph.add_conditional_edges(
    "llm_call_node",
    decision,
    {
        "continue": "tool_node",
        "end": END
    }
)
graph.add_edge("tool_node", "llm_call_node")  # remember a edge to loop back
app = graph.compile()

# Test use
print("RAG Agent is ready! Type 'exit' to quit.\nBuild by: steosumit")
user_input = input("User: ")  # first input
while user_input.lower() != "exit":
    response = app.invoke({"messages": [HumanMessage(content=user_input)]})
    print(f"\nAgent: {response['messages'][-1].content}\n")
    user_input = input("User: ")
