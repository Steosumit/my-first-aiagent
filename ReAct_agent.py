"""
Here I am building a ReAct agent
Packages:
Annotated, Sequence, BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage, ToolNode
add_message
"""

from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import  add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Tool (ReAct agents need tools to Act)
@tool
def add(a: int, b: int) -> int:  # Action
    """This function adds two numbers"""  # doc string is must for LLM to identify the tool
    return a + b

# LLM
tools = [add]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash").bind_tools(tools)

# Nodes
def llm_call(state: AgentState) -> AgentState:  # Reasoning and Planning
    """Call the LLM"""
    system_prompt = SystemMessage(
        content="You are a helpful assistant. Answer the question to the best way possible"
    )
    response = llm.invoke([system_prompt] + state["messages"])  # add the HumanMessage with the sys prommpt
    return {"messages": [response]}  # return the whole response
    # NOTE: that the above is a AgentState response

def decision(state: AgentState)-> AgentState:  # Observation
    """Decide whether to continue or end based on the last message"""
    message = state["messages"]
    last_message = message[-1]
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

# StateGraph
graph = StateGraph(AgentState)
graph.add_node("llm_call_node", llm_call)
tool_node = ToolNode(tools=tools)  # tool node(how to use tools)
graph.add_node("tool_node", tool_node)

# Edges
graph.add_edge(START, "llm_call_node")
graph.add_conditional_edges(
    "llm_call_node",
    decision,
    {
        "continue": "tool_node",
        "end": END,
    },
)
graph.add_edge("tool_node", "llm_call_node")  # remember a edge to loop back to llm_call_node

app = graph.compile()

# Test use
input_user = input("User: ")
result = app.invoke({
    "messages": [HumanMessage(content=input_user)]
})
print(result['messages'][-1])  # final answer from the agent