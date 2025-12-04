"""
Here I will be playing with a memory  agent code
Packages:
Typing, HumanMessage, StateGraph,
"""

from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# State
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Nodes
def process(state: AgentState) -> AgentState:
    """Process the input from the user and make a response"""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

# StateGraph
graph = StateGraph(AgentState)
graph.add_node("process_node", process)
# Edges
graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

agent = graph.compile()

# Test use
conversation_history = []  # memory
user_input = input("User: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    response = agent.invoke({"messages": conversation_history})  # state is a dict after all
    # NOTE: here state["messages"] is changed from HumanMessage to AIMessage
    print(response["messages"])
    conversation_history = response["messages"]  # full list of the prev convo returned
    user_input = input("User: ")

