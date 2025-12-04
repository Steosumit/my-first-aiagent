# Notes and Learnings
## Code Learning

### Why conversaton_history = response["messages]?
```
conversation_history = []  # memory
user_input = input("User: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    response = llm.invoke({"messages": conversation_history})  # state is a dict after all
    # NOTE: here state["messages"] is changed from HumanMessage to AIMessage
    print(response["messages"])
    conversation_history = response["messages"]
```
This happens as the llm object return the full
list of messages including both user and AI messages in the response.
```aiignore
[
  HumanMessage("first user input"),
  AIMessage("first model reply"),
  HumanMessage("second user input"),
  AIMessage("second model reply"),
  ...
]

```

### How add_message solve the memory problem 
in 
```aiignore
class AgentState(TypeDict):
    messages: Annotated[Sequence[BaseMessage], add_message]
```
Here is the things:
```aiignore
old_value = state["messages"]
new_value = node_output["messages"]

merged = add_messages(old_value, new_value)

```

```aiignore
def add_messages(existing, new):
    if existing is None:
        return new
    if new is None:
        return existing
    return list(existing) + list(new)
```

### Why we need to re init the llm agent upon binding 
```aiignore
tools = [retrieval_tool]
llm_with_tools = llm.bind_tools(tools)
```
Yes, you can bind tools directly using llm.bind_tools(),
but you need to assign the result back to a variable since it returns a new instance