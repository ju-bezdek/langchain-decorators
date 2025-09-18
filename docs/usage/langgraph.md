# Concepts

## Graphs
### Sequential  
- all nodes in sequential graph are executed in sequence...  
- if node in sequence is subgraph, or LLM node... if this will finish.. the next is triggered immediately... composing a pipeline of tasks to be executed

### Staged
- each stage is guarded and to get to next stage, specific command "__next__" has to be executed... 
this means "go to the next stage" ... whatever it is 

- this is ideal for pipeline of agents, each having capability of talking to user... 

- for instance stages: analysis, development, testing... each agent has differnt stills, can share some context... at first only analysis agent is activated... unless stage analysis is finished, we only interact with this agent

... this type of graph encapsulates quite common pattern, where, we need to put a router before agents which it always routing to current stage... and then we need to implement also way for the agents to let the parent graph know, that they are done... and we can move next... 
this encapsulation provides easy to set, and easy to read setup... without the need thrashing the inner agent with the knowledge of the parent graph



## LlmNode
Can use thought of simple agent, but can be more complex... i.e. conditional changing prompts
- allows you to define subnodes, conditional edges and tools ... and package it all into one class with its own sub-state ... 

- includes messages key natively


##

# State schema
    define state via typing

    i.e.:

    class MyExplicitlyDefinedGraphSchema:
        messages: Annotated[list[AnyMessage], add_messages] # to share llm chat messages between stages
        custom_field_a: str

    StagedGraph[MyExplicitlyDefinedGraphSchema].start().then( ... )
