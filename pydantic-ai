PydanticAI

# Introduction
_Agent Framework / shim to use Pydantic with LLMs_
PydanticAI is a Python agent framework designed to make it less painful to build production grade applications with Generative AI. 
FastAPI revolutionized web development by offering an innovative and ergonomic design, built on the foundation of [Pydantic](https://docs.pydantic.dev).
Similarly, virtually every agent framework and LLM library in Python uses Pydantic, yet when we began to use LLMs in [Pydantic Logfire](https://pydantic.dev/logfire), we couldn't find anything that gave us the same feeling.
We built PydanticAI with one simple aim: to bring that FastAPI feeling to GenAI app development.
## Why use PydanticAI
  * **Built by the Pydantic Team** : Built by the team behind [Pydantic](https://docs.pydantic.dev/latest/) (the validation layer of the OpenAI SDK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more).
  * **Model-agnostic** : Supports OpenAI, Anthropic, Gemini, Deepseek, Ollama, Groq, Cohere, and Mistral, and there is a simple interface to implement support for [other models](https://ai.pydantic.dev/models/).
  * **Pydantic Logfire Integration** : Seamlessly [integrates](https://ai.pydantic.dev/logfire/) with [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, performance monitoring, and behavior tracking of your LLM-powered applications.
  * **Type-safe** : Designed to make [type checking](https://ai.pydantic.dev/agents/#static-type-checking) as powerful and informative as possible for you.
  * **Python-centric Design** : Leverages Python's familiar control flow and agent composition to build your AI-driven projects, making it easy to apply standard Python best practices you'd use in any other (non-AI) project.
  * **Structured Responses** : Harnesses the power of [Pydantic](https://docs.pydantic.dev/latest/) to [validate and structure](https://ai.pydantic.dev/results/#structured-result-validation) model outputs, ensuring responses are consistent across runs.
  * **Dependency Injection System** : Offers an optional [dependency injection](https://ai.pydantic.dev/dependencies/) system to provide data and services to your agent's [system prompts](https://ai.pydantic.dev/agents/#system-prompts), [tools](https://ai.pydantic.dev/tools/) and [result validators](https://ai.pydantic.dev/results/#result-validators-functions). This is useful for testing and eval-driven iterative development.
  * **Streamed Responses** : Provides the ability to [stream](https://ai.pydantic.dev/results/#streamed-results) LLM outputs continuously, with immediate validation, ensuring rapid and accurate results.
  * **Graph Support** : [Pydantic Graph](https://ai.pydantic.dev/graph/) provides a powerful way to define graphs using typing hints, this is useful in complex applications where standard control flow can degrade to spaghetti code.


## Hello World Example
Here's a minimal example of PydanticAI:
hello_world.py```
frompydantic_aiimport Agent
agent = Agent( 
  'google-gla:gemini-1.5-flash',
  system_prompt='Be concise, reply with one sentence.', 
)
result = agent.run_sync('Where does "hello world" come from?') 
print(result.data)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""

```

_(This example is complete, it can be run "as is")_
The exchange should be very short: PydanticAI will send the system prompt and the user query to the LLM, the model will return a text response.
Not very interesting yet, but we can easily add "tools", dynamic system prompts, and structured responses to build more powerful agents.
## Tools & Dependency Injection Example
Here is a concise example using PydanticAI to build a support agent for a bank:
bank_support.py```
fromdataclassesimport dataclass
frompydanticimport BaseModel, Field
frompydantic_aiimport Agent, RunContext
frombank_databaseimport DatabaseConn

@dataclass
classSupportDependencies: 
  customer_id: int
  db: DatabaseConn 

classSupportResult(BaseModel): 
  support_advice: str = Field(description='Advice returned to the customer')
  block_card: bool = Field(description="Whether to block the customer's card")
  risk: int = Field(description='Risk level of query', ge=0, le=10)

support_agent = Agent( 
  'openai:gpt-4o', 
  deps_type=SupportDependencies,
  result_type=SupportResult, 
  system_prompt=( 
    'You are a support agent in our bank, give the '
    'customer support and judge the risk level of their query.'
  ),
)

@support_agent.system_prompt 
async defadd_customer_name(ctx: RunContext[SupportDependencies]) -> str:
  customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
  return f"The customer's name is {customer_name!r}"

@support_agent.tool 
async defcustomer_balance(
  ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
"""Returns the customer's current account balance.""" 
  return await ctx.deps.db.customer_balance(
    id=ctx.deps.customer_id,
    include_pending=include_pending,
  )

... 

async defmain():
  deps = SupportDependencies(customer_id=123, db=DatabaseConn())
  result = await support_agent.run('What is my balance?', deps=deps) 
  print(result.data) 
"""
  support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
  """
  result = await support_agent.run('I just lost my card!', deps=deps)
  print(result.data)
"""
  support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
  """

```

Complete `bank_support.py` example
The code included here is incomplete for the sake of brevity (the definition of `DatabaseConn` is missing); you can find the complete `bank_support.py` example [here](https://ai.pydantic.dev/examples/bank-support/).
## Instrumentation with Pydantic Logfire
To understand the flow of the above runs, we can watch the agent in action using Pydantic Logfire.
To do this, we need to set up logfire, and add the following to our code:
bank_support_with_logfire.py```
...
frompydantic_aiimport Agent, RunContext
frombank_databaseimport DatabaseConn
importlogfire
logfire.configure() 
logfire.instrument_asyncpg() 

...
support_agent = Agent(
  'openai:gpt-4o',
  deps_type=SupportDependencies,
  result_type=SupportResult,
  system_prompt=(
    'You are a support agent in our bank, give the '
    'customer support and judge the risk level of their query.'
  ),
  instrument=True,
)

```

That's enough to get the following view of your agent in action:
See [Monitoring and Performance](https://ai.pydantic.dev/logfire/) to learn more.
## Next Steps
To try PydanticAI yourself, follow the instructions [in the examples](https://ai.pydantic.dev/examples/).
Read the [docs](https://ai.pydantic.dev/agents/) to learn more about building applications with PydanticAI.
Read the [API Reference](https://ai.pydantic.dev/api/agent/) to understand PydanticAI's interface.

[ Skip to content ](https://ai.pydantic.dev/agents/#introduction)
# Agents
## Introduction
Agents are PydanticAI's primary interface for interacting with LLMs.
In some use cases a single Agent will control an entire application or component, but multiple agents can also interact to embody more complex workflows.
The [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent) class has full API documentation, but conceptually you can think of an agent as a container for:
**Component** | **Description**  
---|---  
[System prompt(s)](https://ai.pydantic.dev/agents/#system-prompts) | A set of instructions for the LLM written by the developer.  
[Function tool(s)](https://ai.pydantic.dev/tools/) | Functions that the LLM may call to get information while generating a response.  
[Structured result type](https://ai.pydantic.dev/results/) | The structured datatype the LLM must return at the end of a run, if specified.  
[Dependency type constraint](https://ai.pydantic.dev/dependencies/) | System prompt functions, tools, and result validators may all use dependencies when they're run.  
Optional default LLM model associated with the agent. Can also be specified when running the agent.  
Optional default model settings to help fine tune requests. Can also be specified when running the agent.  
In typing terms, agents are generic in their dependency and result types, e.g., an agent which required dependencies of type `Foobar` and returned results of type `list[str]` would have type `Agent[Foobar, list[str]]`. In practice, you shouldn't need to care about this, it should just mean your IDE can tell you when you have the right type, and if you choose to use [static type checking](https://ai.pydantic.dev/agents/#static-type-checking) it should work well with PydanticAI.
Here's a toy example of an agent that simulates a roulette wheel:
roulette_wheel.py```
frompydantic_aiimport Agent, RunContext
roulette_agent = Agent( 
  'openai:gpt-4o',
  deps_type=int,
  result_type=bool,
  system_prompt=(
    'Use the `roulette_wheel` function to see if the '
    'customer has won based on the number they provide.'
  ),
)

@roulette_agent.tool
async defroulette_wheel(ctx: RunContext[int], square: int) -> str: 
"""check if the square is a winner"""
  return 'winner' if square == ctx.deps else 'loser'

# Run the agent
success_number = 18 
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.data) 
#> True
result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.data)
#> False

```

Agents are designed for reuse, like FastAPI Apps
Agents are intended to be instantiated once (frequently as module globals) and reused throughout your application, similar to a small [FastAPI](https://fastapi.tiangolo.com/reference/fastapi/#fastapi.FastAPI) app or an [APIRouter](https://fastapi.tiangolo.com/reference/apirouter/#fastapi.APIRouter).
## Running Agents
There are four ways to run an agent:
  1. [`agent.run()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run) — a coroutine which returns a [`RunResult`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult) containing a completed response.
  2. [`agent.run_sync()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync) — a plain, synchronous function which returns a [`RunResult`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult) containing a completed response (internally, this just calls `loop.run_until_complete(self.run())`).
  3. [`agent.run_stream()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream) — a coroutine which returns a [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult), which contains methods to stream a response as an async iterable.
  4. [`agent.iter()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.iter) — a context manager which returns an [`AgentRun`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRun), an async-iterable over the nodes of the agent's underlying [`Graph`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph).


Here's a simple example demonstrating the first three:
run_agent.py```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.data)
#> Rome

async defmain():
  result = await agent.run('What is the capital of France?')
  print(result.data)
  #> Paris
  async with agent.run_stream('What is the capital of the UK?') as response:
    print(await response.get_data())
    #> London

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
You can also pass messages from previous runs to continue a conversation or provide context, as described in [Messages and Chat History](https://ai.pydantic.dev/message-history/).
### Iterating Over an Agent's Graph
Under the hood, each `Agent` in PydanticAI uses **pydantic-graph** to manage its execution flow. **pydantic-graph** is a generic, type-centric library for building and running finite state machines in Python. It doesn't actually depend on PydanticAI — you can use it standalone for workflows that have nothing to do with GenAI — but PydanticAI makes use of it to orchestrate the handling of model requests and model responses in an agent's run.
In many scenarios, you don't need to worry about pydantic-graph at all; calling `agent.run(...)` simply traverses the underlying graph from start to finish. However, if you need deeper insight or control — for example to capture each tool invocation, or to inject your own logic at specific stages — PydanticAI exposes the lower-level iteration process via [`Agent.iter`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.iter). This method returns an [`AgentRun`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRun), which you can async-iterate over, or manually drive node-by-node via the [`next`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRun.next) method. Once the agent's graph returns an [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End), you have the final result along with a detailed history of all steps.
#### `async for` iteration
Here's an example of using `async for` with `iter` to record each node the agent executes:
agent_iter_async_for.py```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')

async defmain():
  nodes = []
  # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
  async with agent.iter('What is the capital of France?') as agent_run:
    async for node in agent_run:
      # Each node represents a step in the agent's execution
      nodes.append(node)
  print(nodes)
"""

    ModelRequestNode(
      request=ModelRequest(
        parts=[
          UserPromptPart(
            content='What is the capital of France?',
            timestamp=datetime.datetime(...),
            part_kind='user-prompt',


        kind='request',

    ),
    CallToolsNode(
      model_response=ModelResponse(
        parts=[TextPart(content='Paris', part_kind='text')],
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
        kind='response',

    ),
    End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

  """
  print(agent_run.result.data)
  #> Paris

```

  * The `AgentRun` is an async iterator that yields each node (`BaseNode` or `End`) in the flow.
  * The run ends when an `End` node is returned.


#### Using `.next(...)` manually
You can also drive the iteration manually by passing the node you want to run next to the `AgentRun.next(...)` method. This allows you to inspect or modify the node before it executes or skip nodes based on your own logic, and to catch errors in `next()` more easily:
agent_iter_next.py```
frompydantic_aiimport Agent
frompydantic_graphimport End
agent = Agent('openai:gpt-4o')

async defmain():
  async with agent.iter('What is the capital of France?') as agent_run:
    node = agent_run.next_node 
    all_nodes = [node]
    # Drive the iteration manually:
    while not isinstance(node, End): 
      node = await agent_run.next(node) 
      all_nodes.append(node) 
    print(all_nodes)
"""

      UserPromptNode(
        user_prompt='What is the capital of France?',
        system_prompts=(),
        system_prompt_functions=[],
        system_prompt_dynamic_functions={},

      ModelRequestNode(
        request=ModelRequest(
          parts=[
            UserPromptPart(
              content='What is the capital of France?',
              timestamp=datetime.datetime(...),
              part_kind='user-prompt',


          kind='request',


      CallToolsNode(
        model_response=ModelResponse(
          parts=[TextPart(content='Paris', part_kind='text')],
          model_name='gpt-4o',
          timestamp=datetime.datetime(...),
          kind='response',


      End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

    """

```

#### Accessing usage and the final result
You can retrieve usage statistics (tokens, requests, etc.) at any time from the [`AgentRun`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRun) object via `agent_run.usage()`. This method returns a [`Usage`](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.Usage) object containing the usage data.
Once the run finishes, `agent_run.final_result` becomes a [`AgentRunResult`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult) object containing the final output (and related metadata).
### Streaming
Here is an example of streaming an agent run in combination with `async for` iteration:
streaming.py```
importasyncio
fromdataclassesimport dataclass
fromdatetimeimport date
frompydantic_aiimport Agent
frompydantic_ai.messagesimport (
  FinalResultEvent,
  FunctionToolCallEvent,
  FunctionToolResultEvent,
  PartDeltaEvent,
  PartStartEvent,
  TextPartDelta,
  ToolCallPartDelta,
)
frompydantic_ai.toolsimport RunContext

@dataclass
classWeatherService:
  async defget_forecast(self, location: str, forecast_date: date) -> str:
    # In real code: call weather API, DB queries, etc.
    return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'
  async defget_historic_weather(self, location: str, forecast_date: date) -> str:
    # In real code: call a historical weather API or DB
    return (
      f'The weather in {location} on {forecast_date} was 18°C and partly cloudy.'
    )

weather_agent = Agent[WeatherService, str](
  'openai:gpt-4o',
  deps_type=WeatherService,
  result_type=str, # We'll produce a final answer as plain text
  system_prompt='Providing a weather forecast at the locations the user provides.',
)

@weather_agent.tool
async defweather_forecast(
  ctx: RunContext[WeatherService],
  location: str,
  forecast_date: date,
) -> str:
  if forecast_date >= date.today():
    return await ctx.deps.get_forecast(location, forecast_date)
  else:
    return await ctx.deps.get_historic_weather(location, forecast_date)

output_messages: list[str] = []

async defmain():
  user_prompt = 'What will the weather be like in Paris on Tuesday?'
  # Begin a node-by-node, streaming iteration
  async with weather_agent.iter(user_prompt, deps=WeatherService()) as run:
    async for node in run:
      if Agent.is_user_prompt_node(node):
        # A user prompt node => The user has provided input
        output_messages.append(f'=== UserPromptNode: {node.user_prompt} ===')
      elif Agent.is_model_request_node(node):
        # A model request node => We can stream tokens from the model's request
        output_messages.append(
          '=== ModelRequestNode: streaming partial request tokens ==='
        )
        async with node.stream(run.ctx) as request_stream:
          async for event in request_stream:
            if isinstance(event, PartStartEvent):
              output_messages.append(
                f'[Request] Starting part {event.index}: {event.part!r}'
              )
            elif isinstance(event, PartDeltaEvent):
              if isinstance(event.delta, TextPartDelta):
                output_messages.append(
                  f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}'
                )
              elif isinstance(event.delta, ToolCallPartDelta):
                output_messages.append(
                  f'[Request] Part {event.index} args_delta={event.delta.args_delta}'
                )
            elif isinstance(event, FinalResultEvent):
              output_messages.append(
                f'[Result] The model produced a final result (tool_name={event.tool_name})'
              )
      elif Agent.is_call_tools_node(node):
        # A handle-response node => The model returned some data, potentially calls a tool
        output_messages.append(
          '=== CallToolsNode: streaming partial response & tool usage ==='
        )
        async with node.stream(run.ctx) as handle_stream:
          async for event in handle_stream:
            if isinstance(event, FunctionToolCallEvent):
              output_messages.append(
                f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
              )
            elif isinstance(event, FunctionToolResultEvent):
              output_messages.append(
                f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}'
              )
      elif Agent.is_end_node(node):
        assert run.result.data == node.data.data
        # Once an End node is reached, the agent run is complete
        output_messages.append(f'=== Final Agent Output: {run.result.data} ===')

if __name__ == '__main__':
  asyncio.run(main())
  print(output_messages)
"""

    '=== ModelRequestNode: streaming partial request tokens ===',
    '[Request] Starting part 0: ToolCallPart(tool_name=\'weather_forecast\', args=\'{"location":"Pa\', tool_call_id=\'0001\', part_kind=\'tool-call\')',
    '[Request] Part 0 args_delta=ris","forecast_',
    '[Request] Part 0 args_delta=date":"2030-01-',
    '[Request] Part 0 args_delta=01"}',
    '=== CallToolsNode: streaming partial response & tool usage ===',
    '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
    "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24°C and sunny.",
    '=== ModelRequestNode: streaming partial request tokens ===',
    "[Request] Starting part 0: TextPart(content='It will be ', part_kind='text')",
    '[Result] The model produced a final result (tool_name=None)',
    "[Request] Part 0 text delta: 'warm and sunny '",
    "[Request] Part 0 text delta: 'in Paris on '",
    "[Request] Part 0 text delta: 'Tuesday.'",
    '=== CallToolsNode: streaming partial response & tool usage ===',
    '=== Final Agent Output: It will be warm and sunny in Paris on Tuesday. ===',

  """

```

### Additional Configuration
#### Usage Limits
PydanticAI offers a [`UsageLimits`](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits) structure to help you limit your usage (tokens and/or requests) on model runs.
You can apply these settings by passing the `usage_limits` argument to the `run{_sync,_stream}` functions.
Consider the following example, where we limit the number of response tokens:
```
frompydantic_aiimport Agent
frompydantic_ai.exceptionsimport UsageLimitExceeded
frompydantic_ai.usageimport UsageLimits
agent = Agent('anthropic:claude-3-5-sonnet-latest')
result_sync = agent.run_sync(
  'What is the capital of Italy? Answer with just the city.',
  usage_limits=UsageLimits(response_tokens_limit=10),
)
print(result_sync.data)
#> Rome
print(result_sync.usage())
"""
Usage(requests=1, request_tokens=62, response_tokens=1, total_tokens=63, details=None)
"""
try:
  result_sync = agent.run_sync(
    'What is the capital of Italy? Answer with a paragraph.',
    usage_limits=UsageLimits(response_tokens_limit=10),
  )
except UsageLimitExceeded as e:
  print(e)
  #> Exceeded the response_tokens_limit of 10 (response_tokens=32)

```

Restricting the number of requests can be useful in preventing infinite loops or excessive tool calling:
```
fromtyping_extensionsimport TypedDict
frompydantic_aiimport Agent, ModelRetry
frompydantic_ai.exceptionsimport UsageLimitExceeded
frompydantic_ai.usageimport UsageLimits

classNeverResultType(TypedDict):
"""
  Never ever coerce data to this type.
  """
  never_use_this: str

agent = Agent(
  'anthropic:claude-3-5-sonnet-latest',
  retries=3,
  result_type=NeverResultType,
  system_prompt='Any time you get a response, call the `infinite_retry_tool` to produce another response.',
)

@agent.tool_plain(retries=5) 
definfinite_retry_tool() -> int:
  raise ModelRetry('Please try again.')

try:
  result_sync = agent.run_sync(
    'Begin infinite retry loop!', usage_limits=UsageLimits(request_limit=3) 
  )
except UsageLimitExceeded as e:
  print(e)
  #> The next request would exceed the request_limit of 3

```

Note
This is especially relevant if you've registered many tools. The `request_limit` can be used to prevent the model from calling them in a loop too many times.
#### Model (Run) Settings
PydanticAI offers a [`settings.ModelSettings`](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings) structure to help you fine tune your requests. This structure allows you to configure common parameters that influence the model's behavior, such as `temperature`, `max_tokens`, `timeout`, and more.
There are two ways to apply these settings: 1. Passing to `run{_sync,_stream}` functions via the `model_settings` argument. This allows for fine-tuning on a per-request basis. 2. Setting during [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent) initialization via the `model_settings` argument. These settings will be applied by default to all subsequent run calls using said agent. However, `model_settings` provided during a specific run call will override the agent's default settings.
For example, if you'd like to set the `temperature` setting to `0.0` to ensure less random behavior, you can do the following:
```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
result_sync = agent.run_sync(
  'What is the capital of Italy?', model_settings={'temperature': 0.0}
)
print(result_sync.data)
#> Rome

```

### Model specific settings
If you wish to further customize model behavior, you can use a subclass of [`ModelSettings`](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings), like [`GeminiModelSettings`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModelSettings), associated with your model of choice.
For example:
```
frompydantic_aiimport Agent, UnexpectedModelBehavior
frompydantic_ai.models.geminiimport GeminiModelSettings
agent = Agent('google-gla:gemini-1.5-flash')
try:
  result = agent.run_sync(
    'Write a list of 5 very rude things that I might say to the universe after stubbing my toe in the dark:',
    model_settings=GeminiModelSettings(
      temperature=0.0, # general model settings can also be specified
      gemini_safety_settings=[
        {
          'category': 'HARM_CATEGORY_HARASSMENT',
          'threshold': 'BLOCK_LOW_AND_ABOVE',
        },
        {
          'category': 'HARM_CATEGORY_HATE_SPEECH',
          'threshold': 'BLOCK_LOW_AND_ABOVE',
        },
      ],
    ),
  )
except UnexpectedModelBehavior as e:
  print(e) 
"""
  Safety settings triggered, body:
  <safety settings details>
  """

```

## Runs vs. Conversations
An agent **run** might represent an entire conversation — there's no limit to how many messages can be exchanged in a single run. However, a **conversation** might also be composed of multiple runs, especially if you need to maintain state between separate interactions or API calls.
Here's an example of a conversation comprised of multiple runs:
conversation_example.py```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.data)
#> Albert Einstein was a German-born theoretical physicist.
# Second run, passing previous messages
result2 = agent.run_sync(
  'What was his most famous equation?',
  message_history=result1.new_messages(), 
)
print(result2.data)
#> Albert Einstein's most famous equation is (E = mc^2).

```

_(This example is complete, it can be run "as is")_
## Type safe by design
PydanticAI is designed to work well with static type checkers, like mypy and pyright.
Typing is (somewhat) optional
PydanticAI is designed to make type checking as useful as possible for you if you choose to use it, but you don't have to use types everywhere all the time.
That said, because PydanticAI uses Pydantic, and Pydantic uses type hints as the definition for schema and validation, some types (specifically type hints on parameters to tools, and the `result_type` arguments to [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent)) are used at runtime.
We (the library developers) have messed up if type hints are confusing you more than helping you, if you find this, please create an [issue](https://github.com/pydantic/pydantic-ai/issues) explaining what's annoying you!
In particular, agents are generic in both the type of their dependencies and the type of results they return, so you can use the type hints to ensure you're using the right types.
Consider the following script with type mistakes:
type_mistakes.py```
fromdataclassesimport dataclass
frompydantic_aiimport Agent, RunContext

@dataclass
classUser:
  name: str

agent = Agent(
  'test',
  deps_type=User, 
  result_type=bool,
)

@agent.system_prompt
defadd_user_name(ctx: RunContext[str]) -> str: 
  return f"The user's name is {ctx.deps}."

deffoobar(x: bytes) -> None:
  pass

result = agent.run_sync('Does their name start with "A"?', deps=User('Anne'))
foobar(result.data) 

```

Running `mypy` on this will give the following output:
```
➤uvrunmypytype_mistakes.py
type_mistakes.py:18:error:Argument1to"system_prompt"of"Agent"hasincompatibletype"Callable[[RunContext[str]], str]";expected"Callable[[RunContext[User]], str]"[arg-type]
type_mistakes.py:28:error:Argument1to"foobar"hasincompatibletype"bool";expected"bytes"[arg-type]
Found2errorsin1file(checked1sourcefile)

```

Running `pyright` would identify the same issues.
## System Prompts
System prompts might seem simple at first glance since they're just strings (or sequences of strings that are concatenated), but crafting the right system prompt is key to getting the model to behave as you want.
Generally, system prompts fall into two categories:
  1. **Static system prompts** : These are known when writing the code and can be defined via the `system_prompt` parameter of the [`Agent` constructor](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__).
  2. **Dynamic system prompts** : These depend in some way on context that isn't known until runtime, and should be defined via functions decorated with [`@agent.system_prompt`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.system_prompt).


You can add both to a single agent; they're appended in the order they're defined at runtime.
Here's an example using both types of system prompts:
system_prompts.py```
fromdatetimeimport date
frompydantic_aiimport Agent, RunContext
agent = Agent(
  'openai:gpt-4o',
  deps_type=str, 
  system_prompt="Use the customer's name while replying to them.", 
)

@agent.system_prompt 
defadd_the_users_name(ctx: RunContext[str]) -> str:
  return f"The user's name is {ctx.deps}."

@agent.system_prompt
defadd_the_date() -> str: 
  return f'The date is {date.today()}.'

result = agent.run_sync('What is the date?', deps='Frank')
print(result.data)
#> Hello Frank, the date today is 2032-01-02.

```

_(This example is complete, it can be run "as is")_
## Reflection and self-correction
Validation errors from both function tool parameter validation and [structured result validation](https://ai.pydantic.dev/results/#structured-result-validation) can be passed back to the model with a request to retry.
You can also raise [`ModelRetry`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.ModelRetry) from within a [tool](https://ai.pydantic.dev/tools/) or [result validator function](https://ai.pydantic.dev/results/#result-validators-functions) to tell the model it should retry generating a response.
  * The default retry count is **1** but can be altered for the [entire agent](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__), a [specific tool](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool), or a [result validator](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__).
  * You can access the current retry count from within a tool or result validator via [`ctx.retry`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext).


Here's an example:
tool_retry.py```
frompydanticimport BaseModel
frompydantic_aiimport Agent, RunContext, ModelRetry
fromfake_databaseimport DatabaseConn

classChatResult(BaseModel):
  user_id: int
  message: str

agent = Agent(
  'openai:gpt-4o',
  deps_type=DatabaseConn,
  result_type=ChatResult,
)

@agent.tool(retries=2)
defget_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
"""Get a user's ID from their full name."""
  print(name)
  #> John
  #> John Doe
  user_id = ctx.deps.users.get(name=name)
  if user_id is None:
    raise ModelRetry(
      f'No user found with name {name!r}, remember to provide their full name'
    )
  return user_id

result = agent.run_sync(
  'Send a message to John Doe asking for coffee next week', deps=DatabaseConn()
)
print(result.data)
"""
user_id=123 message='Hello John, would you be free for coffee sometime next week? Let me know what works for you!'
"""

```

## Model errors
If models behave unexpectedly (e.g., the retry limit is exceeded, or their API returns `503`), agent runs will raise [`UnexpectedModelBehavior`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.UnexpectedModelBehavior).
In these cases, [`capture_run_messages`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.capture_run_messages) can be used to access the messages exchanged during the run to help diagnose the issue.
agent_model_errors.py```
frompydantic_aiimport Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages
agent = Agent('openai:gpt-4o')

@agent.tool_plain
defcalc_volume(size: int) -> int: 
  if size == 42:
    return size**3
  else:
    raise ModelRetry('Please try again.')

with capture_run_messages() as messages: 
  try:
    result = agent.run_sync('Please get me the volume of a box with size 6.')
  except UnexpectedModelBehavior as e:
    print('An error occurred:', e)
    #> An error occurred: Tool exceeded max retries count of 1
    print('cause:', repr(e.__cause__))
    #> cause: ModelRetry('Please try again.')
    print('messages:', messages)
"""
    messages:

      ModelRequest(
        parts=[
          UserPromptPart(
            content='Please get me the volume of a box with size 6.',
            timestamp=datetime.datetime(...),
            part_kind='user-prompt',


        kind='request',

      ModelResponse(
        parts=[
          ToolCallPart(
            tool_name='calc_volume',
            args={'size': 6},
            tool_call_id='pyd_ai_tool_call_id',
            part_kind='tool-call',


        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
        kind='response',

      ModelRequest(
        parts=[
          RetryPromptPart(
            content='Please try again.',
            tool_name='calc_volume',
            tool_call_id='pyd_ai_tool_call_id',
            timestamp=datetime.datetime(...),
            part_kind='retry-prompt',


        kind='request',

      ModelResponse(
        parts=[
          ToolCallPart(
            tool_name='calc_volume',
            args={'size': 6},
            tool_call_id='pyd_ai_tool_call_id',
            part_kind='tool-call',


        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
        kind='response',


    """
  else:
    print(result.data)

```

_(This example is complete, it can be run "as is")_
Note
If you call [`run`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run), [`run_sync`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync), or [`run_stream`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream) more than once within a single `capture_run_messages` context, `messages` will represent the messages exchanged during the first call only.

# `pydantic_ai.agent`
###  Agent `dataclass`
Bases: `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")]`
Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.
Agents are generic in the dependency type they take [`AgentDepsT`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT) and the result data type they return, [`ResultDataT`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT).
By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.
Minimal usage example:
```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
result = agent.run_sync('What is the capital of France?')
print(result.data)
#> Paris

```

Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@final
@dataclasses.dataclass(init=False)
classAgent(Generic[AgentDepsT, ResultDataT]):
"""Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.
  Agents are generic in the dependency type they take [`AgentDepsT`][pydantic_ai.tools.AgentDepsT]
  and the result data type they return, [`ResultDataT`][pydantic_ai.result.ResultDataT].
  By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.
  Minimal usage example:
  ```python
  from pydantic_ai import Agent
  agent = Agent('openai:gpt-4o')
  result = agent.run_sync('What is the capital of France?')
  print(result.data)
  #> Paris
  ```
  """
  # we use dataclass fields in order to conveniently know what attributes are available
  model: models.Model | models.KnownModelName | None
"""The default model configured for this agent."""
  name: str | None
"""The name of the agent, used for logging.
  If `None`, we try to infer the agent name from the call frame when the agent is first run.
  """
  end_strategy: EndStrategy
"""Strategy for handling tool calls when a final result is found."""
  model_settings: ModelSettings | None
"""Optional model request settings to use for this agents's runs, by default.
  Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will
  be merged with this value, with the runtime argument taking priority.
  """
  result_type: type[ResultDataT] = dataclasses.field(repr=False)
"""
  The type of the result data, used to validate the result data, defaults to `str`.
  """
  instrument: InstrumentationSettings | bool | None
"""Options to automatically instrument with OpenTelemetry."""
  _instrument_default: ClassVar[InstrumentationSettings | bool] = False
  _deps_type: type[AgentDepsT] = dataclasses.field(repr=False)
  _result_tool_name: str = dataclasses.field(repr=False)
  _result_tool_description: str | None = dataclasses.field(repr=False)
  _result_schema: _result.ResultSchema[ResultDataT] | None = dataclasses.field(repr=False)
  _result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]] = dataclasses.field(repr=False)
  _system_prompts: tuple[str, ...] = dataclasses.field(repr=False)
  _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
  _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(
    repr=False
  )
  _function_tools: dict[str, Tool[AgentDepsT]] = dataclasses.field(repr=False)
  _mcp_servers: Sequence[MCPServer] = dataclasses.field(repr=False)
  _default_retries: int = dataclasses.field(repr=False)
  _max_result_retries: int = dataclasses.field(repr=False)
  _override_deps: _utils.Option[AgentDepsT] = dataclasses.field(default=None, repr=False)
  _override_model: _utils.Option[models.Model] = dataclasses.field(default=None, repr=False)
  def__init__(
    self,
    model: models.Model | models.KnownModelName | None = None,
    *,
    result_type: type[ResultDataT] = str,
    system_prompt: str | Sequence[str] = (),
    deps_type: type[AgentDepsT] = NoneType,
    name: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 1,
    result_tool_name: str = 'final_result',
    result_tool_description: str | None = None,
    result_retries: int | None = None,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
    mcp_servers: Sequence[MCPServer] = (),
    defer_model_check: bool = False,
    end_strategy: EndStrategy = 'early',
    instrument: InstrumentationSettings | bool | None = None,
  ):
"""Create an agent.
    Args:
      model: The default model to use for this agent, if not provide,
        you must provide the model when calling it.
      result_type: The type of the result data, used to validate the result data, defaults to `str`.
      system_prompt: Static system prompts to use for this agent, you can also register system
        prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
      deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
        parameterize the agent, and therefore get the best out of static type checking.
        If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
        or add a type hint `: Agent[None, <return type>]`.
      name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
        when the agent is first run.
      model_settings: Optional model request settings to use for this agent's runs, by default.
      retries: The default number of retries to allow before raising an error.
      result_tool_name: The name of the tool to use for the final result.
      result_tool_description: The description of the final result tool.
      result_retries: The maximum number of retries to allow for result validation, defaults to `retries`.
      tools: Tools to register with the agent, you can also register tools via the decorators
        [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
      mcp_servers: MCP servers to register with the agent. You should register a [`MCPServer`][pydantic_ai.mcp.MCPServer]
        for each server you want the agent to connect to.
      defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
        it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
        which checks for the necessary environment variables. Set this to `false`
        to defer the evaluation until the first run. Useful if you want to
        [override the model][pydantic_ai.Agent.override] for testing.
      end_strategy: Strategy for handling tool calls that are requested alongside a final result.
        See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
      instrument: Set to True to automatically instrument with OpenTelemetry,
        which will use Logfire if it's configured.
        Set to an instance of [`InstrumentationSettings`][pydantic_ai.agent.InstrumentationSettings] to customize.
        If this isn't set, then the last value set by
        [`Agent.instrument_all()`][pydantic_ai.Agent.instrument_all]
        will be used, which defaults to False.
        See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
    """
    if model is None or defer_model_check:
      self.model = model
    else:
      self.model = models.infer_model(model)
    self.end_strategy = end_strategy
    self.name = name
    self.model_settings = model_settings
    self.result_type = result_type
    self.instrument = instrument
    self._deps_type = deps_type
    self._result_tool_name = result_tool_name
    self._result_tool_description = result_tool_description
    self._result_schema: _result.ResultSchema[ResultDataT] | None = _result.ResultSchema[result_type].build(
      result_type, result_tool_name, result_tool_description
    )
    self._result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]] = []
    self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
    self._system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = []
    self._system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = {}
    self._function_tools: dict[str, Tool[AgentDepsT]] = {}
    self._default_retries = retries
    self._max_result_retries = result_retries if result_retries is not None else retries
    self._mcp_servers = mcp_servers
    for tool in tools:
      if isinstance(tool, Tool):
        self._register_tool(tool)
      else:
        self._register_tool(Tool(tool))
  @staticmethod
  definstrument_all(instrument: InstrumentationSettings | bool = True) -> None:
"""Set the instrumentation options for all agents where `instrument` is not set."""
    Agent._instrument_default = instrument
  @overload
  async defrun(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: None = None,
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AgentRunResult[ResultDataT]: ...
  @overload
  async defrun(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: type[RunResultDataT],
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AgentRunResult[RunResultDataT]: ...
  async defrun(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: type[RunResultDataT] | None = None,
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AgentRunResult[Any]:
"""Run the agent with a user prompt in async mode.
    This method builds an internal agent graph (using system prompts, tools and result schemas) and then
    runs the graph to completion. The result of the run is returned.
    Example:
    ```python
    from pydantic_ai import Agent
    agent = Agent('openai:gpt-4o')
    async def main():
      agent_run = await agent.run('What is the capital of France?')
      print(agent_run.data)
      #> Paris
    ```
    Args:
      user_prompt: User input to start/continue the conversation.
      result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
        result validators since result validators would expect an argument that matches the agent's result type.
      message_history: History of the conversation so far.
      model: Optional model to use for this run, required if `model` was not set when creating the agent.
      deps: Optional dependencies to use for this run.
      model_settings: Optional settings to use for this model's request.
      usage_limits: Optional limits on model request count or token usage.
      usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
      infer_name: Whether to try to infer the agent name from the call frame if it's not set.
    Returns:
      The result of the run.
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    async with self.iter(
      user_prompt=user_prompt,
      result_type=result_type,
      message_history=message_history,
      model=model,
      deps=deps,
      model_settings=model_settings,
      usage_limits=usage_limits,
      usage=usage,
    ) as agent_run:
      async for _ in agent_run:
        pass
    assert (final_result := agent_run.result) is not None, 'The graph run did not finish properly'
    return final_result
  @asynccontextmanager
  async defiter(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: type[RunResultDataT] | None = None,
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
"""A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.
    This method builds an internal agent graph (using system prompts, tools and result schemas) and then returns an
    `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
    executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
    stream of events coming from the execution of tools.
    The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
    and the final result of the run once it has completed.
    For more details, see the documentation of `AgentRun`.
    Example:
    ```python
    from pydantic_ai import Agent
    agent = Agent('openai:gpt-4o')
    async def main():
      nodes = []
      async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
          nodes.append(node)
      print(nodes)
      '''

        ModelRequestNode(
          request=ModelRequest(
            parts=[
              UserPromptPart(
                content='What is the capital of France?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',


            kind='request',


        CallToolsNode(
          model_response=ModelResponse(
            parts=[TextPart(content='Paris', part_kind='text')],
            model_name='gpt-4o',
            timestamp=datetime.datetime(...),
            kind='response',


        End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

      '''
      print(agent_run.result.data)
      #> Paris
    ```
    Args:
      user_prompt: User input to start/continue the conversation.
      result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
        result validators since result validators would expect an argument that matches the agent's result type.
      message_history: History of the conversation so far.
      model: Optional model to use for this run, required if `model` was not set when creating the agent.
      deps: Optional dependencies to use for this run.
      model_settings: Optional settings to use for this model's request.
      usage_limits: Optional limits on model request count or token usage.
      usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
      infer_name: Whether to try to infer the agent name from the call frame if it's not set.
    Returns:
      The result of the run.
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    model_used = self._get_model(model)
    del model
    deps = self._get_deps(deps)
    new_message_index = len(message_history) if message_history else 0
    result_schema: _result.ResultSchema[RunResultDataT] | None = self._prepare_result_schema(result_type)
    # Build the graph
    graph = self._build_graph(result_type)
    # Build the initial state
    state = _agent_graph.GraphAgentState(
      message_history=message_history[:] if message_history else [],
      usage=usage or _usage.Usage(),
      retries=0,
      run_step=0,
    )
    # We consider it a user error if a user tries to restrict the result type while having a result validator that
    # may change the result type from the restricted type to something else. Therefore, we consider the following
    # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
    result_validators = cast(list[_result.ResultValidator[AgentDepsT, RunResultDataT]], self._result_validators)
    # TODO: Instead of this, copy the function tools to ensure they don't share current_retry state between agent
    # runs. Requires some changes to `Tool` to make them copyable though.
    for v in self._function_tools.values():
      v.current_retry = 0
    model_settings = merge_model_settings(self.model_settings, model_settings)
    usage_limits = usage_limits or _usage.UsageLimits()
    if isinstance(model_used, InstrumentedModel):
      tracer = model_used.settings.tracer
    else:
      tracer = NoOpTracer()
    agent_name = self.name or 'agent'
    run_span = tracer.start_span(
      'agent run',
      attributes={
        'model_name': model_used.model_name if model_used else 'no-model',
        'agent_name': agent_name,
        'logfire.msg': f'{agent_name} run',
      },
    )
    graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, RunResultDataT](
      user_deps=deps,
      prompt=user_prompt,
      new_message_index=new_message_index,
      model=model_used,
      model_settings=model_settings,
      usage_limits=usage_limits,
      max_result_retries=self._max_result_retries,
      end_strategy=self.end_strategy,
      result_schema=result_schema,
      result_tools=self._result_schema.tool_defs() if self._result_schema else [],
      result_validators=result_validators,
      function_tools=self._function_tools,
      mcp_servers=self._mcp_servers,
      run_span=run_span,
      tracer=tracer,
    )
    start_node = _agent_graph.UserPromptNode[AgentDepsT](
      user_prompt=user_prompt,
      system_prompts=self._system_prompts,
      system_prompt_functions=self._system_prompt_functions,
      system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
    )
    async with graph.iter(
      start_node,
      state=state,
      deps=graph_deps,
      span=use_span(run_span, end_on_exit=True),
      infer_name=False,
    ) as graph_run:
      yield AgentRun(graph_run)
  @overload
  defrun_sync(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AgentRunResult[ResultDataT]: ...
  @overload
  defrun_sync(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: type[RunResultDataT] | None,
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AgentRunResult[RunResultDataT]: ...
  defrun_sync(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: type[RunResultDataT] | None = None,
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AgentRunResult[Any]:
"""Synchronously run the agent with a user prompt.
    This is a convenience method that wraps [`self.run`][pydantic_ai.Agent.run] with `loop.run_until_complete(...)`.
    You therefore can't use this method inside async code or if there's an active event loop.
    Example:
    ```python
    from pydantic_ai import Agent
    agent = Agent('openai:gpt-4o')
    result_sync = agent.run_sync('What is the capital of Italy?')
    print(result_sync.data)
    #> Rome
    ```
    Args:
      user_prompt: User input to start/continue the conversation.
      result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
        result validators since result validators would expect an argument that matches the agent's result type.
      message_history: History of the conversation so far.
      model: Optional model to use for this run, required if `model` was not set when creating the agent.
      deps: Optional dependencies to use for this run.
      model_settings: Optional settings to use for this model's request.
      usage_limits: Optional limits on model request count or token usage.
      usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
      infer_name: Whether to try to infer the agent name from the call frame if it's not set.
    Returns:
      The result of the run.
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    return get_event_loop().run_until_complete(
      self.run(
        user_prompt,
        result_type=result_type,
        message_history=message_history,
        model=model,
        deps=deps,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        infer_name=False,
      )
    )
  @overload
  defrun_stream(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: None = None,
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, ResultDataT]]: ...
  @overload
  defrun_stream(
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: type[RunResultDataT],
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunResultDataT]]: ...
  @asynccontextmanager
  async defrun_stream( # noqa C901
    self,
    user_prompt: str | Sequence[_messages.UserContent],
    *,
    result_type: type[RunResultDataT] | None = None,
    message_history: list[_messages.ModelMessage] | None = None,
    model: models.Model | models.KnownModelName | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.Usage | None = None,
    infer_name: bool = True,
  ) -> AsyncIterator[result.StreamedRunResult[AgentDepsT, Any]]:
"""Run the agent with a user prompt in async mode, returning a streamed response.
    Example:
    ```python
    from pydantic_ai import Agent
    agent = Agent('openai:gpt-4o')
    async def main():
      async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_data())
        #> London
    ```
    Args:
      user_prompt: User input to start/continue the conversation.
      result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
        result validators since result validators would expect an argument that matches the agent's result type.
      message_history: History of the conversation so far.
      model: Optional model to use for this run, required if `model` was not set when creating the agent.
      deps: Optional dependencies to use for this run.
      model_settings: Optional settings to use for this model's request.
      usage_limits: Optional limits on model request count or token usage.
      usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
      infer_name: Whether to try to infer the agent name from the call frame if it's not set.
    Returns:
      The result of the run.
    """
    # TODO: We need to deprecate this now that we have the `iter` method.
    #  Before that, though, we should add an event for when we reach the final result of the stream.
    if infer_name and self.name is None:
      # f_back because `asynccontextmanager` adds one frame
      if frame := inspect.currentframe(): # pragma: no branch
        self._infer_name(frame.f_back)
    yielded = False
    async with self.iter(
      user_prompt,
      result_type=result_type,
      message_history=message_history,
      model=model,
      deps=deps,
      model_settings=model_settings,
      usage_limits=usage_limits,
      usage=usage,
      infer_name=False,
    ) as agent_run:
      first_node = agent_run.next_node # start with the first node
      assert isinstance(first_node, _agent_graph.UserPromptNode) # the first node should be a user prompt node
      node = first_node
      while True:
        if self.is_model_request_node(node):
          graph_ctx = agent_run.ctx
          async with node._stream(graph_ctx) as streamed_response: # pyright: ignore[reportPrivateUsage]
            async defstream_to_final(
              s: models.StreamedResponse,
            ) -> FinalResult[models.StreamedResponse] | None:
              result_schema = graph_ctx.deps.result_schema
              async for maybe_part_event in streamed_response:
                if isinstance(maybe_part_event, _messages.PartStartEvent):
                  new_part = maybe_part_event.part
                  if isinstance(new_part, _messages.TextPart):
                    if _agent_graph.allow_text_result(result_schema):
                      return FinalResult(s, None, None)
                  elif isinstance(new_part, _messages.ToolCallPart) and result_schema:
                    for call, _ in result_schema.find_tool([new_part]):
                      return FinalResult(s, call.tool_name, call.tool_call_id)
              return None
            final_result_details = await stream_to_final(streamed_response)
            if final_result_details is not None:
              if yielded:
                raise exceptions.AgentRunError('Agent run produced final results')
              yielded = True
              messages = graph_ctx.state.message_history.copy()
              async defon_complete() -> None:
"""Called when the stream has completed.
                The model response will have been added to messages by now
                by `StreamedRunResult._marked_completed`.

                last_message = messages[-1]
                assert isinstance(last_message, _messages.ModelResponse)
                tool_calls = [
                  part for part in last_message.parts if isinstance(part, _messages.ToolCallPart)
                ]
                parts: list[_messages.ModelRequestPart] = []
                async for _event in _agent_graph.process_function_tools(
                  tool_calls,
                  final_result_details.tool_name,
                  final_result_details.tool_call_id,
                  graph_ctx,
                  parts,
                ):
                  pass
                # TODO: Should we do something here related to the retry count?
                #  Maybe we should move the incrementing of the retry count to where we actually make a request?
                # if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
                #   ctx.state.increment_retries(ctx.deps.max_result_retries)
                if parts:
                  messages.append(_messages.ModelRequest(parts))
              yield StreamedRunResult(
                messages,
                graph_ctx.deps.new_message_index,
                graph_ctx.deps.usage_limits,
                streamed_response,
                graph_ctx.deps.result_schema,
                _agent_graph.build_run_context(graph_ctx),
                graph_ctx.deps.result_validators,
                final_result_details.tool_name,
                on_complete,
              )
              break
        next_node = await agent_run.next(node)
        if not isinstance(next_node, _agent_graph.AgentNode):
          raise exceptions.AgentRunError('Should have produced a StreamedRunResult before getting here')
        node = cast(_agent_graph.AgentNode[Any, Any], next_node)
    if not yielded:
      raise exceptions.AgentRunError('Agent run finished without producing a final result')
  @contextmanager
  defoverride(
    self,
    *,
    deps: AgentDepsT | _utils.Unset = _utils.UNSET,
    model: models.Model | models.KnownModelName | _utils.Unset = _utils.UNSET,
  ) -> Iterator[None]:
"""Context manager to temporarily override agent dependencies and model.
    This is particularly useful when testing.
    You can find an example of this [here](../testing-evals.md#overriding-model-via-pytest-fixtures).
    Args:
      deps: The dependencies to use instead of the dependencies passed to the agent run.
      model: The model to use instead of the model passed to the agent run.
    """
    if _utils.is_set(deps):
      override_deps_before = self._override_deps
      self._override_deps = _utils.Some(deps)
    else:
      override_deps_before = _utils.UNSET
    # noinspection PyTypeChecker
    if _utils.is_set(model):
      override_model_before = self._override_model
      # noinspection PyTypeChecker
      self._override_model = _utils.Some(models.infer_model(model)) # pyright: ignore[reportArgumentType]
    else:
      override_model_before = _utils.UNSET
    try:
      yield
    finally:
      if _utils.is_set(override_deps_before):
        self._override_deps = override_deps_before
      if _utils.is_set(override_model_before):
        self._override_model = override_model_before
  @overload
  defsystem_prompt(
    self, func: Callable[[RunContext[AgentDepsT]], str], /
  ) -> Callable[[RunContext[AgentDepsT]], str]: ...
  @overload
  defsystem_prompt(
    self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str]], /
  ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]: ...
  @overload
  defsystem_prompt(self, func: Callable[[], str], /) -> Callable[[], str]: ...
  @overload
  defsystem_prompt(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...
  @overload
  defsystem_prompt(
    self, /, *, dynamic: bool = False
  ) -> Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]: ...
  defsystem_prompt(
    self,
    func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
    /,
    *,
    dynamic: bool = False,
  ) -> (
    Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
    | _system_prompt.SystemPromptFunc[AgentDepsT]
  ):
"""Decorator to register a system prompt function.
    Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
    Can decorate a sync or async functions.
    The decorator can be used either bare (`agent.system_prompt`) or as a function call
    (`agent.system_prompt(...)`), see the examples below.
    Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
    the type of the function, see `tests/typed_agent.py` for tests.
    Args:
      func: The function to decorate
      dynamic: If True, the system prompt will be reevaluated even when `messages_history` is provided,
        see [`SystemPromptPart.dynamic_ref`][pydantic_ai.messages.SystemPromptPart.dynamic_ref]
    Example:
    ```python
    from pydantic_ai import Agent, RunContext
    agent = Agent('test', deps_type=str)
    @agent.system_prompt
    def simple_system_prompt() -> str:
      return 'foobar'
    @agent.system_prompt(dynamic=True)
    async def async_system_prompt(ctx: RunContext[str]) -> str:
      return f'{ctx.deps} is the best'
    ```
    """
    if func is None:
      defdecorator(
        func_: _system_prompt.SystemPromptFunc[AgentDepsT],
      ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
        runner = _system_prompt.SystemPromptRunner[AgentDepsT](func_, dynamic=dynamic)
        self._system_prompt_functions.append(runner)
        if dynamic:
          self._system_prompt_dynamic_functions[func_.__qualname__] = runner
        return func_
      return decorator
    else:
      assert not dynamic, "dynamic can't be True in this case"
      self._system_prompt_functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](func, dynamic=dynamic))
      return func
  @overload
  defresult_validator(
    self, func: Callable[[RunContext[AgentDepsT], ResultDataT], ResultDataT], /
  ) -> Callable[[RunContext[AgentDepsT], ResultDataT], ResultDataT]: ...
  @overload
  defresult_validator(
    self, func: Callable[[RunContext[AgentDepsT], ResultDataT], Awaitable[ResultDataT]], /
  ) -> Callable[[RunContext[AgentDepsT], ResultDataT], Awaitable[ResultDataT]]: ...
  @overload
  defresult_validator(
    self, func: Callable[[ResultDataT], ResultDataT], /
  ) -> Callable[[ResultDataT], ResultDataT]: ...
  @overload
  defresult_validator(
    self, func: Callable[[ResultDataT], Awaitable[ResultDataT]], /
  ) -> Callable[[ResultDataT], Awaitable[ResultDataT]]: ...
  defresult_validator(
    self, func: _result.ResultValidatorFunc[AgentDepsT, ResultDataT], /
  ) -> _result.ResultValidatorFunc[AgentDepsT, ResultDataT]:
"""Decorator to register a result validator function.
    Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
    Can decorate a sync or async functions.
    Overloads for every possible signature of `result_validator` are included so the decorator doesn't obscure
    the type of the function, see `tests/typed_agent.py` for tests.
    Example:
    ```python
    from pydantic_ai import Agent, ModelRetry, RunContext
    agent = Agent('test', deps_type=str)
    @agent.result_validator
    def result_validator_simple(data: str) -> str:
      if 'wrong' in data:
        raise ModelRetry('wrong response')
      return data
    @agent.result_validator
    async def result_validator_deps(ctx: RunContext[str], data: str) -> str:
      if ctx.deps in data:
        raise ModelRetry('wrong response')
      return data
    result = agent.run_sync('foobar', deps='spam')
    print(result.data)
    #> success (no tool calls)
    ```
    """
    self._result_validators.append(_result.ResultValidator[AgentDepsT, Any](func))
    return func
  @overload
  deftool(self, func: ToolFuncContext[AgentDepsT, ToolParams], /) -> ToolFuncContext[AgentDepsT, ToolParams]: ...
  @overload
  deftool(
    self,
    /,
    *,
    name: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
  ) -> Callable[[ToolFuncContext[AgentDepsT, ToolParams]], ToolFuncContext[AgentDepsT, ToolParams]]: ...
  deftool(
    self,
    func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
    /,
    *,
    name: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
  ) -> Any:
"""Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
    Can decorate a sync or async functions.
    The docstring is inspected to extract both the tool description and description of each parameter,
    [learn more](../tools.md#function-tools-and-schema).
    We can't add overloads for every possible signature of tool, since the return type is a recursive union
    so the signature of functions decorated with `@agent.tool` is obscured.
    Example:
    ```python
    from pydantic_ai import Agent, RunContext
    agent = Agent('test', deps_type=int)
    @agent.tool
    def foobar(ctx: RunContext[int], x: int) -> int:
      return ctx.deps + x
    @agent.tool(retries=2)
    async def spam(ctx: RunContext[str], y: float) -> float:
      return ctx.deps + y
    result = agent.run_sync('foobar', deps=1)
    print(result.data)
    #> {"foobar":1,"spam":1.0}
    ```
    Args:
      func: The tool function to register.
      name: The name of the tool, defaults to the function name.
      retries: The number of retries to allow for this tool, defaults to the agent's default retries,
        which defaults to 1.
      prepare: custom method to prepare the tool definition for each step, return `None` to omit this
        tool from a given step. This is useful if you want to customise a tool at call time,
        or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
      docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
        Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
      require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
      schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
    """
    if func is None:
      deftool_decorator(
        func_: ToolFuncContext[AgentDepsT, ToolParams],
      ) -> ToolFuncContext[AgentDepsT, ToolParams]:
        # noinspection PyTypeChecker
        self._register_function(
          func_,
          True,
          name,
          retries,
          prepare,
          docstring_format,
          require_parameter_descriptions,
          schema_generator,
        )
        return func_
      return tool_decorator
    else:
      # noinspection PyTypeChecker
      self._register_function(
        func, True, name, retries, prepare, docstring_format, require_parameter_descriptions, schema_generator
      )
      return func
  @overload
  deftool_plain(self, func: ToolFuncPlain[ToolParams], /) -> ToolFuncPlain[ToolParams]: ...
  @overload
  deftool_plain(
    self,
    /,
    *,
    name: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
  ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...
  deftool_plain(
    self,
    func: ToolFuncPlain[ToolParams] | None = None,
    /,
    *,
    name: str | None = None,
    retries: int | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
  ) -> Any:
"""Decorator to register a tool function which DOES NOT take `RunContext` as an argument.
    Can decorate a sync or async functions.
    The docstring is inspected to extract both the tool description and description of each parameter,
    [learn more](../tools.md#function-tools-and-schema).
    We can't add overloads for every possible signature of tool, since the return type is a recursive union
    so the signature of functions decorated with `@agent.tool` is obscured.
    Example:
    ```python
    from pydantic_ai import Agent, RunContext
    agent = Agent('test')
    @agent.tool
    def foobar(ctx: RunContext[int]) -> int:
      return 123
    @agent.tool(retries=2)
    async def spam(ctx: RunContext[str]) -> float:
      return 3.14
    result = agent.run_sync('foobar', deps=1)
    print(result.data)
    #> {"foobar":123,"spam":3.14}
    ```
    Args:
      func: The tool function to register.
      name: The name of the tool, defaults to the function name.
      retries: The number of retries to allow for this tool, defaults to the agent's default retries,
        which defaults to 1.
      prepare: custom method to prepare the tool definition for each step, return `None` to omit this
        tool from a given step. This is useful if you want to customise a tool at call time,
        or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
      docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
        Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
      require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
      schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
    """
    if func is None:
      deftool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
        # noinspection PyTypeChecker
        self._register_function(
          func_,
          False,
          name,
          retries,
          prepare,
          docstring_format,
          require_parameter_descriptions,
          schema_generator,
        )
        return func_
      return tool_decorator
    else:
      self._register_function(
        func, False, name, retries, prepare, docstring_format, require_parameter_descriptions, schema_generator
      )
      return func
  def_register_function(
    self,
    func: ToolFuncEither[AgentDepsT, ToolParams],
    takes_ctx: bool,
    name: str | None,
    retries: int | None,
    prepare: ToolPrepareFunc[AgentDepsT] | None,
    docstring_format: DocstringFormat,
    require_parameter_descriptions: bool,
    schema_generator: type[GenerateJsonSchema],
  ) -> None:
"""Private utility to register a function as a tool."""
    retries_ = retries if retries is not None else self._default_retries
    tool = Tool[AgentDepsT](
      func,
      takes_ctx=takes_ctx,
      name=name,
      max_retries=retries_,
      prepare=prepare,
      docstring_format=docstring_format,
      require_parameter_descriptions=require_parameter_descriptions,
      schema_generator=schema_generator,
    )
    self._register_tool(tool)
  def_register_tool(self, tool: Tool[AgentDepsT]) -> None:
"""Private utility to register a tool instance."""
    if tool.max_retries is None:
      # noinspection PyTypeChecker
      tool = dataclasses.replace(tool, max_retries=self._default_retries)
    if tool.name in self._function_tools:
      raise exceptions.UserError(f'Tool name conflicts with existing tool: {tool.name!r}')
    if self._result_schema and tool.name in self._result_schema.tools:
      raise exceptions.UserError(f'Tool name conflicts with result schema name: {tool.name!r}')
    self._function_tools[tool.name] = tool
  def_get_model(self, model: models.Model | models.KnownModelName | None) -> models.Model:
"""Create a model configured for this agent.
    Args:
      model: model to use for this run, required if `model` was not set when creating the agent.
    Returns:
      The model used
    """
    model_: models.Model
    if some_model := self._override_model:
      # we don't want `override()` to cover up errors from the model not being defined, hence this check
      if model is None and self.model is None:
        raise exceptions.UserError(
          '`model` must be set either when creating the agent or when calling it. '
          '(Even when `override(model=...)` is customizing the model that will actually be called)'
        )
      model_ = some_model.value
    elif model is not None:
      model_ = models.infer_model(model)
    elif self.model is not None:
      # noinspection PyTypeChecker
      model_ = self.model = models.infer_model(self.model)
    else:
      raise exceptions.UserError('`model` must be set either when creating the agent or when calling it.')
    instrument = self.instrument
    if instrument is None:
      instrument = self._instrument_default
    if instrument and not isinstance(model_, InstrumentedModel):
      if instrument is True:
        instrument = InstrumentationSettings()
      model_ = InstrumentedModel(model_, instrument)
    return model_
  def_get_deps(self: Agent[T, ResultDataT], deps: T) -> T:
"""Get deps for a run.
    If we've overridden deps via `_override_deps`, use that, otherwise use the deps passed to the call.
    We could do runtime type checking of deps against `self._deps_type`, but that's a slippery slope.
    """
    if some_deps := self._override_deps:
      return some_deps.value
    else:
      return deps
  def_infer_name(self, function_frame: FrameType | None) -> None:
"""Infer the agent name from the call frame.
    Usage should be `self._infer_name(inspect.currentframe())`.
    """
    assert self.name is None, 'Name already set'
    if function_frame is not None: # pragma: no branch
      if parent_frame := function_frame.f_back: # pragma: no branch
        for name, item in parent_frame.f_locals.items():
          if item is self:
            self.name = name
            return
        if parent_frame.f_locals != parent_frame.f_globals:
          # if we couldn't find the agent in locals and globals are a different dict, try globals
          for name, item in parent_frame.f_globals.items():
            if item is self:
              self.name = name
              return
  @property
  @deprecated(
    'The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.', category=None
  )
  deflast_run_messages(self) -> list[_messages.ModelMessage]:
    raise AttributeError('The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.')
  def_build_graph(
    self, result_type: type[RunResultDataT] | None
  ) -> Graph[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[Any]]:
    return _agent_graph.build_agent_graph(self.name, self._deps_type, result_type or self.result_type)
  def_prepare_result_schema(
    self, result_type: type[RunResultDataT] | None
  ) -> _result.ResultSchema[RunResultDataT] | None:
    if result_type is not None:
      if self._result_validators:
        raise exceptions.UserError('Cannot set a custom run `result_type` when the agent has result validators')
      return _result.ResultSchema[result_type].build(
        result_type, self._result_tool_name, self._result_tool_description
      )
    else:
      return self._result_schema # pyright: ignore[reportReturnType]
  @staticmethod
  defis_model_request_node(
    node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
  ) -> TypeGuard[_agent_graph.ModelRequestNode[T, S]]:
"""Check if the node is a `ModelRequestNode`, narrowing the type if it is.
    This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
    """
    return isinstance(node, _agent_graph.ModelRequestNode)
  @staticmethod
  defis_call_tools_node(
    node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
  ) -> TypeGuard[_agent_graph.CallToolsNode[T, S]]:
"""Check if the node is a `CallToolsNode`, narrowing the type if it is.
    This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
    """
    return isinstance(node, _agent_graph.CallToolsNode)
  @staticmethod
  defis_user_prompt_node(
    node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
  ) -> TypeGuard[_agent_graph.UserPromptNode[T, S]]:
"""Check if the node is a `UserPromptNode`, narrowing the type if it is.
    This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
    """
    return isinstance(node, _agent_graph.UserPromptNode)
  @staticmethod
  defis_end_node(
    node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
  ) -> TypeGuard[End[result.FinalResult[S]]]:
"""Check if the node is a `End`, narrowing the type if it is.
    This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
    """
    return isinstance(node, End)
  @asynccontextmanager
  async defrun_mcp_servers(self) -> AsyncIterator[None]:
"""Run [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] so they can be used by the agent.
    Returns: a context manager to start and shutdown the servers.
    """
    exit_stack = AsyncExitStack()
    try:
      for mcp_server in self._mcp_servers:
        await exit_stack.enter_async_context(mcp_server)
      yield
    finally:
      await exit_stack.aclose()

```
  
---|---  
####  model `instance-attribute`
```
model:  |  | None

```

The default model configured for this agent.
####  __init__
```
__init__(
  model:  |  | None = None,
  *,
  result_type: [] = ,
  system_prompt:  | [] = (),
  deps_type: [] = NoneType,
  name:  | None = None,
  model_settings:  | None = None,
  retries:  = 1,
  result_tool_name:  = "final_result",
  result_tool_description:  | None = None,
  result_retries:  | None = None,
  tools: [
    [] | [, ...]
  ] = (),
  mcp_servers: [] = (),
  defer_model_check:  = False,
  end_strategy:  = "early",
  instrument:  |  | None = None
)

```

Create an agent.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`model` |  `Model[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model "pydantic_ai.models.Model") | KnownModelName[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.KnownModelName "pydantic_ai.models.KnownModelName") | None` |  The default model to use for this agent, if not provide, you must provide the model when calling it. |  `None`  
`result_type` |  |  The type of the result data, used to validate the result data, defaults to `str`.  
`system_prompt` |  |  Static system prompts to use for this agent, you can also register system prompts via a function with [`system_prompt`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.system_prompt).  
`deps_type` |  |  The type used for dependency injection, this parameter exists solely to allow you to fully parameterize the agent, and therefore get the best out of static type checking. If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright or add a type hint `: Agent[None, <return type>]`. |  `NoneType`  
`name` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame when the agent is first run. |  `None`  
`model_settings` |  `ModelSettings[](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings "pydantic_ai.settings.ModelSettings") | None` |  Optional model request settings to use for this agent's runs, by default. |  `None`  
`retries` |  The default number of retries to allow before raising an error.  
`result_tool_name` |  The name of the tool to use for the final result. |  `'final_result'`  
`result_tool_description` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The description of the final result tool. |  `None`  
`result_retries` |  `int[](https://docs.python.org/3/library/functions.html#int) | None` |  The maximum number of retries to allow for result validation, defaults to `retries`. |  `None`  
`tools` |  `Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[Tool[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.Tool "pydantic_ai.tools.Tool")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT")] | ToolFuncEither[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolFuncEither "pydantic_ai.tools.ToolFuncEither")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), ...]]` |  Tools to register with the agent, you can also register tools via the decorators [`@agent.tool`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool) and [`@agent.tool_plain`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool_plain).  
`mcp_servers` |  |  MCP servers to register with the agent. You should register a [`MCPServer`](https://ai.pydantic.dev/api/mcp/#pydantic_ai.mcp.MCPServer) for each server you want the agent to connect to.  
`defer_model_check` |  by default, if you provide a [named](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.KnownModelName) model, it's evaluated to create a [`Model`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model) instance immediately, which checks for the necessary environment variables. Set this to `false` to defer the evaluation until the first run. Useful if you want to [override the model](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.override) for testing. |  `False`  
`end_strategy` |  |  Strategy for handling tool calls that are requested alongside a final result. See [`EndStrategy`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.EndStrategy) for more information. |  `'early'`  
`instrument` |  `InstrumentationSettings[](https://ai.pydantic.dev/api/models/instrumented/#pydantic_ai.models.instrumented.InstrumentationSettings "pydantic_ai.models.instrumented.InstrumentationSettings") | bool[](https://docs.python.org/3/library/functions.html#bool) | None` |  Set to True to automatically instrument with OpenTelemetry, which will use Logfire if it's configured. Set to an instance of [`InstrumentationSettings`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.InstrumentationSettings) to customize. If this isn't set, then the last value set by [`Agent.instrument_all()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.instrument_all) will be used, which defaults to False. See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
def__init__(
  self,
  model: models.Model | models.KnownModelName | None = None,
  *,
  result_type: type[ResultDataT] = str,
  system_prompt: str | Sequence[str] = (),
  deps_type: type[AgentDepsT] = NoneType,
  name: str | None = None,
  model_settings: ModelSettings | None = None,
  retries: int = 1,
  result_tool_name: str = 'final_result',
  result_tool_description: str | None = None,
  result_retries: int | None = None,
  tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
  mcp_servers: Sequence[MCPServer] = (),
  defer_model_check: bool = False,
  end_strategy: EndStrategy = 'early',
  instrument: InstrumentationSettings | bool | None = None,
):
"""Create an agent.
  Args:
    model: The default model to use for this agent, if not provide,
      you must provide the model when calling it.
    result_type: The type of the result data, used to validate the result data, defaults to `str`.
    system_prompt: Static system prompts to use for this agent, you can also register system
      prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
    deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
      parameterize the agent, and therefore get the best out of static type checking.
      If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
      or add a type hint `: Agent[None, <return type>]`.
    name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
      when the agent is first run.
    model_settings: Optional model request settings to use for this agent's runs, by default.
    retries: The default number of retries to allow before raising an error.
    result_tool_name: The name of the tool to use for the final result.
    result_tool_description: The description of the final result tool.
    result_retries: The maximum number of retries to allow for result validation, defaults to `retries`.
    tools: Tools to register with the agent, you can also register tools via the decorators
      [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
    mcp_servers: MCP servers to register with the agent. You should register a [`MCPServer`][pydantic_ai.mcp.MCPServer]
      for each server you want the agent to connect to.
    defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
      it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
      which checks for the necessary environment variables. Set this to `false`
      to defer the evaluation until the first run. Useful if you want to
      [override the model][pydantic_ai.Agent.override] for testing.
    end_strategy: Strategy for handling tool calls that are requested alongside a final result.
      See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
    instrument: Set to True to automatically instrument with OpenTelemetry,
      which will use Logfire if it's configured.
      Set to an instance of [`InstrumentationSettings`][pydantic_ai.agent.InstrumentationSettings] to customize.
      If this isn't set, then the last value set by
      [`Agent.instrument_all()`][pydantic_ai.Agent.instrument_all]
      will be used, which defaults to False.
      See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
  """
  if model is None or defer_model_check:
    self.model = model
  else:
    self.model = models.infer_model(model)
  self.end_strategy = end_strategy
  self.name = name
  self.model_settings = model_settings
  self.result_type = result_type
  self.instrument = instrument
  self._deps_type = deps_type
  self._result_tool_name = result_tool_name
  self._result_tool_description = result_tool_description
  self._result_schema: _result.ResultSchema[ResultDataT] | None = _result.ResultSchema[result_type].build(
    result_type, result_tool_name, result_tool_description
  )
  self._result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]] = []
  self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
  self._system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = []
  self._system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = {}
  self._function_tools: dict[str, Tool[AgentDepsT]] = {}
  self._default_retries = retries
  self._max_result_retries = result_retries if result_retries is not None else retries
  self._mcp_servers = mcp_servers
  for tool in tools:
    if isinstance(tool, Tool):
      self._register_tool(tool)
    else:
      self._register_tool(Tool(tool))

```
  
---|---  
####  end_strategy `instance-attribute`
```
end_strategy:  = end_strategy

```

Strategy for handling tool calls when a final result is found.
####  name `instance-attribute`
```
name:  | None = 

```

The name of the agent, used for logging.
If `None`, we try to infer the agent name from the call frame when the agent is first run.
####  model_settings `instance-attribute`
```
model_settings:  | None = model_settings

```

Optional model request settings to use for this agents's runs, by default.
Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will be merged with this value, with the runtime argument taking priority.
####  result_type `class-attribute` `instance-attribute`
```
result_type: [] = result_type

```

The type of the result data, used to validate the result data, defaults to `str`.
####  instrument `instance-attribute`
```
instrument:  |  | None = (
  instrument
)

```

Options to automatically instrument with OpenTelemetry.
####  instrument_all `staticmethod`
```
instrument_all(
  instrument:  |  = True,
) -> None

```

Set the instrumentation options for all agents where `instrument` is not set.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@staticmethod
definstrument_all(instrument: InstrumentationSettings | bool = True) -> None:
"""Set the instrumentation options for all agents where `instrument` is not set."""
  Agent._instrument_default = instrument

```
  
---|---  
####  run `async`
```
run(
  user_prompt:  | [UserContent],
  *,
  result_type: None = None,
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> []

```

```
run(
  user_prompt:  | [UserContent],
  *,
  result_type: [],
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> []

```

```
run(
  user_prompt:  | [UserContent],
  *,
  result_type: [] | None = None,
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> []

```

Run the agent with a user prompt in async mode.
This method builds an internal agent graph (using system prompts, tools and result schemas) and then runs the graph to completion. The result of the run is returned.
Example: 
```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
async defmain():
  agent_run = await agent.run('What is the capital of France?')
  print(agent_run.data)
  #> Paris

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`user_prompt` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[UserContent]` |  User input to start/continue the conversation. |  _required_  
`result_type` |  `type[](https://docs.python.org/3/library/functions.html#type)[RunResultDataT[](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.RunResultDataT "pydantic_ai.agent.RunResultDataT")] | None` |  Custom result type to use for this run, `result_type` may only be used if the agent has no result validators since result validators would expect an argument that matches the agent's result type. |  `None`  
`message_history` |  `list[](https://docs.python.org/3/library/stdtypes.html#list)[ModelMessage[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelMessage "pydantic_ai.messages.ModelMessage")] | None` |  History of the conversation so far. |  `None`  
`model` |  `Model[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model "pydantic_ai.models.Model") | KnownModelName[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.KnownModelName "pydantic_ai.models.KnownModelName") | None` |  Optional model to use for this run, required if `model` was not set when creating the agent. |  `None`  
`deps` |  |  Optional dependencies to use for this run. |  `None`  
`model_settings` |  `ModelSettings[](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings "pydantic_ai.settings.ModelSettings") | None` |  Optional settings to use for this model's request. |  `None`  
`usage_limits` |  `UsageLimits[](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits "pydantic_ai.usage.UsageLimits") | None` |  Optional limits on model request count or token usage. |  `None`  
`usage` |  `Usage[](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.Usage "pydantic_ai.usage.Usage") | None` |  Optional usage to start with, useful for resuming a conversation or agents used in tools. |  `None`  
`infer_name` |  Whether to try to infer the agent name from the call frame if it's not set. |  `True`  
Returns:
Type | Description  
---|---  
|  The result of the run.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
async defrun(
  self,
  user_prompt: str | Sequence[_messages.UserContent],
  *,
  result_type: type[RunResultDataT] | None = None,
  message_history: list[_messages.ModelMessage] | None = None,
  model: models.Model | models.KnownModelName | None = None,
  deps: AgentDepsT = None,
  model_settings: ModelSettings | None = None,
  usage_limits: _usage.UsageLimits | None = None,
  usage: _usage.Usage | None = None,
  infer_name: bool = True,
) -> AgentRunResult[Any]:
"""Run the agent with a user prompt in async mode.
  This method builds an internal agent graph (using system prompts, tools and result schemas) and then
  runs the graph to completion. The result of the run is returned.
  Example:
  ```python
  from pydantic_ai import Agent
  agent = Agent('openai:gpt-4o')
  async def main():
    agent_run = await agent.run('What is the capital of France?')
    print(agent_run.data)
    #> Paris
  ```
  Args:
    user_prompt: User input to start/continue the conversation.
    result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
      result validators since result validators would expect an argument that matches the agent's result type.
    message_history: History of the conversation so far.
    model: Optional model to use for this run, required if `model` was not set when creating the agent.
    deps: Optional dependencies to use for this run.
    model_settings: Optional settings to use for this model's request.
    usage_limits: Optional limits on model request count or token usage.
    usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
    infer_name: Whether to try to infer the agent name from the call frame if it's not set.
  Returns:
    The result of the run.
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  async with self.iter(
    user_prompt=user_prompt,
    result_type=result_type,
    message_history=message_history,
    model=model,
    deps=deps,
    model_settings=model_settings,
    usage_limits=usage_limits,
    usage=usage,
  ) as agent_run:
    async for _ in agent_run:
      pass
  assert (final_result := agent_run.result) is not None, 'The graph run did not finish properly'
  return final_result

```
  
---|---  
####  iter `async`
```
iter(
  user_prompt:  | [UserContent],
  *,
  result_type: [] | None = None,
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> [[, ]]

```

A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.
This method builds an internal agent graph (using system prompts, tools and result schemas) and then returns an `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the stream of events coming from the execution of tools.
The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics, and the final result of the run once it has completed.
For more details, see the documentation of `AgentRun`.
Example: 
```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
async defmain():
  nodes = []
  async with agent.iter('What is the capital of France?') as agent_run:
    async for node in agent_run:
      nodes.append(node)
  print(nodes)
'''

    ModelRequestNode(
      request=ModelRequest(
        parts=[
          UserPromptPart(
            content='What is the capital of France?',
            timestamp=datetime.datetime(...),
            part_kind='user-prompt',


        kind='request',

    ),
    CallToolsNode(
      model_response=ModelResponse(
        parts=[TextPart(content='Paris', part_kind='text')],
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
        kind='response',

    ),
    End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

  '''
  print(agent_run.result.data)
  #> Paris

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`user_prompt` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[UserContent]` |  User input to start/continue the conversation. |  _required_  
`result_type` |  `type[](https://docs.python.org/3/library/functions.html#type)[RunResultDataT[](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.RunResultDataT "pydantic_ai.agent.RunResultDataT")] | None` |  Custom result type to use for this run, `result_type` may only be used if the agent has no result validators since result validators would expect an argument that matches the agent's result type. |  `None`  
`message_history` |  `list[](https://docs.python.org/3/library/stdtypes.html#list)[ModelMessage[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelMessage "pydantic_ai.messages.ModelMessage")] | None` |  History of the conversation so far. |  `None`  
`model` |  `Model[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model "pydantic_ai.models.Model") | KnownModelName[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.KnownModelName "pydantic_ai.models.KnownModelName") | None` |  Optional model to use for this run, required if `model` was not set when creating the agent. |  `None`  
`deps` |  |  Optional dependencies to use for this run. |  `None`  
`model_settings` |  `ModelSettings[](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings "pydantic_ai.settings.ModelSettings") | None` |  Optional settings to use for this model's request. |  `None`  
`usage_limits` |  `UsageLimits[](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits "pydantic_ai.usage.UsageLimits") | None` |  Optional limits on model request count or token usage. |  `None`  
`usage` |  `Usage[](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.Usage "pydantic_ai.usage.Usage") | None` |  Optional usage to start with, useful for resuming a conversation or agents used in tools. |  `None`  
`infer_name` |  Whether to try to infer the agent name from the call frame if it's not set. |  `True`  
Returns:
Type | Description  
---|---  
`AsyncIterator[](https://docs.python.org/3/library/collections.abc.html#collections.abc.AsyncIterator "collections.abc.AsyncIterator")[AgentRun[](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRun "pydantic_ai.agent.AgentRun")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), Any[](https://docs.python.org/3/library/typing.html#typing.Any "typing.Any")]]` |  The result of the run.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@asynccontextmanager
async defiter(
  self,
  user_prompt: str | Sequence[_messages.UserContent],
  *,
  result_type: type[RunResultDataT] | None = None,
  message_history: list[_messages.ModelMessage] | None = None,
  model: models.Model | models.KnownModelName | None = None,
  deps: AgentDepsT = None,
  model_settings: ModelSettings | None = None,
  usage_limits: _usage.UsageLimits | None = None,
  usage: _usage.Usage | None = None,
  infer_name: bool = True,
) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
"""A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.
  This method builds an internal agent graph (using system prompts, tools and result schemas) and then returns an
  `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
  executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
  stream of events coming from the execution of tools.
  The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
  and the final result of the run once it has completed.
  For more details, see the documentation of `AgentRun`.
  Example:
  ```python
  from pydantic_ai import Agent
  agent = Agent('openai:gpt-4o')
  async def main():
    nodes = []
    async with agent.iter('What is the capital of France?') as agent_run:
      async for node in agent_run:
        nodes.append(node)
    print(nodes)
    '''

      ModelRequestNode(
        request=ModelRequest(
          parts=[
            UserPromptPart(
              content='What is the capital of France?',
              timestamp=datetime.datetime(...),
              part_kind='user-prompt',


          kind='request',


      CallToolsNode(
        model_response=ModelResponse(
          parts=[TextPart(content='Paris', part_kind='text')],
          model_name='gpt-4o',
          timestamp=datetime.datetime(...),
          kind='response',


      End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

    '''
    print(agent_run.result.data)
    #> Paris
  ```
  Args:
    user_prompt: User input to start/continue the conversation.
    result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
      result validators since result validators would expect an argument that matches the agent's result type.
    message_history: History of the conversation so far.
    model: Optional model to use for this run, required if `model` was not set when creating the agent.
    deps: Optional dependencies to use for this run.
    model_settings: Optional settings to use for this model's request.
    usage_limits: Optional limits on model request count or token usage.
    usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
    infer_name: Whether to try to infer the agent name from the call frame if it's not set.
  Returns:
    The result of the run.
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  model_used = self._get_model(model)
  del model
  deps = self._get_deps(deps)
  new_message_index = len(message_history) if message_history else 0
  result_schema: _result.ResultSchema[RunResultDataT] | None = self._prepare_result_schema(result_type)
  # Build the graph
  graph = self._build_graph(result_type)
  # Build the initial state
  state = _agent_graph.GraphAgentState(
    message_history=message_history[:] if message_history else [],
    usage=usage or _usage.Usage(),
    retries=0,
    run_step=0,
  )
  # We consider it a user error if a user tries to restrict the result type while having a result validator that
  # may change the result type from the restricted type to something else. Therefore, we consider the following
  # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
  result_validators = cast(list[_result.ResultValidator[AgentDepsT, RunResultDataT]], self._result_validators)
  # TODO: Instead of this, copy the function tools to ensure they don't share current_retry state between agent
  # runs. Requires some changes to `Tool` to make them copyable though.
  for v in self._function_tools.values():
    v.current_retry = 0
  model_settings = merge_model_settings(self.model_settings, model_settings)
  usage_limits = usage_limits or _usage.UsageLimits()
  if isinstance(model_used, InstrumentedModel):
    tracer = model_used.settings.tracer
  else:
    tracer = NoOpTracer()
  agent_name = self.name or 'agent'
  run_span = tracer.start_span(
    'agent run',
    attributes={
      'model_name': model_used.model_name if model_used else 'no-model',
      'agent_name': agent_name,
      'logfire.msg': f'{agent_name} run',
    },
  )
  graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, RunResultDataT](
    user_deps=deps,
    prompt=user_prompt,
    new_message_index=new_message_index,
    model=model_used,
    model_settings=model_settings,
    usage_limits=usage_limits,
    max_result_retries=self._max_result_retries,
    end_strategy=self.end_strategy,
    result_schema=result_schema,
    result_tools=self._result_schema.tool_defs() if self._result_schema else [],
    result_validators=result_validators,
    function_tools=self._function_tools,
    mcp_servers=self._mcp_servers,
    run_span=run_span,
    tracer=tracer,
  )
  start_node = _agent_graph.UserPromptNode[AgentDepsT](
    user_prompt=user_prompt,
    system_prompts=self._system_prompts,
    system_prompt_functions=self._system_prompt_functions,
    system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
  )
  async with graph.iter(
    start_node,
    state=state,
    deps=graph_deps,
    span=use_span(run_span, end_on_exit=True),
    infer_name=False,
  ) as graph_run:
    yield AgentRun(graph_run)

```
  
---|---  
####  run_sync
```
run_sync(
  user_prompt:  | [UserContent],
  *,
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> []

```

```
run_sync(
  user_prompt:  | [UserContent],
  *,
  result_type: [] | None,
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> []

```

```
run_sync(
  user_prompt:  | [UserContent],
  *,
  result_type: [] | None = None,
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> []

```

Synchronously run the agent with a user prompt.
This is a convenience method that wraps [`self.run`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run) with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.
Example: 
```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.data)
#> Rome

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`user_prompt` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[UserContent]` |  User input to start/continue the conversation. |  _required_  
`result_type` |  `type[](https://docs.python.org/3/library/functions.html#type)[RunResultDataT[](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.RunResultDataT "pydantic_ai.agent.RunResultDataT")] | None` |  Custom result type to use for this run, `result_type` may only be used if the agent has no result validators since result validators would expect an argument that matches the agent's result type. |  `None`  
`message_history` |  `list[](https://docs.python.org/3/library/stdtypes.html#list)[ModelMessage[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelMessage "pydantic_ai.messages.ModelMessage")] | None` |  History of the conversation so far. |  `None`  
`model` |  `Model[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model "pydantic_ai.models.Model") | KnownModelName[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.KnownModelName "pydantic_ai.models.KnownModelName") | None` |  Optional model to use for this run, required if `model` was not set when creating the agent. |  `None`  
`deps` |  |  Optional dependencies to use for this run. |  `None`  
`model_settings` |  `ModelSettings[](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings "pydantic_ai.settings.ModelSettings") | None` |  Optional settings to use for this model's request. |  `None`  
`usage_limits` |  `UsageLimits[](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits "pydantic_ai.usage.UsageLimits") | None` |  Optional limits on model request count or token usage. |  `None`  
`usage` |  `Usage[](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.Usage "pydantic_ai.usage.Usage") | None` |  Optional usage to start with, useful for resuming a conversation or agents used in tools. |  `None`  
`infer_name` |  Whether to try to infer the agent name from the call frame if it's not set. |  `True`  
Returns:
Type | Description  
---|---  
|  The result of the run.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defrun_sync(
  self,
  user_prompt: str | Sequence[_messages.UserContent],
  *,
  result_type: type[RunResultDataT] | None = None,
  message_history: list[_messages.ModelMessage] | None = None,
  model: models.Model | models.KnownModelName | None = None,
  deps: AgentDepsT = None,
  model_settings: ModelSettings | None = None,
  usage_limits: _usage.UsageLimits | None = None,
  usage: _usage.Usage | None = None,
  infer_name: bool = True,
) -> AgentRunResult[Any]:
"""Synchronously run the agent with a user prompt.
  This is a convenience method that wraps [`self.run`][pydantic_ai.Agent.run] with `loop.run_until_complete(...)`.
  You therefore can't use this method inside async code or if there's an active event loop.
  Example:
  ```python
  from pydantic_ai import Agent
  agent = Agent('openai:gpt-4o')
  result_sync = agent.run_sync('What is the capital of Italy?')
  print(result_sync.data)
  #> Rome
  ```
  Args:
    user_prompt: User input to start/continue the conversation.
    result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
      result validators since result validators would expect an argument that matches the agent's result type.
    message_history: History of the conversation so far.
    model: Optional model to use for this run, required if `model` was not set when creating the agent.
    deps: Optional dependencies to use for this run.
    model_settings: Optional settings to use for this model's request.
    usage_limits: Optional limits on model request count or token usage.
    usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
    infer_name: Whether to try to infer the agent name from the call frame if it's not set.
  Returns:
    The result of the run.
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  return get_event_loop().run_until_complete(
    self.run(
      user_prompt,
      result_type=result_type,
      message_history=message_history,
      model=model,
      deps=deps,
      model_settings=model_settings,
      usage_limits=usage_limits,
      usage=usage,
      infer_name=False,
    )
  )

```
  
---|---  
####  run_stream `async`
```
run_stream(
  user_prompt:  | [UserContent],
  *,
  result_type: None = None,
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> [
  [, ]
]

```

```
run_stream(
  user_prompt:  | [UserContent],
  *,
  result_type: [],
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> [
  [, ]
]

```

```
run_stream(
  user_prompt:  | [UserContent],
  *,
  result_type: [] | None = None,
  message_history: [] | None = None,
  model:  |  | None = None,
  deps:  = None,
  model_settings:  | None = None,
  usage_limits:  | None = None,
  usage:  | None = None,
  infer_name:  = True
) -> [[, ]]

```

Run the agent with a user prompt in async mode, returning a streamed response.
Example: 
```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
async defmain():
  async with agent.run_stream('What is the capital of the UK?') as response:
    print(await response.get_data())
    #> London

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`user_prompt` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[UserContent]` |  User input to start/continue the conversation. |  _required_  
`result_type` |  `type[](https://docs.python.org/3/library/functions.html#type)[RunResultDataT[](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.RunResultDataT "pydantic_ai.agent.RunResultDataT")] | None` |  Custom result type to use for this run, `result_type` may only be used if the agent has no result validators since result validators would expect an argument that matches the agent's result type. |  `None`  
`message_history` |  `list[](https://docs.python.org/3/library/stdtypes.html#list)[ModelMessage[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelMessage "pydantic_ai.messages.ModelMessage")] | None` |  History of the conversation so far. |  `None`  
`model` |  `Model[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model "pydantic_ai.models.Model") | KnownModelName[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.KnownModelName "pydantic_ai.models.KnownModelName") | None` |  Optional model to use for this run, required if `model` was not set when creating the agent. |  `None`  
`deps` |  |  Optional dependencies to use for this run. |  `None`  
`model_settings` |  `ModelSettings[](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings "pydantic_ai.settings.ModelSettings") | None` |  Optional settings to use for this model's request. |  `None`  
`usage_limits` |  `UsageLimits[](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits "pydantic_ai.usage.UsageLimits") | None` |  Optional limits on model request count or token usage. |  `None`  
`usage` |  `Usage[](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.Usage "pydantic_ai.usage.Usage") | None` |  Optional usage to start with, useful for resuming a conversation or agents used in tools. |  `None`  
`infer_name` |  Whether to try to infer the agent name from the call frame if it's not set. |  `True`  
Returns:
Type | Description  
---|---  
`AsyncIterator[](https://docs.python.org/3/library/collections.abc.html#collections.abc.AsyncIterator "collections.abc.AsyncIterator")[StreamedRunResult[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult "pydantic_ai.result.StreamedRunResult")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), Any[](https://docs.python.org/3/library/typing.html#typing.Any "typing.Any")]]` |  The result of the run.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@asynccontextmanager
async defrun_stream( # noqa C901
  self,
  user_prompt: str | Sequence[_messages.UserContent],
  *,
  result_type: type[RunResultDataT] | None = None,
  message_history: list[_messages.ModelMessage] | None = None,
  model: models.Model | models.KnownModelName | None = None,
  deps: AgentDepsT = None,
  model_settings: ModelSettings | None = None,
  usage_limits: _usage.UsageLimits | None = None,
  usage: _usage.Usage | None = None,
  infer_name: bool = True,
) -> AsyncIterator[result.StreamedRunResult[AgentDepsT, Any]]:
"""Run the agent with a user prompt in async mode, returning a streamed response.
  Example:
  ```python
  from pydantic_ai import Agent
  agent = Agent('openai:gpt-4o')
  async def main():
    async with agent.run_stream('What is the capital of the UK?') as response:
      print(await response.get_data())
      #> London
  ```
  Args:
    user_prompt: User input to start/continue the conversation.
    result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
      result validators since result validators would expect an argument that matches the agent's result type.
    message_history: History of the conversation so far.
    model: Optional model to use for this run, required if `model` was not set when creating the agent.
    deps: Optional dependencies to use for this run.
    model_settings: Optional settings to use for this model's request.
    usage_limits: Optional limits on model request count or token usage.
    usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
    infer_name: Whether to try to infer the agent name from the call frame if it's not set.
  Returns:
    The result of the run.
  """
  # TODO: We need to deprecate this now that we have the `iter` method.
  #  Before that, though, we should add an event for when we reach the final result of the stream.
  if infer_name and self.name is None:
    # f_back because `asynccontextmanager` adds one frame
    if frame := inspect.currentframe(): # pragma: no branch
      self._infer_name(frame.f_back)
  yielded = False
  async with self.iter(
    user_prompt,
    result_type=result_type,
    message_history=message_history,
    model=model,
    deps=deps,
    model_settings=model_settings,
    usage_limits=usage_limits,
    usage=usage,
    infer_name=False,
  ) as agent_run:
    first_node = agent_run.next_node # start with the first node
    assert isinstance(first_node, _agent_graph.UserPromptNode) # the first node should be a user prompt node
    node = first_node
    while True:
      if self.is_model_request_node(node):
        graph_ctx = agent_run.ctx
        async with node._stream(graph_ctx) as streamed_response: # pyright: ignore[reportPrivateUsage]
          async defstream_to_final(
            s: models.StreamedResponse,
          ) -> FinalResult[models.StreamedResponse] | None:
            result_schema = graph_ctx.deps.result_schema
            async for maybe_part_event in streamed_response:
              if isinstance(maybe_part_event, _messages.PartStartEvent):
                new_part = maybe_part_event.part
                if isinstance(new_part, _messages.TextPart):
                  if _agent_graph.allow_text_result(result_schema):
                    return FinalResult(s, None, None)
                elif isinstance(new_part, _messages.ToolCallPart) and result_schema:
                  for call, _ in result_schema.find_tool([new_part]):
                    return FinalResult(s, call.tool_name, call.tool_call_id)
            return None
          final_result_details = await stream_to_final(streamed_response)
          if final_result_details is not None:
            if yielded:
              raise exceptions.AgentRunError('Agent run produced final results')
            yielded = True
            messages = graph_ctx.state.message_history.copy()
            async defon_complete() -> None:
"""Called when the stream has completed.
              The model response will have been added to messages by now
              by `StreamedRunResult._marked_completed`.

              last_message = messages[-1]
              assert isinstance(last_message, _messages.ModelResponse)
              tool_calls = [
                part for part in last_message.parts if isinstance(part, _messages.ToolCallPart)
              ]
              parts: list[_messages.ModelRequestPart] = []
              async for _event in _agent_graph.process_function_tools(
                tool_calls,
                final_result_details.tool_name,
                final_result_details.tool_call_id,
                graph_ctx,
                parts,
              ):
                pass
              # TODO: Should we do something here related to the retry count?
              #  Maybe we should move the incrementing of the retry count to where we actually make a request?
              # if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
              #   ctx.state.increment_retries(ctx.deps.max_result_retries)
              if parts:
                messages.append(_messages.ModelRequest(parts))
            yield StreamedRunResult(
              messages,
              graph_ctx.deps.new_message_index,
              graph_ctx.deps.usage_limits,
              streamed_response,
              graph_ctx.deps.result_schema,
              _agent_graph.build_run_context(graph_ctx),
              graph_ctx.deps.result_validators,
              final_result_details.tool_name,
              on_complete,
            )
            break
      next_node = await agent_run.next(node)
      if not isinstance(next_node, _agent_graph.AgentNode):
        raise exceptions.AgentRunError('Should have produced a StreamedRunResult before getting here')
      node = cast(_agent_graph.AgentNode[Any, Any], next_node)
  if not yielded:
    raise exceptions.AgentRunError('Agent run finished without producing a final result')

```
  
---|---  
####  override
```
override(
  *,
  deps:  |  = ,
  model:  |  |  = 
) -> [None]

```

Context manager to temporarily override agent dependencies and model.
This is particularly useful when testing. You can find an example of this [here](https://ai.pydantic.dev/testing-evals/#overriding-model-via-pytest-fixtures).
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`deps` |  `AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT") | Unset` |  The dependencies to use instead of the dependencies passed to the agent run. |  `UNSET`  
`model` |  `Model[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model "pydantic_ai.models.Model") | KnownModelName[](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.KnownModelName "pydantic_ai.models.KnownModelName") | Unset` |  The model to use instead of the model passed to the agent run. |  `UNSET`  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@contextmanager
defoverride(
  self,
  *,
  deps: AgentDepsT | _utils.Unset = _utils.UNSET,
  model: models.Model | models.KnownModelName | _utils.Unset = _utils.UNSET,
) -> Iterator[None]:
"""Context manager to temporarily override agent dependencies and model.
  This is particularly useful when testing.
  You can find an example of this [here](../testing-evals.md#overriding-model-via-pytest-fixtures).
  Args:
    deps: The dependencies to use instead of the dependencies passed to the agent run.
    model: The model to use instead of the model passed to the agent run.
  """
  if _utils.is_set(deps):
    override_deps_before = self._override_deps
    self._override_deps = _utils.Some(deps)
  else:
    override_deps_before = _utils.UNSET
  # noinspection PyTypeChecker
  if _utils.is_set(model):
    override_model_before = self._override_model
    # noinspection PyTypeChecker
    self._override_model = _utils.Some(models.infer_model(model)) # pyright: ignore[reportArgumentType]
  else:
    override_model_before = _utils.UNSET
  try:
    yield
  finally:
    if _utils.is_set(override_deps_before):
      self._override_deps = override_deps_before
    if _utils.is_set(override_model_before):
      self._override_model = override_model_before

```
  
---|---  
####  system_prompt
```
system_prompt(
  func: [[[]], ],
) -> [[[]], ]

```

```
system_prompt(
  func: [
    [[]], []
  ],
) -> [[[]], []]

```

```
system_prompt(func: [[], ]) -> [[], ]

```

```
system_prompt(
  func: [[], []],
) -> [[], []]

```

```
system_prompt(*, dynamic:  = False) -> [
  [[]],
  [],
]

```

```
system_prompt(
  func: [] | None = None,
  /,
  *,
  dynamic:  = False,
) -> (
  [
    [[]],
    [],
  ]
  | []
)

```

Decorator to register a system prompt function.
Optionally takes [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) as its only argument. Can decorate a sync or async functions.
The decorator can be used either bare (`agent.system_prompt`) or as a function call (`agent.system_prompt(...)`), see the examples below.
Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure the type of the function, see `tests/typed_agent.py` for tests.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`func` |  `SystemPromptFunc[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.SystemPromptFunc "pydantic_ai._system_prompt.SystemPromptFunc")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT")] | None` |  The function to decorate |  `None`  
`dynamic` |  If True, the system prompt will be reevaluated even when `messages_history` is provided, see [`SystemPromptPart.dynamic_ref`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.SystemPromptPart.dynamic_ref) |  `False`  
Example: 
```
frompydantic_aiimport Agent, RunContext
agent = Agent('test', deps_type=str)
@agent.system_prompt
defsimple_system_prompt() -> str:
  return 'foobar'
@agent.system_prompt(dynamic=True)
async defasync_system_prompt(ctx: RunContext[str]) -> str:
  return f'{ctx.deps} is the best'

```

Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defsystem_prompt(
  self,
  func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
  /,
  *,
  dynamic: bool = False,
) -> (
  Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
  | _system_prompt.SystemPromptFunc[AgentDepsT]
):
"""Decorator to register a system prompt function.
  Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
  Can decorate a sync or async functions.
  The decorator can be used either bare (`agent.system_prompt`) or as a function call
  (`agent.system_prompt(...)`), see the examples below.
  Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
  the type of the function, see `tests/typed_agent.py` for tests.
  Args:
    func: The function to decorate
    dynamic: If True, the system prompt will be reevaluated even when `messages_history` is provided,
      see [`SystemPromptPart.dynamic_ref`][pydantic_ai.messages.SystemPromptPart.dynamic_ref]
  Example:
  ```python
  from pydantic_ai import Agent, RunContext
  agent = Agent('test', deps_type=str)
  @agent.system_prompt
  def simple_system_prompt() -> str:
    return 'foobar'
  @agent.system_prompt(dynamic=True)
  async def async_system_prompt(ctx: RunContext[str]) -> str:
    return f'{ctx.deps} is the best'
  ```
  """
  if func is None:
    defdecorator(
      func_: _system_prompt.SystemPromptFunc[AgentDepsT],
    ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
      runner = _system_prompt.SystemPromptRunner[AgentDepsT](func_, dynamic=dynamic)
      self._system_prompt_functions.append(runner)
      if dynamic:
        self._system_prompt_dynamic_functions[func_.__qualname__] = runner
      return func_
    return decorator
  else:
    assert not dynamic, "dynamic can't be True in this case"
    self._system_prompt_functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](func, dynamic=dynamic))
    return func

```
  
---|---  
####  result_validator
```
result_validator(
  func: [
    [[], ], 
  ],
) -> [
  [[], ], 
]

```

```
result_validator(
  func: [
    [[], ],
    [],
  ],
) -> [
  [[], ],
  [],
]

```

```
result_validator(
  func: [[], ],
) -> [[], ]

```

```
result_validator(
  func: [[], []],
) -> [[], []]

```

```
result_validator(
  func: ResultValidatorFunc[, ],
) -> ResultValidatorFunc[, ]

```

Decorator to register a result validator function.
Optionally takes [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) as its first argument. Can decorate a sync or async functions.
Overloads for every possible signature of `result_validator` are included so the decorator doesn't obscure the type of the function, see `tests/typed_agent.py` for tests.
Example: 
```
frompydantic_aiimport Agent, ModelRetry, RunContext
agent = Agent('test', deps_type=str)
@agent.result_validator
defresult_validator_simple(data: str) -> str:
  if 'wrong' in data:
    raise ModelRetry('wrong response')
  return data
@agent.result_validator
async defresult_validator_deps(ctx: RunContext[str], data: str) -> str:
  if ctx.deps in data:
    raise ModelRetry('wrong response')
  return data
result = agent.run_sync('foobar', deps='spam')
print(result.data)
#> success (no tool calls)

```

Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defresult_validator(
  self, func: _result.ResultValidatorFunc[AgentDepsT, ResultDataT], /
) -> _result.ResultValidatorFunc[AgentDepsT, ResultDataT]:
"""Decorator to register a result validator function.
  Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
  Can decorate a sync or async functions.
  Overloads for every possible signature of `result_validator` are included so the decorator doesn't obscure
  the type of the function, see `tests/typed_agent.py` for tests.
  Example:
  ```python
  from pydantic_ai import Agent, ModelRetry, RunContext
  agent = Agent('test', deps_type=str)
  @agent.result_validator
  def result_validator_simple(data: str) -> str:
    if 'wrong' in data:
      raise ModelRetry('wrong response')
    return data
  @agent.result_validator
  async def result_validator_deps(ctx: RunContext[str], data: str) -> str:
    if ctx.deps in data:
      raise ModelRetry('wrong response')
    return data
  result = agent.run_sync('foobar', deps='spam')
  print(result.data)
  #> success (no tool calls)
  ```
  """
  self._result_validators.append(_result.ResultValidator[AgentDepsT, Any](func))
  return func

```
  
---|---  
####  tool
```
tool(
  func: [, ],
) -> [, ]

```

```
tool(
  *,
  name:  | None = None,
  retries:  | None = None,
  prepare: [] | None = None,
  docstring_format:  = "auto",
  require_parameter_descriptions:  = False,
  schema_generator: [
    
  ] = GenerateToolJsonSchema
) -> [
  [[, ]],
  [, ],
]

```

```
tool(
  func: (
    [, ] | None
  ) = None,
  /,
  *,
  name:  | None = None,
  retries:  | None = None,
  prepare: [] | None = None,
  docstring_format:  = "auto",
  require_parameter_descriptions:  = False,
  schema_generator: [
    
  ] = GenerateToolJsonSchema,
) -> 

```

Decorator to register a tool function which takes [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) as its first argument.
Can decorate a sync or async functions.
The docstring is inspected to extract both the tool description and description of each parameter, [learn more](https://ai.pydantic.dev/tools/#function-tools-and-schema).
We can't add overloads for every possible signature of tool, since the return type is a recursive union so the signature of functions decorated with `@agent.tool` is obscured.
Example: 
```
frompydantic_aiimport Agent, RunContext
agent = Agent('test', deps_type=int)
@agent.tool
deffoobar(ctx: RunContext[int], x: int) -> int:
  return ctx.deps + x
@agent.tool(retries=2)
async defspam(ctx: RunContext[str], y: float) -> float:
  return ctx.deps + y
result = agent.run_sync('foobar', deps=1)
print(result.data)
#> {"foobar":1,"spam":1.0}

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`func` |  `ToolFuncContext[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolFuncContext "pydantic_ai.tools.ToolFuncContext")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), ToolParams[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolParams "pydantic_ai.tools.ToolParams")] | None` |  The tool function to register. |  `None`  
`name` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The name of the tool, defaults to the function name. |  `None`  
`retries` |  `int[](https://docs.python.org/3/library/functions.html#int) | None` |  The number of retries to allow for this tool, defaults to the agent's default retries, which defaults to 1. |  `None`  
`prepare` |  `ToolPrepareFunc[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolPrepareFunc "pydantic_ai.tools.ToolPrepareFunc")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT")] | None` |  custom method to prepare the tool definition for each step, return `None` to omit this tool from a given step. This is useful if you want to customise a tool at call time, or omit it completely from a step. See [`ToolPrepareFunc`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolPrepareFunc). |  `None`  
`docstring_format` |  |  The format of the docstring, see [`DocstringFormat`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.DocstringFormat). Defaults to `'auto'`, such that the format is inferred from the structure of the docstring. |  `'auto'`  
`require_parameter_descriptions` |  If True, raise an error if a parameter description is missing. Defaults to False. |  `False`  
`schema_generator` |  |  The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`. |  `GenerateToolJsonSchema`  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
deftool(
  self,
  func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
  /,
  *,
  name: str | None = None,
  retries: int | None = None,
  prepare: ToolPrepareFunc[AgentDepsT] | None = None,
  docstring_format: DocstringFormat = 'auto',
  require_parameter_descriptions: bool = False,
  schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
) -> Any:
"""Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
  Can decorate a sync or async functions.
  The docstring is inspected to extract both the tool description and description of each parameter,
  [learn more](../tools.md#function-tools-and-schema).
  We can't add overloads for every possible signature of tool, since the return type is a recursive union
  so the signature of functions decorated with `@agent.tool` is obscured.
  Example:
  ```python
  from pydantic_ai import Agent, RunContext
  agent = Agent('test', deps_type=int)
  @agent.tool
  def foobar(ctx: RunContext[int], x: int) -> int:
    return ctx.deps + x
  @agent.tool(retries=2)
  async def spam(ctx: RunContext[str], y: float) -> float:
    return ctx.deps + y
  result = agent.run_sync('foobar', deps=1)
  print(result.data)
  #> {"foobar":1,"spam":1.0}
  ```
  Args:
    func: The tool function to register.
    name: The name of the tool, defaults to the function name.
    retries: The number of retries to allow for this tool, defaults to the agent's default retries,
      which defaults to 1.
    prepare: custom method to prepare the tool definition for each step, return `None` to omit this
      tool from a given step. This is useful if you want to customise a tool at call time,
      or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
    docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
      Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
    require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
    schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
  """
  if func is None:
    deftool_decorator(
      func_: ToolFuncContext[AgentDepsT, ToolParams],
    ) -> ToolFuncContext[AgentDepsT, ToolParams]:
      # noinspection PyTypeChecker
      self._register_function(
        func_,
        True,
        name,
        retries,
        prepare,
        docstring_format,
        require_parameter_descriptions,
        schema_generator,
      )
      return func_
    return tool_decorator
  else:
    # noinspection PyTypeChecker
    self._register_function(
      func, True, name, retries, prepare, docstring_format, require_parameter_descriptions, schema_generator
    )
    return func

```
  
---|---  
####  tool_plain
```
tool_plain(
  func: [],
) -> []

```

```
tool_plain(
  *,
  name:  | None = None,
  retries:  | None = None,
  prepare: [] | None = None,
  docstring_format:  = "auto",
  require_parameter_descriptions:  = False,
  schema_generator: [
    
  ] = GenerateToolJsonSchema
) -> [
  [[]], []
]

```

```
tool_plain(
  func: [] | None = None,
  /,
  *,
  name:  | None = None,
  retries:  | None = None,
  prepare: [] | None = None,
  docstring_format:  = "auto",
  require_parameter_descriptions:  = False,
  schema_generator: [
    
  ] = GenerateToolJsonSchema,
) -> 

```

Decorator to register a tool function which DOES NOT take `RunContext` as an argument.
Can decorate a sync or async functions.
The docstring is inspected to extract both the tool description and description of each parameter, [learn more](https://ai.pydantic.dev/tools/#function-tools-and-schema).
We can't add overloads for every possible signature of tool, since the return type is a recursive union so the signature of functions decorated with `@agent.tool` is obscured.
Example: 
```
frompydantic_aiimport Agent, RunContext
agent = Agent('test')
@agent.tool
deffoobar(ctx: RunContext[int]) -> int:
  return 123
@agent.tool(retries=2)
async defspam(ctx: RunContext[str]) -> float:
  return 3.14
result = agent.run_sync('foobar', deps=1)
print(result.data)
#> {"foobar":123,"spam":3.14}

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`func` |  `ToolFuncPlain[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolFuncPlain "pydantic_ai.tools.ToolFuncPlain")[ToolParams[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolParams "pydantic_ai.tools.ToolParams")] | None` |  The tool function to register. |  `None`  
`name` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The name of the tool, defaults to the function name. |  `None`  
`retries` |  `int[](https://docs.python.org/3/library/functions.html#int) | None` |  The number of retries to allow for this tool, defaults to the agent's default retries, which defaults to 1. |  `None`  
`prepare` |  `ToolPrepareFunc[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolPrepareFunc "pydantic_ai.tools.ToolPrepareFunc")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT")] | None` |  custom method to prepare the tool definition for each step, return `None` to omit this tool from a given step. This is useful if you want to customise a tool at call time, or omit it completely from a step. See [`ToolPrepareFunc`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolPrepareFunc). |  `None`  
`docstring_format` |  |  The format of the docstring, see [`DocstringFormat`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.DocstringFormat). Defaults to `'auto'`, such that the format is inferred from the structure of the docstring. |  `'auto'`  
`require_parameter_descriptions` |  If True, raise an error if a parameter description is missing. Defaults to False. |  `False`  
`schema_generator` |  |  The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`. |  `GenerateToolJsonSchema`  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
deftool_plain(
  self,
  func: ToolFuncPlain[ToolParams] | None = None,
  /,
  *,
  name: str | None = None,
  retries: int | None = None,
  prepare: ToolPrepareFunc[AgentDepsT] | None = None,
  docstring_format: DocstringFormat = 'auto',
  require_parameter_descriptions: bool = False,
  schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
) -> Any:
"""Decorator to register a tool function which DOES NOT take `RunContext` as an argument.
  Can decorate a sync or async functions.
  The docstring is inspected to extract both the tool description and description of each parameter,
  [learn more](../tools.md#function-tools-and-schema).
  We can't add overloads for every possible signature of tool, since the return type is a recursive union
  so the signature of functions decorated with `@agent.tool` is obscured.
  Example:
  ```python
  from pydantic_ai import Agent, RunContext
  agent = Agent('test')
  @agent.tool
  def foobar(ctx: RunContext[int]) -> int:
    return 123
  @agent.tool(retries=2)
  async def spam(ctx: RunContext[str]) -> float:
    return 3.14
  result = agent.run_sync('foobar', deps=1)
  print(result.data)
  #> {"foobar":123,"spam":3.14}
  ```
  Args:
    func: The tool function to register.
    name: The name of the tool, defaults to the function name.
    retries: The number of retries to allow for this tool, defaults to the agent's default retries,
      which defaults to 1.
    prepare: custom method to prepare the tool definition for each step, return `None` to omit this
      tool from a given step. This is useful if you want to customise a tool at call time,
      or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
    docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
      Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
    require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
    schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
  """
  if func is None:
    deftool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
      # noinspection PyTypeChecker
      self._register_function(
        func_,
        False,
        name,
        retries,
        prepare,
        docstring_format,
        require_parameter_descriptions,
        schema_generator,
      )
      return func_
    return tool_decorator
  else:
    self._register_function(
      func, False, name, retries, prepare, docstring_format, require_parameter_descriptions, schema_generator
    )
    return func

```
  
---|---  
####  is_model_request_node `staticmethod`
```
is_model_request_node(
  node: AgentNode[, ] | [FinalResult[]],
) -> [ModelRequestNode[, ]]

```

Check if the node is a `ModelRequestNode`, narrowing the type if it is.
This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@staticmethod
defis_model_request_node(
  node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
) -> TypeGuard[_agent_graph.ModelRequestNode[T, S]]:
"""Check if the node is a `ModelRequestNode`, narrowing the type if it is.
  This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
  """
  return isinstance(node, _agent_graph.ModelRequestNode)

```
  
---|---  
####  is_call_tools_node `staticmethod`
```
is_call_tools_node(
  node: AgentNode[, ] | [FinalResult[]],
) -> [CallToolsNode[, ]]

```

Check if the node is a `CallToolsNode`, narrowing the type if it is.
This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@staticmethod
defis_call_tools_node(
  node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
) -> TypeGuard[_agent_graph.CallToolsNode[T, S]]:
"""Check if the node is a `CallToolsNode`, narrowing the type if it is.
  This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
  """
  return isinstance(node, _agent_graph.CallToolsNode)

```
  
---|---  
####  is_user_prompt_node `staticmethod`
```
is_user_prompt_node(
  node: AgentNode[, ] | [FinalResult[]],
) -> [UserPromptNode[, ]]

```

Check if the node is a `UserPromptNode`, narrowing the type if it is.
This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@staticmethod
defis_user_prompt_node(
  node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
) -> TypeGuard[_agent_graph.UserPromptNode[T, S]]:
"""Check if the node is a `UserPromptNode`, narrowing the type if it is.
  This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
  """
  return isinstance(node, _agent_graph.UserPromptNode)

```
  
---|---  
####  is_end_node `staticmethod`
```
is_end_node(
  node: AgentNode[, ] | [FinalResult[]],
) -> [[FinalResult[]]]

```

Check if the node is a `End`, narrowing the type if it is.
This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@staticmethod
defis_end_node(
  node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
) -> TypeGuard[End[result.FinalResult[S]]]:
"""Check if the node is a `End`, narrowing the type if it is.
  This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
  """
  return isinstance(node, End)

```
  
---|---  
####  run_mcp_servers `async`
```
run_mcp_servers() -> [None]

```

Run [`MCPServerStdio`s](https://ai.pydantic.dev/api/mcp/#pydantic_ai.mcp.MCPServerStdio) so they can be used by the agent.
Returns: a context manager to start and shutdown the servers.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@asynccontextmanager
async defrun_mcp_servers(self) -> AsyncIterator[None]:
"""Run [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] so they can be used by the agent.
  Returns: a context manager to start and shutdown the servers.
  """
  exit_stack = AsyncExitStack()
  try:
    for mcp_server in self._mcp_servers:
      await exit_stack.enter_async_context(mcp_server)
    yield
  finally:
    await exit_stack.aclose()

```
  
---|---  
###  AgentRun `dataclass`
Bases: `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")]`
A stateful, async-iterable run of an [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent).
You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.
Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) is reached, the run finishes and [`result`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRun.result) becomes available.
Example: 
```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
async defmain():
  nodes = []
  # Iterate through the run, recording each node along the way:
  async with agent.iter('What is the capital of France?') as agent_run:
    async for node in agent_run:
      nodes.append(node)
  print(nodes)
'''

    ModelRequestNode(
      request=ModelRequest(
        parts=[
          UserPromptPart(
            content='What is the capital of France?',
            timestamp=datetime.datetime(...),
            part_kind='user-prompt',


        kind='request',

    ),
    CallToolsNode(
      model_response=ModelResponse(
        parts=[TextPart(content='Paris', part_kind='text')],
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
        kind='response',

    ),
    End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

  '''
  print(agent_run.result.data)
  #> Paris

```

You can also manually drive the iteration using the [`next`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRun.next) method for more granular control.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@dataclasses.dataclass(repr=False)
classAgentRun(Generic[AgentDepsT, ResultDataT]):
"""A stateful, async-iterable run of an [`Agent`][pydantic_ai.agent.Agent].
  You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.
  Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an
  [`End`][pydantic_graph.nodes.End] is reached, the run finishes and [`result`][pydantic_ai.agent.AgentRun.result]
  becomes available.
  Example:
  ```python
  from pydantic_ai import Agent
  agent = Agent('openai:gpt-4o')
  async def main():
    nodes = []
    # Iterate through the run, recording each node along the way:
    async with agent.iter('What is the capital of France?') as agent_run:
      async for node in agent_run:
        nodes.append(node)
    print(nodes)
    '''

      ModelRequestNode(
        request=ModelRequest(
          parts=[
            UserPromptPart(
              content='What is the capital of France?',
              timestamp=datetime.datetime(...),
              part_kind='user-prompt',


          kind='request',


      CallToolsNode(
        model_response=ModelResponse(
          parts=[TextPart(content='Paris', part_kind='text')],
          model_name='gpt-4o',
          timestamp=datetime.datetime(...),
          kind='response',


      End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

    '''
    print(agent_run.result.data)
    #> Paris
  ```
  You can also manually drive the iteration using the [`next`][pydantic_ai.agent.AgentRun.next] method for
  more granular control.
  """
  _graph_run: GraphRun[
    _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[ResultDataT]
  ]
  @property
  defctx(self) -> GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]]:
"""The current context of the agent run."""
    return GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]](
      self._graph_run.state, self._graph_run.deps
    )
  @property
  defnext_node(
    self,
  ) -> _agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]:
"""The next node that will be run in the agent graph.
    This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
    """
    next_node = self._graph_run.next_node
    if isinstance(next_node, End):
      return next_node
    if _agent_graph.is_agent_node(next_node):
      return next_node
    raise exceptions.AgentRunError(f'Unexpected node type: {type(next_node)}') # pragma: no cover
  @property
  defresult(self) -> AgentRunResult[ResultDataT] | None:
"""The final result of the run if it has ended, otherwise `None`.
    Once the run returns an [`End`][pydantic_graph.nodes.End] node, `result` is populated
    with an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
    """
    graph_run_result = self._graph_run.result
    if graph_run_result is None:
      return None
    return AgentRunResult(
      graph_run_result.output.data,
      graph_run_result.output.tool_name,
      graph_run_result.state,
      self._graph_run.deps.new_message_index,
    )
  def__aiter__(
    self,
  ) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]]:
"""Provide async-iteration over the nodes in the agent run."""
    return self
  async def__anext__(
    self,
  ) -> _agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]:
"""Advance to the next node automatically based on the last returned node."""
    next_node = await self._graph_run.__anext__()
    if _agent_graph.is_agent_node(next_node):
      return next_node
    assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
    return next_node
  async defnext(
    self,
    node: _agent_graph.AgentNode[AgentDepsT, ResultDataT],
  ) -> _agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]:
"""Manually drive the agent run by passing in the node you want to run next.
    This lets you inspect or mutate the node before continuing execution, or skip certain nodes
    under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.nodes.End]
    node.
    Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_graph import End
    agent = Agent('openai:gpt-4o')
    async def main():
      async with agent.iter('What is the capital of France?') as agent_run:
        next_node = agent_run.next_node # start with the first node
        nodes = [next_node]
        while not isinstance(next_node, End):
          next_node = await agent_run.next(next_node)
          nodes.append(next_node)
        # Once `next_node` is an End, we've finished:
        print(nodes)
        '''

          UserPromptNode(
            user_prompt='What is the capital of France?',
            system_prompts=(),
            system_prompt_functions=[],
            system_prompt_dynamic_functions={},

          ModelRequestNode(
            request=ModelRequest(
              parts=[
                UserPromptPart(
                  content='What is the capital of France?',
                  timestamp=datetime.datetime(...),
                  part_kind='user-prompt',


              kind='request',


          CallToolsNode(
            model_response=ModelResponse(
              parts=[TextPart(content='Paris', part_kind='text')],
              model_name='gpt-4o',
              timestamp=datetime.datetime(...),
              kind='response',


          End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

        '''
        print('Final result:', agent_run.result.data)
        #> Final result: Paris
    ```
    Args:
      node: The node to run next in the graph.
    Returns:
      The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
      the run has completed.
    """
    # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
    # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
    next_node = await self._graph_run.next(node)
    if _agent_graph.is_agent_node(next_node):
      return next_node
    assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
    return next_node
  defusage(self) -> _usage.Usage:
"""Get usage statistics for the run so far, including token usage, model requests, and so on."""
    return self._graph_run.state.usage
  def__repr__(self) -> str:
    result = self._graph_run.result
    result_repr = '<run not finished>' if result is None else repr(result.output)
    return f'<{type(self).__name__} result={result_repr} usage={self.usage()}>'

```
  
---|---  
####  ctx `property`
```
ctx: [
  GraphAgentState, GraphAgentDeps[, ]
]

```

The current context of the agent run.
####  next_node `property`
```
next_node: (
  AgentNode[, ]
  | [FinalResult[]]
)

```

The next node that will be run in the agent graph.
This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
####  result `property`
```
result: [] | None

```

The final result of the run if it has ended, otherwise `None`.
Once the run returns an [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) node, `result` is populated with an [`AgentRunResult`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult).
####  __aiter__
```
__aiter__() -> (
  [
    AgentNode[, ]
    | [FinalResult[]]
  ]
)

```

Provide async-iteration over the nodes in the agent run.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
def__aiter__(
  self,
) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]]:
"""Provide async-iteration over the nodes in the agent run."""
  return self

```
  
---|---  
####  __anext__ `async`
```
__anext__() -> (
  AgentNode[, ]
  | [FinalResult[]]
)

```

Advance to the next node automatically based on the last returned node.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
async def__anext__(
  self,
) -> _agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]:
"""Advance to the next node automatically based on the last returned node."""
  next_node = await self._graph_run.__anext__()
  if _agent_graph.is_agent_node(next_node):
    return next_node
  assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
  return next_node

```
  
---|---  
####  next `async`
```
next(
  node: AgentNode[, ],
) -> (
  AgentNode[, ]
  | [FinalResult[]]
)

```

Manually drive the agent run by passing in the node you want to run next.
This lets you inspect or mutate the node before continuing execution, or skip certain nodes under dynamic conditions. The agent run should be stopped when you return an [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) node.
Example: 
```
frompydantic_aiimport Agent
frompydantic_graphimport End
agent = Agent('openai:gpt-4o')
async defmain():
  async with agent.iter('What is the capital of France?') as agent_run:
    next_node = agent_run.next_node # start with the first node
    nodes = [next_node]
    while not isinstance(next_node, End):
      next_node = await agent_run.next(next_node)
      nodes.append(next_node)
    # Once `next_node` is an End, we've finished:
    print(nodes)
'''

      UserPromptNode(
        user_prompt='What is the capital of France?',
        system_prompts=(),
        system_prompt_functions=[],
        system_prompt_dynamic_functions={},

      ModelRequestNode(
        request=ModelRequest(
          parts=[
            UserPromptPart(
              content='What is the capital of France?',
              timestamp=datetime.datetime(...),
              part_kind='user-prompt',


          kind='request',


      CallToolsNode(
        model_response=ModelResponse(
          parts=[TextPart(content='Paris', part_kind='text')],
          model_name='gpt-4o',
          timestamp=datetime.datetime(...),
          kind='response',


      End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

    '''
    print('Final result:', agent_run.result.data)
    #> Final result: Paris

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`node` |  `AgentNode[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")]` |  The node to run next in the graph. |  _required_  
Returns:
Type | Description  
---|---  
`AgentNode[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")] | End[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End "pydantic_graph.End")[FinalResult[ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")]]` |  The next node returned by the graph logic, or an [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) node if  
`AgentNode[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")] | End[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End "pydantic_graph.End")[FinalResult[ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")]]` |  the run has completed.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
async defnext(
  self,
  node: _agent_graph.AgentNode[AgentDepsT, ResultDataT],
) -> _agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]:
"""Manually drive the agent run by passing in the node you want to run next.
  This lets you inspect or mutate the node before continuing execution, or skip certain nodes
  under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.nodes.End]
  node.
  Example:
  ```python
  from pydantic_ai import Agent
  from pydantic_graph import End
  agent = Agent('openai:gpt-4o')
  async def main():
    async with agent.iter('What is the capital of France?') as agent_run:
      next_node = agent_run.next_node # start with the first node
      nodes = [next_node]
      while not isinstance(next_node, End):
        next_node = await agent_run.next(next_node)
        nodes.append(next_node)
      # Once `next_node` is an End, we've finished:
      print(nodes)
      '''

        UserPromptNode(
          user_prompt='What is the capital of France?',
          system_prompts=(),
          system_prompt_functions=[],
          system_prompt_dynamic_functions={},

        ModelRequestNode(
          request=ModelRequest(
            parts=[
              UserPromptPart(
                content='What is the capital of France?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',


            kind='request',


        CallToolsNode(
          model_response=ModelResponse(
            parts=[TextPart(content='Paris', part_kind='text')],
            model_name='gpt-4o',
            timestamp=datetime.datetime(...),
            kind='response',


        End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),

      '''
      print('Final result:', agent_run.result.data)
      #> Final result: Paris
  ```
  Args:
    node: The node to run next in the graph.
  Returns:
    The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
    the run has completed.
  """
  # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
  # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
  next_node = await self._graph_run.next(node)
  if _agent_graph.is_agent_node(next_node):
    return next_node
  assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
  return next_node

```
  
---|---  
####  usage
```
usage() -> 

```

Get usage statistics for the run so far, including token usage, model requests, and so on.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defusage(self) -> _usage.Usage:
"""Get usage statistics for the run so far, including token usage, model requests, and so on."""
  return self._graph_run.state.usage

```
  
---|---  
###  AgentRunResult `dataclass`
Bases: 
The final result of an agent run.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
@dataclasses.dataclass
classAgentRunResult(Generic[ResultDataT]):
"""The final result of an agent run."""
  data: ResultDataT # TODO: rename this to output. I'm putting this off for now mostly to reduce the size of the diff
  _result_tool_name: str | None = dataclasses.field(repr=False)
  _state: _agent_graph.GraphAgentState = dataclasses.field(repr=False)
  _new_message_index: int = dataclasses.field(repr=False)
  def_set_result_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
"""Set return content for the result tool.
    Useful if you want to continue the conversation and want to set the response to the result tool call.
    """
    if not self._result_tool_name:
      raise ValueError('Cannot set result tool return content when the return type is `str`.')
    messages = deepcopy(self._state.message_history)
    last_message = messages[-1]
    for part in last_message.parts:
      if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self._result_tool_name:
        part.content = return_content
        return messages
    raise LookupError(f'No tool call found with tool name {self._result_tool_name!r}.')
  defall_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
"""Return the history of _messages.
    Args:
      result_tool_return_content: The return content of the tool call to set in the last message.
        This provides a convenient way to modify the content of the result tool call if you want to continue
        the conversation and want to set the response to the result tool call. If `None`, the last message will
        not be modified.
    Returns:
      List of messages.
    """
    if result_tool_return_content is not None:
      return self._set_result_tool_return(result_tool_return_content)
    else:
      return self._state.message_history
  defall_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
"""Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.
    Args:
      result_tool_return_content: The return content of the tool call to set in the last message.
        This provides a convenient way to modify the content of the result tool call if you want to continue
        the conversation and want to set the response to the result tool call. If `None`, the last message will
        not be modified.
    Returns:
      JSON bytes representing the messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(
      self.all_messages(result_tool_return_content=result_tool_return_content)
    )
  defnew_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
"""Return new messages associated with this run.
    Messages from older runs are excluded.
    Args:
      result_tool_return_content: The return content of the tool call to set in the last message.
        This provides a convenient way to modify the content of the result tool call if you want to continue
        the conversation and want to set the response to the result tool call. If `None`, the last message will
        not be modified.
    Returns:
      List of new messages.
    """
    return self.all_messages(result_tool_return_content=result_tool_return_content)[self._new_message_index :]
  defnew_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
"""Return new messages from [`new_messages`][pydantic_ai.agent.AgentRunResult.new_messages] as JSON bytes.
    Args:
      result_tool_return_content: The return content of the tool call to set in the last message.
        This provides a convenient way to modify the content of the result tool call if you want to continue
        the conversation and want to set the response to the result tool call. If `None`, the last message will
        not be modified.
    Returns:
      JSON bytes representing the new messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(
      self.new_messages(result_tool_return_content=result_tool_return_content)
    )
  defusage(self) -> _usage.Usage:
"""Return the usage of the whole run."""
    return self._state.usage

```
  
---|---  
####  all_messages
```
all_messages(
  *, result_tool_return_content:  | None = None
) -> []

```

Return the history of _messages.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`result_tool_return_content` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the result tool call if you want to continue the conversation and want to set the response to the result tool call. If `None`, the last message will not be modified. |  `None`  
Returns:
Type | Description  
---|---  
|  List of messages.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defall_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
"""Return the history of _messages.
  Args:
    result_tool_return_content: The return content of the tool call to set in the last message.
      This provides a convenient way to modify the content of the result tool call if you want to continue
      the conversation and want to set the response to the result tool call. If `None`, the last message will
      not be modified.
  Returns:
    List of messages.
  """
  if result_tool_return_content is not None:
    return self._set_result_tool_return(result_tool_return_content)
  else:
    return self._state.message_history

```
  
---|---  
####  all_messages_json
```
all_messages_json(
  *, result_tool_return_content:  | None = None
) -> 

```

Return all messages from [`all_messages`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult.all_messages) as JSON bytes.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`result_tool_return_content` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the result tool call if you want to continue the conversation and want to set the response to the result tool call. If `None`, the last message will not be modified. |  `None`  
Returns:
Type | Description  
---|---  
|  JSON bytes representing the messages.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defall_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
"""Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.
  Args:
    result_tool_return_content: The return content of the tool call to set in the last message.
      This provides a convenient way to modify the content of the result tool call if you want to continue
      the conversation and want to set the response to the result tool call. If `None`, the last message will
      not be modified.
  Returns:
    JSON bytes representing the messages.
  """
  return _messages.ModelMessagesTypeAdapter.dump_json(
    self.all_messages(result_tool_return_content=result_tool_return_content)
  )

```
  
---|---  
####  new_messages
```
new_messages(
  *, result_tool_return_content:  | None = None
) -> []

```

Return new messages associated with this run.
Messages from older runs are excluded.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`result_tool_return_content` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the result tool call if you want to continue the conversation and want to set the response to the result tool call. If `None`, the last message will not be modified. |  `None`  
Returns:
Type | Description  
---|---  
|  List of new messages.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defnew_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
"""Return new messages associated with this run.
  Messages from older runs are excluded.
  Args:
    result_tool_return_content: The return content of the tool call to set in the last message.
      This provides a convenient way to modify the content of the result tool call if you want to continue
      the conversation and want to set the response to the result tool call. If `None`, the last message will
      not be modified.
  Returns:
    List of new messages.
  """
  return self.all_messages(result_tool_return_content=result_tool_return_content)[self._new_message_index :]

```
  
---|---  
####  new_messages_json
```
new_messages_json(
  *, result_tool_return_content:  | None = None
) -> 

```

Return new messages from [`new_messages`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult.new_messages) as JSON bytes.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`result_tool_return_content` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the result tool call if you want to continue the conversation and want to set the response to the result tool call. If `None`, the last message will not be modified. |  `None`  
Returns:
Type | Description  
---|---  
|  JSON bytes representing the new messages.  
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defnew_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
"""Return new messages from [`new_messages`][pydantic_ai.agent.AgentRunResult.new_messages] as JSON bytes.
  Args:
    result_tool_return_content: The return content of the tool call to set in the last message.
      This provides a convenient way to modify the content of the result tool call if you want to continue
      the conversation and want to set the response to the result tool call. If `None`, the last message will
      not be modified.
  Returns:
    JSON bytes representing the new messages.
  """
  return _messages.ModelMessagesTypeAdapter.dump_json(
    self.new_messages(result_tool_return_content=result_tool_return_content)
  )

```
  
---|---  
####  usage
```
usage() -> 

```

Return the usage of the whole run.
Source code in `pydantic_ai_slim/pydantic_ai/agent.py`
```
```
| ```
defusage(self) -> _usage.Usage:
"""Return the usage of the whole run."""
  return self._state.usage

```
  
---|---  
###  EndStrategy `module-attribute`
```
EndStrategy = EndStrategy

```

###  RunResultDataT `module-attribute`
```
RunResultDataT = TypeVar('RunResultDataT')

```

Type variable for the result data of a run where `result_type` was customized on the run call.
###  capture_run_messages `module-attribute`
```
capture_run_messages = capture_run_messages

```

###  InstrumentationSettings `dataclass`
Options for instrumenting models and agents with OpenTelemetry.
Used in:
  * `Agent(instrument=...)`
  * [`Agent.instrument_all()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.instrument_all)


See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
Source code in `pydantic_ai_slim/pydantic_ai/models/instrumented.py`
```
```
| ```
@dataclass(init=False)
classInstrumentationSettings:
"""Options for instrumenting models and agents with OpenTelemetry.
  Used in:
  - `Agent(instrument=...)`
  - [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all]
  - [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel]
  See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
  """
  tracer: Tracer = field(repr=False)
  event_logger: EventLogger = field(repr=False)
  event_mode: Literal['attributes', 'logs'] = 'attributes'
  def__init__(
    self,
    *,
    event_mode: Literal['attributes', 'logs'] = 'attributes',
    tracer_provider: TracerProvider | None = None,
    event_logger_provider: EventLoggerProvider | None = None,
  ):
"""Create instrumentation options.
    Args:
      event_mode: The mode for emitting events. If `'attributes'`, events are attached to the span as attributes.
        If `'logs'`, events are emitted as OpenTelemetry log-based events.
      tracer_provider: The OpenTelemetry tracer provider to use.
        If not provided, the global tracer provider is used.
        Calling `logfire.configure()` sets the global tracer provider, so most users don't need this.
      event_logger_provider: The OpenTelemetry event logger provider to use.
        If not provided, the global event logger provider is used.
        Calling `logfire.configure()` sets the global event logger provider, so most users don't need this.
        This is only used if `event_mode='logs'`.
    """
    frompydantic_aiimport __version__
    tracer_provider = tracer_provider or get_tracer_provider()
    event_logger_provider = event_logger_provider or get_event_logger_provider()
    self.tracer = tracer_provider.get_tracer('pydantic-ai', __version__)
    self.event_logger = event_logger_provider.get_event_logger('pydantic-ai', __version__)
    self.event_mode = event_mode

```
  
---|---  
####  __init__
```
__init__(
  *,
  event_mode: [
    "attributes", "logs"
  ] = "attributes",
  tracer_provider: TracerProvider | None = None,
  event_logger_provider: EventLoggerProvider | None = None
)

```

Create instrumentation options.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`event_mode` |  `Literal[](https://docs.python.org/3/library/typing.html#typing.Literal "typing.Literal")['attributes', 'logs']` |  The mode for emitting events. If `'attributes'`, events are attached to the span as attributes. If `'logs'`, events are emitted as OpenTelemetry log-based events. |  `'attributes'`  
`tracer_provider` |  `TracerProvider | None` |  The OpenTelemetry tracer provider to use. If not provided, the global tracer provider is used. Calling `logfire.configure()` sets the global tracer provider, so most users don't need this. |  `None`  
`event_logger_provider` |  `EventLoggerProvider | None` |  The OpenTelemetry event logger provider to use. If not provided, the global event logger provider is used. Calling `logfire.configure()` sets the global event logger provider, so most users don't need this. This is only used if `event_mode='logs'`. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/models/instrumented.py`
```
```
| ```
def__init__(
  self,
  *,
  event_mode: Literal['attributes', 'logs'] = 'attributes',
  tracer_provider: TracerProvider | None = None,
  event_logger_provider: EventLoggerProvider | None = None,
):
"""Create instrumentation options.
  Args:
    event_mode: The mode for emitting events. If `'attributes'`, events are attached to the span as attributes.
      If `'logs'`, events are emitted as OpenTelemetry log-based events.
    tracer_provider: The OpenTelemetry tracer provider to use.
      If not provided, the global tracer provider is used.
      Calling `logfire.configure()` sets the global tracer provider, so most users don't need this.
    event_logger_provider: The OpenTelemetry event logger provider to use.
      If not provided, the global event logger provider is used.
      Calling `logfire.configure()` sets the global event logger provider, so most users don't need this.
      This is only used if `event_mode='logs'`.
  """
  frompydantic_aiimport __version__
  tracer_provider = tracer_provider or get_tracer_provider()
  event_logger_provider = event_logger_provider or get_event_logger_provider()
  self.tracer = tracer_provider.get_tracer('pydantic-ai', __version__)
  self.event_logger = event_logger_provider.get_event_logger('pydantic-ai', __version__)
  self.event_mode = event_mode

```
  
---|---

# `pydantic_ai.common_tools`
###  duckduckgo_search_tool
```
duckduckgo_search_tool(
  duckduckgo_client:  | None = None,
  max_results:  | None = None,
)

```

Creates a DuckDuckGo search tool.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`duckduckgo_client` |  `DDGS | None` |  The DuckDuckGo search client. |  `None`  
`max_results` |  `int[](https://docs.python.org/3/library/functions.html#int) | None` |  The maximum number of results. If None, returns results only from the first response. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/common_tools/duckduckgo.py`
```
```
| ```
defduckduckgo_search_tool(duckduckgo_client: DDGS | None = None, max_results: int | None = None):
"""Creates a DuckDuckGo search tool.
  Args:
    duckduckgo_client: The DuckDuckGo search client.
    max_results: The maximum number of results. If None, returns results only from the first response.
  """
  return Tool(
    DuckDuckGoSearchTool(client=duckduckgo_client or DDGS(), max_results=max_results).__call__,
    name='duckduckgo_search',
    description='Searches DuckDuckGo for the given query and returns the results.',
  )

```
  
---|---  
###  tavily_search_tool
```
tavily_search_tool(api_key: )

```

Creates a Tavily search tool.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`api_key` |  The Tavily API key. You can get one by signing up at <https://app.tavily.com/home>. |  _required_  
Source code in `pydantic_ai_slim/pydantic_ai/common_tools/tavily.py`
```
```
| ```
deftavily_search_tool(api_key: str):
"""Creates a Tavily search tool.
  Args:
    api_key: The Tavily API key.
      You can get one by signing up at [https://app.tavily.com/home](https://app.tavily.com/home).
  """
  return Tool(
    TavilySearchTool(client=AsyncTavilyClient(api_key)).__call__,
    name='tavily_search',
    description='Searches Tavily for the given query and returns the results.',
  )

```
  
---|---

# `pydantic_ai.exceptions`
###  ModelRetry
Bases: 
Exception raised when a tool function should be retried.
The agent will return the message to the model and ask it to try calling the function/tool again.
Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`
```
```
| ```
classModelRetry(Exception):
"""Exception raised when a tool function should be retried.
  The agent will return the message to the model and ask it to try calling the function/tool again.
  """
  message: str
"""The message to return to the model."""
  def__init__(self, message: str):
    self.message = message
    super().__init__(message)

```
  
---|---  
####  message `instance-attribute`
```
message:  = message

```

The message to return to the model.
###  UserError
Bases: 
Error caused by a usage mistake by the application developer — You!
Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`
```
```
| ```
classUserError(RuntimeError):
"""Error caused by a usage mistake by the application developer — You!"""
  message: str
"""Description of the mistake."""
  def__init__(self, message: str):
    self.message = message
    super().__init__(message)

```
  
---|---  
####  message `instance-attribute`
```
message:  = message

```

Description of the mistake.
###  AgentRunError
Bases: 
Base class for errors occurring during an agent run.
Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`
```
```
| ```
classAgentRunError(RuntimeError):
"""Base class for errors occurring during an agent run."""
  message: str
"""The error message."""
  def__init__(self, message: str):
    self.message = message
    super().__init__(message)
  def__str__(self) -> str:
    return self.message

```
  
---|---  
####  message `instance-attribute`
```
message:  = message

```

The error message.
###  UsageLimitExceeded
Bases: 
Error raised when a Model's usage exceeds the specified limits.
Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`
```
```
| ```
classUsageLimitExceeded(AgentRunError):
"""Error raised when a Model's usage exceeds the specified limits."""

```
  
---|---  
###  UnexpectedModelBehavior
Bases: 
Error caused by unexpected Model behavior, e.g. an unexpected response code.
Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`
```
```
| ```
classUnexpectedModelBehavior(AgentRunError):
"""Error caused by unexpected Model behavior, e.g. an unexpected response code."""
  message: str
"""Description of the unexpected behavior."""
  body: str | None
"""The body of the response, if available."""
  def__init__(self, message: str, body: str | None = None):
    self.message = message
    if body is None:
      self.body: str | None = None
    else:
      try:
        self.body = json.dumps(json.loads(body), indent=2)
      except ValueError:
        self.body = body
    super().__init__(message)
  def__str__(self) -> str:
    if self.body:
      return f'{self.message}, body:\n{self.body}'
    else:
      return self.message

```
  
---|---  
####  message `instance-attribute`
```
message:  = message

```

Description of the unexpected behavior.
####  body `instance-attribute`
```
body:  | None = ((), indent=2)

```

The body of the response, if available.
###  ModelHTTPError
Bases: 
Raised when an model provider response has a status code of 4xx or 5xx.
Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`
```
```
| ```
classModelHTTPError(AgentRunError):
"""Raised when an model provider response has a status code of 4xx or 5xx."""
  status_code: int
"""The HTTP status code returned by the API."""
  model_name: str
"""The name of the model associated with the error."""
  body: object | None
"""The body of the response, if available."""
  message: str
"""The error message with the status code and response body, if available."""
  def__init__(self, status_code: int, model_name: str, body: object | None = None):
    self.status_code = status_code
    self.model_name = model_name
    self.body = body
    message = f'status_code: {status_code}, model_name: {model_name}, body: {body}'
    super().__init__(message)

```
  
---|---  
####  message `instance-attribute`
```
message: 

```

The error message with the status code and response body, if available.
####  status_code `instance-attribute`
```
status_code:  = status_code

```

The HTTP status code returned by the API.
####  model_name `instance-attribute`
```
model_name:  = model_name

```

The name of the model associated with the error.
####  body `instance-attribute`
```
body:  | None = 

```

The body of the response, if available.
###  FallbackExceptionGroup
Bases: `ExceptionGroup`
A group of exceptions that can be raised when all fallback models fail.
Source code in `pydantic_ai_slim/pydantic_ai/exceptions.py`
```
```
| ```
classFallbackExceptionGroup(ExceptionGroup):
"""A group of exceptions that can be raised when all fallback models fail."""

```
  
---|---

# `pydantic_ai.format_as_xml`
###  format_as_xml
```
format_as_xml(
  obj: ,
  root_tag:  = "examples",
  item_tag:  = "example",
  include_root_tag:  = True,
  none_str:  = "null",
  indent:  | None = " ",
) -> 

```

Format a Python object as XML.
This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML, rather than JSON etc.
Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`, `Iterable`, `dataclass`, and `BaseModel`.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`obj` |  Python Object to serialize to XML. |  _required_  
`root_tag` |  Outer tag to wrap the XML in, use `None` to omit the outer tag. |  `'examples'`  
`item_tag` |  Tag to use for each item in an iterable (e.g. list), this is overridden by the class name for dataclasses and Pydantic models. |  `'example'`  
`include_root_tag` |  Whether to include the root tag in the output (The root tag is always included if it includes a body - e.g. when the input is a simple value). |  `True`  
`none_str` |  String to use for `None` values. |  `'null'`  
`indent` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  Indentation string to use for pretty printing. |  `' '`  
Returns:
Type | Description  
---|---  
XML representation of the object.  
Example: 
format_as_xml_example.py```
frompydantic_ai.format_as_xmlimport format_as_xml
print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
'''
<user>
 <name>John</name>
 <height>6</height>
 <weight>200</weight>
</user>
'''

```

Source code in `pydantic_ai_slim/pydantic_ai/format_as_xml.py`
```
```
| ```
defformat_as_xml(
  obj: Any,
  root_tag: str = 'examples',
  item_tag: str = 'example',
  include_root_tag: bool = True,
  none_str: str = 'null',
  indent: str | None = ' ',
) -> str:
"""Format a Python object as XML.
  This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
  rather than JSON etc.
  Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
  `Iterable`, `dataclass`, and `BaseModel`.
  Args:
    obj: Python Object to serialize to XML.
    root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
    item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
      for dataclasses and Pydantic models.
    include_root_tag: Whether to include the root tag in the output
      (The root tag is always included if it includes a body - e.g. when the input is a simple value).
    none_str: String to use for `None` values.
    indent: Indentation string to use for pretty printing.
  Returns:
    XML representation of the object.
  Example:
  ```python {title="format_as_xml_example.py" lint="skip"}
  from pydantic_ai.format_as_xml import format_as_xml
  print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
  '''
  <user>
   <name>John</name>
   <height>6</height>
   <weight>200</weight>
  </user>
  '''
  ```
  """
  el = _ToXml(item_tag=item_tag, none_str=none_str).to_xml(obj, root_tag)
  if not include_root_tag and el.text is None:
    join = '' if indent is None else '\n'
    return join.join(_rootless_xml_elements(el, indent))
  else:
    if indent is not None:
      ElementTree.indent(el, space=indent)
    return ElementTree.tostring(el, encoding='unicode')

```
  
---|---

# `pydantic_ai.messages`
The structure of [`ModelMessage`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelMessage) can be shown as a graph:
###  SystemPromptPart `dataclass`
A system prompt, generally written by the application developer.
This gives the model context and guidance on how to respond.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classSystemPromptPart:
"""A system prompt, generally written by the application developer.
  This gives the model context and guidance on how to respond.
  """
  content: str
"""The content of the prompt."""
  timestamp: datetime = field(default_factory=_now_utc)
"""The timestamp of the prompt."""
  dynamic_ref: str | None = None
"""The ref of the dynamic system prompt function that generated this part.
  Only set if system prompt is dynamic, see [`system_prompt`][pydantic_ai.Agent.system_prompt] for more information.
  """
  part_kind: Literal['system-prompt'] = 'system-prompt'
"""Part type identifier, this is available on all parts as a discriminator."""
  defotel_event(self) -> Event:
    return Event('gen_ai.system.message', body={'content': self.content, 'role': 'system'})

```
  
---|---  
####  content `instance-attribute`
```
content: 

```

The content of the prompt.
####  timestamp `class-attribute` `instance-attribute`
```
timestamp:  = (default_factory=now_utc)

```

The timestamp of the prompt.
####  dynamic_ref `class-attribute` `instance-attribute`
```
dynamic_ref:  | None = None

```

The ref of the dynamic system prompt function that generated this part.
Only set if system prompt is dynamic, see [`system_prompt`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.system_prompt) for more information.
####  part_kind `class-attribute` `instance-attribute`
```
part_kind: ['system-prompt'] = 'system-prompt'

```

Part type identifier, this is available on all parts as a discriminator.
###  AudioUrl `dataclass`
A URL to an audio file.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classAudioUrl:
"""A URL to an audio file."""
  url: str
"""The URL of the audio file."""
  kind: Literal['audio-url'] = 'audio-url'
"""Type identifier, this is available on all parts as a discriminator."""
  @property
  defmedia_type(self) -> AudioMediaType:
"""Return the media type of the audio file, based on the url."""
    if self.url.endswith('.mp3'):
      return 'audio/mpeg'
    elif self.url.endswith('.wav'):
      return 'audio/wav'
    else:
      raise ValueError(f'Unknown audio file extension: {self.url}')

```
  
---|---  
####  url `instance-attribute`
```
url: 

```

The URL of the audio file.
####  kind `class-attribute` `instance-attribute`
```
kind: ['audio-url'] = 'audio-url'

```

Type identifier, this is available on all parts as a discriminator.
####  media_type `property`
```
media_type: AudioMediaType

```

Return the media type of the audio file, based on the url.
###  ImageUrl `dataclass`
A URL to an image.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classImageUrl:
"""A URL to an image."""
  url: str
"""The URL of the image."""
  kind: Literal['image-url'] = 'image-url'
"""Type identifier, this is available on all parts as a discriminator."""
  @property
  defmedia_type(self) -> ImageMediaType:
"""Return the media type of the image, based on the url."""
    if self.url.endswith(('.jpg', '.jpeg')):
      return 'image/jpeg'
    elif self.url.endswith('.png'):
      return 'image/png'
    elif self.url.endswith('.gif'):
      return 'image/gif'
    elif self.url.endswith('.webp'):
      return 'image/webp'
    else:
      raise ValueError(f'Unknown image file extension: {self.url}')
  @property
  defformat(self) -> ImageFormat:
"""The file format of the image.
    The choice of supported formats were based on the Bedrock Converse API. Other APIs don't require to use a format.
    """
    return _image_format(self.media_type)

```
  
---|---  
####  url `instance-attribute`
```
url: 

```

The URL of the image.
####  kind `class-attribute` `instance-attribute`
```
kind: ['image-url'] = 'image-url'

```

Type identifier, this is available on all parts as a discriminator.
####  media_type `property`
```
media_type: ImageMediaType

```

Return the media type of the image, based on the url.
####  format `property`
```
format: ImageFormat

```

The file format of the image.
The choice of supported formats were based on the Bedrock Converse API. Other APIs don't require to use a format.
###  DocumentUrl `dataclass`
The URL of the document.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classDocumentUrl:
"""The URL of the document."""
  url: str
"""The URL of the document."""
  kind: Literal['document-url'] = 'document-url'
"""Type identifier, this is available on all parts as a discriminator."""
  @property
  defmedia_type(self) -> str:
"""Return the media type of the document, based on the url."""
    type_, _ = guess_type(self.url)
    if type_ is None:
      raise RuntimeError(f'Unknown document file extension: {self.url}')
    return type_
  @property
  defformat(self) -> DocumentFormat:
"""The file format of the document.
    The choice of supported formats were based on the Bedrock Converse API. Other APIs don't require to use a format.
    """
    return _document_format(self.media_type)

```
  
---|---  
####  url `instance-attribute`
```
url: 

```

The URL of the document.
####  kind `class-attribute` `instance-attribute`
```
kind: ['document-url'] = 'document-url'

```

Type identifier, this is available on all parts as a discriminator.
####  media_type `property`
```
media_type: 

```

Return the media type of the document, based on the url.
####  format `property`
```
format: DocumentFormat

```

The file format of the document.
The choice of supported formats were based on the Bedrock Converse API. Other APIs don't require to use a format.
###  BinaryContent `dataclass`
Binary content, e.g. an audio or image file.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classBinaryContent:
"""Binary content, e.g. an audio or image file."""
  data: bytes
"""The binary data."""
  media_type: AudioMediaType | ImageMediaType | DocumentMediaType | str
"""The media type of the binary data."""
  kind: Literal['binary'] = 'binary'
"""Type identifier, this is available on all parts as a discriminator."""
  @property
  defis_audio(self) -> bool:
"""Return `True` if the media type is an audio type."""
    return self.media_type.startswith('audio/')
  @property
  defis_image(self) -> bool:
"""Return `True` if the media type is an image type."""
    return self.media_type.startswith('image/')
  @property
  defis_document(self) -> bool:
"""Return `True` if the media type is a document type."""
    return self.media_type in {
      'application/pdf',
      'text/plain',
      'text/csv',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/html',
      'text/markdown',
      'application/vnd.ms-excel',
    }
  @property
  defformat(self) -> str:
"""The file format of the binary content."""
    if self.is_audio:
      if self.media_type == 'audio/mpeg':
        return 'mp3'
      elif self.media_type == 'audio/wav':
        return 'wav'
    elif self.is_image:
      return _image_format(self.media_type)
    elif self.is_document:
      return _document_format(self.media_type)
    raise ValueError(f'Unknown media type: {self.media_type}')

```
  
---|---  
####  data `instance-attribute`
```
data: 

```

The binary data.
####  media_type `instance-attribute`
```
media_type: (
  AudioMediaType
  | ImageMediaType
  | DocumentMediaType
  | 
)

```

The media type of the binary data.
####  kind `class-attribute` `instance-attribute`
```
kind: ['binary'] = 'binary'

```

Type identifier, this is available on all parts as a discriminator.
####  is_audio `property`
```
is_audio: 

```

Return `True` if the media type is an audio type.
####  is_image `property`
```
is_image: 

```

Return `True` if the media type is an image type.
####  is_document `property`
```
is_document: 

```

Return `True` if the media type is a document type.
####  format `property`
```
format: 

```

The file format of the binary content.
###  UserPromptPart `dataclass`
A user prompt, generally written by the end user.
Content comes from the `user_prompt` parameter of [`Agent.run`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run), [`Agent.run_sync`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync), and [`Agent.run_stream`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream).
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classUserPromptPart:
"""A user prompt, generally written by the end user.
  Content comes from the `user_prompt` parameter of [`Agent.run`][pydantic_ai.Agent.run],
  [`Agent.run_sync`][pydantic_ai.Agent.run_sync], and [`Agent.run_stream`][pydantic_ai.Agent.run_stream].
  """
  content: str | Sequence[UserContent]
"""The content of the prompt."""
  timestamp: datetime = field(default_factory=_now_utc)
"""The timestamp of the prompt."""
  part_kind: Literal['user-prompt'] = 'user-prompt'
"""Part type identifier, this is available on all parts as a discriminator."""
  defotel_event(self) -> Event:
    if isinstance(self.content, str):
      content = self.content
    else:
      # TODO figure out what to record for images and audio
      content = [part if isinstance(part, str) else {'kind': part.kind} for part in self.content]
    return Event('gen_ai.user.message', body={'content': content, 'role': 'user'})

```
  
---|---  
####  content `instance-attribute`
```
content:  | [UserContent]

```

The content of the prompt.
####  timestamp `class-attribute` `instance-attribute`
```
timestamp:  = (default_factory=now_utc)

```

The timestamp of the prompt.
####  part_kind `class-attribute` `instance-attribute`
```
part_kind: ['user-prompt'] = 'user-prompt'

```

Part type identifier, this is available on all parts as a discriminator.
###  ToolReturnPart `dataclass`
A tool return message, this encodes the result of running a tool.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classToolReturnPart:
"""A tool return message, this encodes the result of running a tool."""
  tool_name: str
"""The name of the "tool" was called."""
  content: Any
"""The return value."""
  tool_call_id: str
"""The tool call identifier, this is used by some models including OpenAI."""
  timestamp: datetime = field(default_factory=_now_utc)
"""The timestamp, when the tool returned."""
  part_kind: Literal['tool-return'] = 'tool-return'
"""Part type identifier, this is available on all parts as a discriminator."""
  defmodel_response_str(self) -> str:
"""Return a string representation of the content for the model."""
    if isinstance(self.content, str):
      return self.content
    else:
      return tool_return_ta.dump_json(self.content).decode()
  defmodel_response_object(self) -> dict[str, Any]:
"""Return a dictionary representation of the content, wrapping non-dict types appropriately."""
    # gemini supports JSON dict return values, but no other JSON types, hence we wrap anything else in a dict
    if isinstance(self.content, dict):
      return tool_return_ta.dump_python(self.content, mode='json') # pyright: ignore[reportUnknownMemberType]
    else:
      return {'return_value': tool_return_ta.dump_python(self.content, mode='json')}
  defotel_event(self) -> Event:
    return Event(
      'gen_ai.tool.message',
      body={'content': self.content, 'role': 'tool', 'id': self.tool_call_id, 'name': self.tool_name},
    )

```
  
---|---  
####  tool_name `instance-attribute`
```
tool_name: 

```

The name of the "tool" was called.
####  content `instance-attribute`
```
content: 

```

The return value.
####  tool_call_id `instance-attribute`
```
tool_call_id: 

```

The tool call identifier, this is used by some models including OpenAI.
####  timestamp `class-attribute` `instance-attribute`
```
timestamp:  = (default_factory=now_utc)

```

The timestamp, when the tool returned.
####  part_kind `class-attribute` `instance-attribute`
```
part_kind: ['tool-return'] = 'tool-return'

```

Part type identifier, this is available on all parts as a discriminator.
####  model_response_str
```
model_response_str() -> 

```

Return a string representation of the content for the model.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defmodel_response_str(self) -> str:
"""Return a string representation of the content for the model."""
  if isinstance(self.content, str):
    return self.content
  else:
    return tool_return_ta.dump_json(self.content).decode()

```
  
---|---  
####  model_response_object
```
model_response_object() -> [, ]

```

Return a dictionary representation of the content, wrapping non-dict types appropriately.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defmodel_response_object(self) -> dict[str, Any]:
"""Return a dictionary representation of the content, wrapping non-dict types appropriately."""
  # gemini supports JSON dict return values, but no other JSON types, hence we wrap anything else in a dict
  if isinstance(self.content, dict):
    return tool_return_ta.dump_python(self.content, mode='json') # pyright: ignore[reportUnknownMemberType]
  else:
    return {'return_value': tool_return_ta.dump_python(self.content, mode='json')}

```
  
---|---  
###  RetryPromptPart `dataclass`
A message back to a model asking it to try again.
This can be sent for a number of reasons:
  * Pydantic validation of tool arguments failed, here content is derived from a Pydantic [`ValidationError`](https://docs.pydantic.dev/latest/api/pydantic_core/#pydantic_core.ValidationError)
  * a tool raised a [`ModelRetry`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.ModelRetry) exception
  * no tool was found for the tool name
  * the model returned plain text when a structured response was expected
  * Pydantic validation of a structured response failed, here content is derived from a Pydantic [`ValidationError`](https://docs.pydantic.dev/latest/api/pydantic_core/#pydantic_core.ValidationError)
  * a result validator raised a [`ModelRetry`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.ModelRetry) exception

Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classRetryPromptPart:
"""A message back to a model asking it to try again.
  This can be sent for a number of reasons:
  * Pydantic validation of tool arguments failed, here content is derived from a Pydantic
   [`ValidationError`][pydantic_core.ValidationError]
  * a tool raised a [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] exception
  * no tool was found for the tool name
  * the model returned plain text when a structured response was expected
  * Pydantic validation of a structured response failed, here content is derived from a Pydantic
   [`ValidationError`][pydantic_core.ValidationError]
  * a result validator raised a [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] exception
  """
  content: list[pydantic_core.ErrorDetails] | str
"""Details of why and how the model should retry.
  If the retry was triggered by a [`ValidationError`][pydantic_core.ValidationError], this will be a list of
  error details.
  """
  tool_name: str | None = None
"""The name of the tool that was called, if any."""
  tool_call_id: str = field(default_factory=_generate_tool_call_id)
"""The tool call identifier, this is used by some models including OpenAI.
  In case the tool call id is not provided by the model, PydanticAI will generate a random one.
  """
  timestamp: datetime = field(default_factory=_now_utc)
"""The timestamp, when the retry was triggered."""
  part_kind: Literal['retry-prompt'] = 'retry-prompt'
"""Part type identifier, this is available on all parts as a discriminator."""
  defmodel_response(self) -> str:
"""Return a string message describing why the retry is requested."""
    if isinstance(self.content, str):
      description = self.content
    else:
      json_errors = error_details_ta.dump_json(self.content, exclude={'__all__': {'ctx'}}, indent=2)
      description = f'{len(self.content)} validation errors: {json_errors.decode()}'
    return f'{description}\n\nFix the errors and try again.'
  defotel_event(self) -> Event:
    if self.tool_name is None:
      return Event('gen_ai.user.message', body={'content': self.model_response(), 'role': 'user'})
    else:
      return Event(
        'gen_ai.tool.message',
        body={
          'content': self.model_response(),
          'role': 'tool',
          'id': self.tool_call_id,
          'name': self.tool_name,
        },
      )

```
  
---|---  
####  content `instance-attribute`
```
content: [] | 

```

Details of why and how the model should retry.
If the retry was triggered by a [`ValidationError`](https://docs.pydantic.dev/latest/api/pydantic_core/#pydantic_core.ValidationError), this will be a list of error details.
####  tool_name `class-attribute` `instance-attribute`
```
tool_name:  | None = None

```

The name of the tool that was called, if any.
####  tool_call_id `class-attribute` `instance-attribute`
```
tool_call_id:  = (
  default_factory=generate_tool_call_id
)

```

The tool call identifier, this is used by some models including OpenAI.
In case the tool call id is not provided by the model, PydanticAI will generate a random one.
####  timestamp `class-attribute` `instance-attribute`
```
timestamp:  = (default_factory=now_utc)

```

The timestamp, when the retry was triggered.
####  part_kind `class-attribute` `instance-attribute`
```
part_kind: ['retry-prompt'] = 'retry-prompt'

```

Part type identifier, this is available on all parts as a discriminator.
####  model_response
```
model_response() -> 

```

Return a string message describing why the retry is requested.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defmodel_response(self) -> str:
"""Return a string message describing why the retry is requested."""
  if isinstance(self.content, str):
    description = self.content
  else:
    json_errors = error_details_ta.dump_json(self.content, exclude={'__all__': {'ctx'}}, indent=2)
    description = f'{len(self.content)} validation errors: {json_errors.decode()}'
  return f'{description}\n\nFix the errors and try again.'

```
  
---|---  
###  ModelRequestPart `module-attribute`
```
ModelRequestPart = [
  [
    ,
    ,
    ,
    ,
  ],
  ("part_kind"),
]

```

A message part sent by PydanticAI to a model.
###  ModelRequest `dataclass`
A request generated by PydanticAI and sent to a model, e.g. a message from the PydanticAI app to the model.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classModelRequest:
"""A request generated by PydanticAI and sent to a model, e.g. a message from the PydanticAI app to the model."""
  parts: list[ModelRequestPart]
"""The parts of the user message."""
  kind: Literal['request'] = 'request'
"""Message type identifier, this is available on all parts as a discriminator."""

```
  
---|---  
####  parts `instance-attribute`
```
parts: []

```

The parts of the user message.
####  kind `class-attribute` `instance-attribute`
```
kind: ['request'] = 'request'

```

Message type identifier, this is available on all parts as a discriminator.
###  TextPart `dataclass`
A plain text response from a model.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classTextPart:
"""A plain text response from a model."""
  content: str
"""The text content of the response."""
  part_kind: Literal['text'] = 'text'
"""Part type identifier, this is available on all parts as a discriminator."""
  defhas_content(self) -> bool:
"""Return `True` if the text content is non-empty."""
    return bool(self.content)

```
  
---|---  
####  content `instance-attribute`
```
content: 

```

The text content of the response.
####  part_kind `class-attribute` `instance-attribute`
```
part_kind: ['text'] = 'text'

```

Part type identifier, this is available on all parts as a discriminator.
####  has_content
```
has_content() -> 

```

Return `True` if the text content is non-empty.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defhas_content(self) -> bool:
"""Return `True` if the text content is non-empty."""
  return bool(self.content)

```
  
---|---  
###  ToolCallPart `dataclass`
A tool call from a model.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classToolCallPart:
"""A tool call from a model."""
  tool_name: str
"""The name of the tool to call."""
  args: str | dict[str, Any]
"""The arguments to pass to the tool.
  This is stored either as a JSON string or a Python dictionary depending on how data was received.
  """
  tool_call_id: str = field(default_factory=_generate_tool_call_id)
"""The tool call identifier, this is used by some models including OpenAI.
  In case the tool call id is not provided by the model, PydanticAI will generate a random one.
  """
  part_kind: Literal['tool-call'] = 'tool-call'
"""Part type identifier, this is available on all parts as a discriminator."""
  defargs_as_dict(self) -> dict[str, Any]:
"""Return the arguments as a Python dictionary.
    This is just for convenience with models that require dicts as input.
    """
    if isinstance(self.args, dict):
      return self.args
    args = pydantic_core.from_json(self.args)
    assert isinstance(args, dict), 'args should be a dict'
    return cast(dict[str, Any], args)
  defargs_as_json_str(self) -> str:
"""Return the arguments as a JSON string.
    This is just for convenience with models that require JSON strings as input.
    """
    if isinstance(self.args, str):
      return self.args
    return pydantic_core.to_json(self.args).decode()
  defhas_content(self) -> bool:
"""Return `True` if the arguments contain any data."""
    if isinstance(self.args, dict):
      # TODO: This should probably return True if you have the value False, or 0, etc.
      #  It makes sense to me to ignore empty strings, but not sure about empty lists or dicts
      return any(self.args.values())
    else:
      return bool(self.args)

```
  
---|---  
####  tool_name `instance-attribute`
```
tool_name: 

```

The name of the tool to call.
####  args `instance-attribute`
```
args:  | [, ]

```

The arguments to pass to the tool.
This is stored either as a JSON string or a Python dictionary depending on how data was received.
####  tool_call_id `class-attribute` `instance-attribute`
```
tool_call_id:  = (
  default_factory=generate_tool_call_id
)

```

The tool call identifier, this is used by some models including OpenAI.
In case the tool call id is not provided by the model, PydanticAI will generate a random one.
####  part_kind `class-attribute` `instance-attribute`
```
part_kind: ['tool-call'] = 'tool-call'

```

Part type identifier, this is available on all parts as a discriminator.
####  args_as_dict
```
args_as_dict() -> [, ]

```

Return the arguments as a Python dictionary.
This is just for convenience with models that require dicts as input.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defargs_as_dict(self) -> dict[str, Any]:
"""Return the arguments as a Python dictionary.
  This is just for convenience with models that require dicts as input.
  """
  if isinstance(self.args, dict):
    return self.args
  args = pydantic_core.from_json(self.args)
  assert isinstance(args, dict), 'args should be a dict'
  return cast(dict[str, Any], args)

```
  
---|---  
####  args_as_json_str
```
args_as_json_str() -> 

```

Return the arguments as a JSON string.
This is just for convenience with models that require JSON strings as input.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defargs_as_json_str(self) -> str:
"""Return the arguments as a JSON string.
  This is just for convenience with models that require JSON strings as input.
  """
  if isinstance(self.args, str):
    return self.args
  return pydantic_core.to_json(self.args).decode()

```
  
---|---  
####  has_content
```
has_content() -> 

```

Return `True` if the arguments contain any data.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defhas_content(self) -> bool:
"""Return `True` if the arguments contain any data."""
  if isinstance(self.args, dict):
    # TODO: This should probably return True if you have the value False, or 0, etc.
    #  It makes sense to me to ignore empty strings, but not sure about empty lists or dicts
    return any(self.args.values())
  else:
    return bool(self.args)

```
  
---|---  
###  ModelResponsePart `module-attribute`
```
ModelResponsePart = [
  [, ],
  ("part_kind"),
]

```

A message part returned by a model.
###  ModelResponse `dataclass`
A response from a model, e.g. a message from the model to the PydanticAI app.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classModelResponse:
"""A response from a model, e.g. a message from the model to the PydanticAI app."""
  parts: list[ModelResponsePart]
"""The parts of the model message."""
  model_name: str | None = None
"""The name of the model that generated the response."""
  timestamp: datetime = field(default_factory=_now_utc)
"""The timestamp of the response.
  If the model provides a timestamp in the response (as OpenAI does) that will be used.
  """
  kind: Literal['response'] = 'response'
"""Message type identifier, this is available on all parts as a discriminator."""
  defotel_events(self) -> list[Event]:
"""Return OpenTelemetry events for the response."""
    result: list[Event] = []
    defnew_event_body():
      new_body: dict[str, Any] = {'role': 'assistant'}
      ev = Event('gen_ai.assistant.message', body=new_body)
      result.append(ev)
      return new_body
    body = new_event_body()
    for part in self.parts:
      if isinstance(part, ToolCallPart):
        body.setdefault('tool_calls', []).append(
          {
            'id': part.tool_call_id,
            'type': 'function', # TODO https://github.com/pydantic/pydantic-ai/issues/888
            'function': {
              'name': part.tool_name,
              'arguments': part.args,
            },
          }
        )
      elif isinstance(part, TextPart):
        if body.get('content'):
          body = new_event_body()
        body['content'] = part.content
    return result

```
  
---|---  
####  parts `instance-attribute`
```
parts: []

```

The parts of the model message.
####  model_name `class-attribute` `instance-attribute`
```
model_name:  | None = None

```

The name of the model that generated the response.
####  timestamp `class-attribute` `instance-attribute`
```
timestamp:  = (default_factory=now_utc)

```

The timestamp of the response.
If the model provides a timestamp in the response (as OpenAI does) that will be used.
####  kind `class-attribute` `instance-attribute`
```
kind: ['response'] = 'response'

```

Message type identifier, this is available on all parts as a discriminator.
####  otel_events
```
otel_events() -> []

```

Return OpenTelemetry events for the response.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defotel_events(self) -> list[Event]:
"""Return OpenTelemetry events for the response."""
  result: list[Event] = []
  defnew_event_body():
    new_body: dict[str, Any] = {'role': 'assistant'}
    ev = Event('gen_ai.assistant.message', body=new_body)
    result.append(ev)
    return new_body
  body = new_event_body()
  for part in self.parts:
    if isinstance(part, ToolCallPart):
      body.setdefault('tool_calls', []).append(
        {
          'id': part.tool_call_id,
          'type': 'function', # TODO https://github.com/pydantic/pydantic-ai/issues/888
          'function': {
            'name': part.tool_name,
            'arguments': part.args,
          },
        }
      )
    elif isinstance(part, TextPart):
      if body.get('content'):
        body = new_event_body()
      body['content'] = part.content
  return result

```
  
---|---  
###  ModelMessage `module-attribute`
```
ModelMessage = [
  [, ],
  ("kind"),
]

```

Any message sent to or returned by a model.
###  ModelMessagesTypeAdapter `module-attribute`
```
ModelMessagesTypeAdapter = (
  [],
  config=(
    defer_build=True, ser_json_bytes="base64"
  ),
)

```

Pydantic [`TypeAdapter`](https://docs.pydantic.dev/latest/api/type_adapter/#pydantic.type_adapter.TypeAdapter) for (de)serializing messages.
###  TextPartDelta `dataclass`
A partial update (delta) for a `TextPart` to append new text content.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classTextPartDelta:
"""A partial update (delta) for a `TextPart` to append new text content."""
  content_delta: str
"""The incremental text content to add to the existing `TextPart` content."""
  part_delta_kind: Literal['text'] = 'text'
"""Part delta type identifier, used as a discriminator."""
  defapply(self, part: ModelResponsePart) -> TextPart:
"""Apply this text delta to an existing `TextPart`.
    Args:
      part: The existing model response part, which must be a `TextPart`.
    Returns:
      A new `TextPart` with updated text content.
    Raises:
      ValueError: If `part` is not a `TextPart`.
    """
    if not isinstance(part, TextPart):
      raise ValueError('Cannot apply TextPartDeltas to non-TextParts')
    return replace(part, content=part.content + self.content_delta)

```
  
---|---  
####  content_delta `instance-attribute`
```
content_delta: 

```

The incremental text content to add to the existing `TextPart` content.
####  part_delta_kind `class-attribute` `instance-attribute`
```
part_delta_kind: ['text'] = 'text'

```

Part delta type identifier, used as a discriminator.
####  apply
```
apply(part: ) -> 

```

Apply this text delta to an existing `TextPart`.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`part` |  |  The existing model response part, which must be a `TextPart`. |  _required_  
Returns:
Type | Description  
---|---  
|  A new `TextPart` with updated text content.  
Raises:
Type | Description  
---|---  
|  If `part` is not a `TextPart`.  
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defapply(self, part: ModelResponsePart) -> TextPart:
"""Apply this text delta to an existing `TextPart`.
  Args:
    part: The existing model response part, which must be a `TextPart`.
  Returns:
    A new `TextPart` with updated text content.
  Raises:
    ValueError: If `part` is not a `TextPart`.
  """
  if not isinstance(part, TextPart):
    raise ValueError('Cannot apply TextPartDeltas to non-TextParts')
  return replace(part, content=part.content + self.content_delta)

```
  
---|---  
###  ToolCallPartDelta `dataclass`
A partial update (delta) for a `ToolCallPart` to modify tool name, arguments, or tool call ID.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classToolCallPartDelta:
"""A partial update (delta) for a `ToolCallPart` to modify tool name, arguments, or tool call ID."""
  tool_name_delta: str | None = None
"""Incremental text to add to the existing tool name, if any."""
  args_delta: str | dict[str, Any] | None = None
"""Incremental data to add to the tool arguments.
  If this is a string, it will be appended to existing JSON arguments.
  If this is a dict, it will be merged with existing dict arguments.
  """
  tool_call_id: str | None = None
"""Optional tool call identifier, this is used by some models including OpenAI.
  Note this is never treated as a delta — it can replace None, but otherwise if a
  non-matching value is provided an error will be raised."""
  part_delta_kind: Literal['tool_call'] = 'tool_call'
"""Part delta type identifier, used as a discriminator."""
  defas_part(self) -> ToolCallPart | None:
"""Convert this delta to a fully formed `ToolCallPart` if possible, otherwise return `None`.
    Returns:
      A `ToolCallPart` if both `tool_name_delta` and `args_delta` are set, otherwise `None`.
    """
    if self.tool_name_delta is None or self.args_delta is None:
      return None
    return ToolCallPart(self.tool_name_delta, self.args_delta, self.tool_call_id or _generate_tool_call_id())
  @overload
  defapply(self, part: ModelResponsePart) -> ToolCallPart: ...
  @overload
  defapply(self, part: ModelResponsePart | ToolCallPartDelta) -> ToolCallPart | ToolCallPartDelta: ...
  defapply(self, part: ModelResponsePart | ToolCallPartDelta) -> ToolCallPart | ToolCallPartDelta:
"""Apply this delta to a part or delta, returning a new part or delta with the changes applied.
    Args:
      part: The existing model response part or delta to update.
    Returns:
      Either a new `ToolCallPart` or an updated `ToolCallPartDelta`.
    Raises:
      ValueError: If `part` is neither a `ToolCallPart` nor a `ToolCallPartDelta`.
      UnexpectedModelBehavior: If applying JSON deltas to dict arguments or vice versa.
    """
    if isinstance(part, ToolCallPart):
      return self._apply_to_part(part)
    if isinstance(part, ToolCallPartDelta):
      return self._apply_to_delta(part)
    raise ValueError(f'Can only apply ToolCallPartDeltas to ToolCallParts or ToolCallPartDeltas, not {part}')
  def_apply_to_delta(self, delta: ToolCallPartDelta) -> ToolCallPart | ToolCallPartDelta:
"""Internal helper to apply this delta to another delta."""
    if self.tool_name_delta:
      # Append incremental text to the existing tool_name_delta
      updated_tool_name_delta = (delta.tool_name_delta or '') + self.tool_name_delta
      delta = replace(delta, tool_name_delta=updated_tool_name_delta)
    if isinstance(self.args_delta, str):
      if isinstance(delta.args_delta, dict):
        raise UnexpectedModelBehavior(
          f'Cannot apply JSON deltas to non-JSON tool arguments ({delta=}, {self=})'
        )
      updated_args_delta = (delta.args_delta or '') + self.args_delta
      delta = replace(delta, args_delta=updated_args_delta)
    elif isinstance(self.args_delta, dict):
      if isinstance(delta.args_delta, str):
        raise UnexpectedModelBehavior(
          f'Cannot apply dict deltas to non-dict tool arguments ({delta=}, {self=})'
        )
      updated_args_delta = {**(delta.args_delta or {}), **self.args_delta}
      delta = replace(delta, args_delta=updated_args_delta)
    if self.tool_call_id:
      delta = replace(delta, tool_call_id=self.tool_call_id)
    # If we now have enough data to create a full ToolCallPart, do so
    if delta.tool_name_delta is not None and delta.args_delta is not None:
      return ToolCallPart(delta.tool_name_delta, delta.args_delta, delta.tool_call_id or _generate_tool_call_id())
    return delta
  def_apply_to_part(self, part: ToolCallPart) -> ToolCallPart:
"""Internal helper to apply this delta directly to a `ToolCallPart`."""
    if self.tool_name_delta:
      # Append incremental text to the existing tool_name
      tool_name = part.tool_name + self.tool_name_delta
      part = replace(part, tool_name=tool_name)
    if isinstance(self.args_delta, str):
      if not isinstance(part.args, str):
        raise UnexpectedModelBehavior(f'Cannot apply JSON deltas to non-JSON tool arguments ({part=}, {self=})')
      updated_json = part.args + self.args_delta
      part = replace(part, args=updated_json)
    elif isinstance(self.args_delta, dict):
      if not isinstance(part.args, dict):
        raise UnexpectedModelBehavior(f'Cannot apply dict deltas to non-dict tool arguments ({part=}, {self=})')
      updated_dict = {**(part.args or {}), **self.args_delta}
      part = replace(part, args=updated_dict)
    if self.tool_call_id:
      part = replace(part, tool_call_id=self.tool_call_id)
    return part

```
  
---|---  
####  tool_name_delta `class-attribute` `instance-attribute`
```
tool_name_delta:  | None = None

```

Incremental text to add to the existing tool name, if any.
####  args_delta `class-attribute` `instance-attribute`
```
args_delta:  | [, ] | None = None

```

Incremental data to add to the tool arguments.
If this is a string, it will be appended to existing JSON arguments. If this is a dict, it will be merged with existing dict arguments.
####  tool_call_id `class-attribute` `instance-attribute`
```
tool_call_id:  | None = None

```

Optional tool call identifier, this is used by some models including OpenAI.
Note this is never treated as a delta — it can replace None, but otherwise if a non-matching value is provided an error will be raised.
####  part_delta_kind `class-attribute` `instance-attribute`
```
part_delta_kind: ['tool_call'] = 'tool_call'

```

Part delta type identifier, used as a discriminator.
####  as_part
```
as_part() ->  | None

```

Convert this delta to a fully formed `ToolCallPart` if possible, otherwise return `None`.
Returns:
Type | Description  
---|---  
`ToolCallPart[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ToolCallPart "pydantic_ai.messages.ToolCallPart") | None` |  A `ToolCallPart` if both `tool_name_delta` and `args_delta` are set, otherwise `None`.  
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defas_part(self) -> ToolCallPart | None:
"""Convert this delta to a fully formed `ToolCallPart` if possible, otherwise return `None`.
  Returns:
    A `ToolCallPart` if both `tool_name_delta` and `args_delta` are set, otherwise `None`.
  """
  if self.tool_name_delta is None or self.args_delta is None:
    return None
  return ToolCallPart(self.tool_name_delta, self.args_delta, self.tool_call_id or _generate_tool_call_id())

```
  
---|---  
####  apply
```
apply(part: ) -> 

```

```
apply(
  part:  | ,
) ->  | 

```

```
apply(
  part:  | ,
) ->  | 

```

Apply this delta to a part or delta, returning a new part or delta with the changes applied.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`part` |  `ModelResponsePart[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelResponsePart "pydantic_ai.messages.ModelResponsePart") | ToolCallPartDelta[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ToolCallPartDelta "pydantic_ai.messages.ToolCallPartDelta")` |  The existing model response part or delta to update. |  _required_  
Returns:
Type | Description  
---|---  
`ToolCallPart[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ToolCallPart "pydantic_ai.messages.ToolCallPart") | ToolCallPartDelta[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ToolCallPartDelta "pydantic_ai.messages.ToolCallPartDelta")` |  Either a new `ToolCallPart` or an updated `ToolCallPartDelta`.  
Raises:
Type | Description  
---|---  
|  If `part` is neither a `ToolCallPart` nor a `ToolCallPartDelta`.  
|  If applying JSON deltas to dict arguments or vice versa.  
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
defapply(self, part: ModelResponsePart | ToolCallPartDelta) -> ToolCallPart | ToolCallPartDelta:
"""Apply this delta to a part or delta, returning a new part or delta with the changes applied.
  Args:
    part: The existing model response part or delta to update.
  Returns:
    Either a new `ToolCallPart` or an updated `ToolCallPartDelta`.
  Raises:
    ValueError: If `part` is neither a `ToolCallPart` nor a `ToolCallPartDelta`.
    UnexpectedModelBehavior: If applying JSON deltas to dict arguments or vice versa.
  """
  if isinstance(part, ToolCallPart):
    return self._apply_to_part(part)
  if isinstance(part, ToolCallPartDelta):
    return self._apply_to_delta(part)
  raise ValueError(f'Can only apply ToolCallPartDeltas to ToolCallParts or ToolCallPartDeltas, not {part}')

```
  
---|---  
###  ModelResponsePartDelta `module-attribute`
```
ModelResponsePartDelta = [
  [, ],
  ("part_delta_kind"),
]

```

A partial update (delta) for any model response part.
###  PartStartEvent `dataclass`
An event indicating that a new part has started.
If multiple `PartStartEvent`s are received with the same index, the new one should fully replace the old one.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classPartStartEvent:
"""An event indicating that a new part has started.
  If multiple `PartStartEvent`s are received with the same index,
  the new one should fully replace the old one.
  """
  index: int
"""The index of the part within the overall response parts list."""
  part: ModelResponsePart
"""The newly started `ModelResponsePart`."""
  event_kind: Literal['part_start'] = 'part_start'
"""Event type identifier, used as a discriminator."""

```
  
---|---  
####  index `instance-attribute`
```
index: 

```

The index of the part within the overall response parts list.
####  part `instance-attribute`
```
part: 

```

The newly started `ModelResponsePart`.
####  event_kind `class-attribute` `instance-attribute`
```
event_kind: ['part_start'] = 'part_start'

```

Event type identifier, used as a discriminator.
###  PartDeltaEvent `dataclass`
An event indicating a delta update for an existing part.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classPartDeltaEvent:
"""An event indicating a delta update for an existing part."""
  index: int
"""The index of the part within the overall response parts list."""
  delta: ModelResponsePartDelta
"""The delta to apply to the specified part."""
  event_kind: Literal['part_delta'] = 'part_delta'
"""Event type identifier, used as a discriminator."""

```
  
---|---  
####  index `instance-attribute`
```
index: 

```

The index of the part within the overall response parts list.
####  delta `instance-attribute`
```
delta: 

```

The delta to apply to the specified part.
####  event_kind `class-attribute` `instance-attribute`
```
event_kind: ['part_delta'] = 'part_delta'

```

Event type identifier, used as a discriminator.
###  FinalResultEvent `dataclass`
An event indicating the response to the current model request matches the result schema.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classFinalResultEvent:
"""An event indicating the response to the current model request matches the result schema."""
  tool_name: str | None
"""The name of the result tool that was called. `None` if the result is from text content and not from a tool."""
  tool_call_id: str | None
"""The tool call ID, if any, that this result is associated with."""
  event_kind: Literal['final_result'] = 'final_result'
"""Event type identifier, used as a discriminator."""

```
  
---|---  
####  tool_name `instance-attribute`
```
tool_name:  | None

```

The name of the result tool that was called. `None` if the result is from text content and not from a tool.
####  tool_call_id `instance-attribute`
```
tool_call_id:  | None

```

The tool call ID, if any, that this result is associated with.
####  event_kind `class-attribute` `instance-attribute`
```
event_kind: ['final_result'] = 'final_result'

```

Event type identifier, used as a discriminator.
###  ModelResponseStreamEvent `module-attribute`
```
ModelResponseStreamEvent = [
  [, ],
  ("event_kind"),
]

```

An event in the model response stream, either starting a new part or applying a delta to an existing one.
###  AgentStreamEvent `module-attribute`
```
AgentStreamEvent = [
  [, , ],
  ("event_kind"),
]

```

An event in the agent stream.
###  FunctionToolCallEvent `dataclass`
An event indicating the start to a call to a function tool.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classFunctionToolCallEvent:
"""An event indicating the start to a call to a function tool."""
  part: ToolCallPart
"""The (function) tool call to make."""
  call_id: str = field(init=False)
"""An ID used for matching details about the call to its result. If present, defaults to the part's tool_call_id."""
  event_kind: Literal['function_tool_call'] = 'function_tool_call'
"""Event type identifier, used as a discriminator."""
  def__post_init__(self):
    self.call_id = self.part.tool_call_id or str(uuid.uuid4())

```
  
---|---  
####  part `instance-attribute`
```
part: 

```

The (function) tool call to make.
####  call_id `class-attribute` `instance-attribute`
```
call_id:  = (=False)

```

An ID used for matching details about the call to its result. If present, defaults to the part's tool_call_id.
####  event_kind `class-attribute` `instance-attribute`
```
event_kind: ["function_tool_call"] = (
  "function_tool_call"
)

```

Event type identifier, used as a discriminator.
###  FunctionToolResultEvent `dataclass`
An event indicating the result of a function tool call.
Source code in `pydantic_ai_slim/pydantic_ai/messages.py`
```
```
| ```
@dataclass
classFunctionToolResultEvent:
"""An event indicating the result of a function tool call."""
  result: ToolReturnPart | RetryPromptPart
"""The result of the call to the function tool."""
  tool_call_id: str
"""An ID used to match the result to its original call."""
  event_kind: Literal['function_tool_result'] = 'function_tool_result'
"""Event type identifier, used as a discriminator."""

```
  
---|---  
####  result `instance-attribute`
```
result:  | 

```

The result of the call to the function tool.
####  tool_call_id `instance-attribute`
```
tool_call_id: 

```

An ID used to match the result to its original call.
####  event_kind `class-attribute` `instance-attribute`
```
event_kind: ["function_tool_result"] = (
  "function_tool_result"
)

```

Event type identifier, used as a discriminator.

# `pydantic_ai.models.anthropic`
## Setup
For details on how to set up authentication with this model, see [model configuration for Anthropic](https://ai.pydantic.dev/models/#anthropic).
###  LatestAnthropicModelNames `module-attribute`
```
LatestAnthropicModelNames = [
  "claude-3-7-sonnet-latest",
  "claude-3-5-haiku-latest",
  "claude-3-5-sonnet-latest",
  "claude-3-opus-latest",
]

```

Latest Anthropic models.
###  AnthropicModelName `module-attribute`
```
AnthropicModelName = [, ]

```

Possible Anthropic model names.
Since Anthropic supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the Anthropic docs](https://docs.anthropic.com/en/docs/about-claude/models) for a full list.
###  AnthropicModelSettings
Bases: 
Settings used for an Anthropic model request.
Source code in `pydantic_ai_slim/pydantic_ai/models/anthropic.py`
```
```
| ```
classAnthropicModelSettings(ModelSettings):
"""Settings used for an Anthropic model request."""
  anthropic_metadata: MetadataParam
"""An object describing metadata about the request.
  Contains `user_id`, an external identifier for the user who is associated with the request."""

```
  
---|---  
####  anthropic_metadata `instance-attribute`
```
anthropic_metadata: MetadataParam

```

An object describing metadata about the request.
Contains `user_id`, an external identifier for the user who is associated with the request.
###  AnthropicModel `dataclass`
Bases: 
A model that uses the Anthropic API.
Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.
Apart from `__init__`, all methods are private or match those of the base class.
Note
The `AnthropicModel` class does not yet support streaming responses. We anticipate adding support for streaming responses in a near-term future release.
Source code in `pydantic_ai_slim/pydantic_ai/models/anthropic.py`
```
```
| ```
@dataclass(init=False)
classAnthropicModel(Model):
"""A model that uses the Anthropic API.
  Internally, this uses the [Anthropic Python client](https://github.com/anthropics/anthropic-sdk-python) to interact with the API.
  Apart from `__init__`, all methods are private or match those of the base class.
  !!! note
    The `AnthropicModel` class does not yet support streaming responses.
    We anticipate adding support for streaming responses in a near-term future release.
  """
  client: AsyncAnthropic = field(repr=False)
  _model_name: AnthropicModelName = field(repr=False)
  _system: str = field(default='anthropic', repr=False)
  def__init__(
    self,
    model_name: AnthropicModelName,
    *,
    provider: Literal['anthropic'] | Provider[AsyncAnthropic] = 'anthropic',
  ):
"""Initialize an Anthropic model.
    Args:
      model_name: The name of the Anthropic model to use. List of model names available
        [here](https://docs.anthropic.com/en/docs/about-claude/models).
      provider: The provider to use for the Anthropic API. Can be either the string 'anthropic' or an
        instance of `Provider[AsyncAnthropic]`. If not provided, the other parameters will be used.
    """
    self._model_name = model_name
    if isinstance(provider, str):
      provider = infer_provider(provider)
    self.client = provider.client
  @property
  defbase_url(self) -> str:
    return str(self.client.base_url)
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, usage.Usage]:
    check_allow_model_requests()
    response = await self._messages_create(
      messages, False, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
    )
    return self._process_response(response), _map_usage(response)
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
    check_allow_model_requests()
    response = await self._messages_create(
      messages, True, cast(AnthropicModelSettings, model_settings or {}), model_request_parameters
    )
    async with response:
      yield await self._process_streamed_response(response)
  @property
  defmodel_name(self) -> AnthropicModelName:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider."""
    return self._system
  @overload
  async def_messages_create(
    self,
    messages: list[ModelMessage],
    stream: Literal[True],
    model_settings: AnthropicModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncStream[RawMessageStreamEvent]:
    pass
  @overload
  async def_messages_create(
    self,
    messages: list[ModelMessage],
    stream: Literal[False],
    model_settings: AnthropicModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> AnthropicMessage:
    pass
  async def_messages_create(
    self,
    messages: list[ModelMessage],
    stream: bool,
    model_settings: AnthropicModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> AnthropicMessage | AsyncStream[RawMessageStreamEvent]:
    # standalone function to make it easier to override
    tools = self._get_tools(model_request_parameters)
    tool_choice: ToolChoiceParam | None
    if not tools:
      tool_choice = None
    else:
      if not model_request_parameters.allow_text_result:
        tool_choice = {'type': 'any'}
      else:
        tool_choice = {'type': 'auto'}
      if (allow_parallel_tool_calls := model_settings.get('parallel_tool_calls')) is not None:
        tool_choice['disable_parallel_tool_use'] = not allow_parallel_tool_calls
    system_prompt, anthropic_messages = await self._map_message(messages)
    try:
      return await self.client.messages.create(
        max_tokens=model_settings.get('max_tokens', 1024),
        system=system_prompt or NOT_GIVEN,
        messages=anthropic_messages,
        model=self._model_name,
        tools=tools or NOT_GIVEN,
        tool_choice=tool_choice or NOT_GIVEN,
        stream=stream,
        temperature=model_settings.get('temperature', NOT_GIVEN),
        top_p=model_settings.get('top_p', NOT_GIVEN),
        timeout=model_settings.get('timeout', NOT_GIVEN),
        metadata=model_settings.get('anthropic_metadata', NOT_GIVEN),
      )
    except APIStatusError as e:
      if (status_code := e.status_code) >= 400:
        raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) frome
      raise
  def_process_response(self, response: AnthropicMessage) -> ModelResponse:
"""Process a non-streamed response, and prepare a message to return."""
    items: list[ModelResponsePart] = []
    for item in response.content:
      if isinstance(item, TextBlock):
        items.append(TextPart(content=item.text))
      else:
        assert isinstance(item, ToolUseBlock), 'unexpected item type'
        items.append(
          ToolCallPart(
            tool_name=item.name,
            args=cast(dict[str, Any], item.input),
            tool_call_id=item.id,
          )
        )
    return ModelResponse(items, model_name=response.model)
  async def_process_streamed_response(self, response: AsyncStream[RawMessageStreamEvent]) -> StreamedResponse:
    peekable_response = _utils.PeekableAsyncStream(response)
    first_chunk = await peekable_response.peek()
    if isinstance(first_chunk, _utils.Unset):
      raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')
    # Since Anthropic doesn't provide a timestamp in the message, we'll use the current time
    timestamp = datetime.now(tz=timezone.utc)
    return AnthropicStreamedResponse(
      _model_name=self._model_name, _response=peekable_response, _timestamp=timestamp
    )
  def_get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolParam]:
    tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
    if model_request_parameters.result_tools:
      tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
    return tools
  async def_map_message(self, messages: list[ModelMessage]) -> tuple[str, list[MessageParam]]:
"""Just maps a `pydantic_ai.Message` to a `anthropic.types.MessageParam`."""
    system_prompt: str = ''
    anthropic_messages: list[MessageParam] = []
    for m in messages:
      if isinstance(m, ModelRequest):
        user_content_params: list[
          ToolResultBlockParam | TextBlockParam | ImageBlockParam | DocumentBlockParam
        ] = []
        for request_part in m.parts:
          if isinstance(request_part, SystemPromptPart):
            system_prompt += request_part.content
          elif isinstance(request_part, UserPromptPart):
            async for content in self._map_user_prompt(request_part):
              user_content_params.append(content)
          elif isinstance(request_part, ToolReturnPart):
            tool_result_block_param = ToolResultBlockParam(
              tool_use_id=_guard_tool_call_id(t=request_part),
              type='tool_result',
              content=request_part.model_response_str(),
              is_error=False,
            )
            user_content_params.append(tool_result_block_param)
          elif isinstance(request_part, RetryPromptPart):
            if request_part.tool_name is None:
              retry_param = TextBlockParam(type='text', text=request_part.model_response())
            else:
              retry_param = ToolResultBlockParam(
                tool_use_id=_guard_tool_call_id(t=request_part),
                type='tool_result',
                content=request_part.model_response(),
                is_error=True,
              )
            user_content_params.append(retry_param)
        anthropic_messages.append(MessageParam(role='user', content=user_content_params))
      elif isinstance(m, ModelResponse):
        assistant_content_params: list[TextBlockParam | ToolUseBlockParam] = []
        for response_part in m.parts:
          if isinstance(response_part, TextPart):
            assistant_content_params.append(TextBlockParam(text=response_part.content, type='text'))
          else:
            tool_use_block_param = ToolUseBlockParam(
              id=_guard_tool_call_id(t=response_part),
              type='tool_use',
              name=response_part.tool_name,
              input=response_part.args_as_dict(),
            )
            assistant_content_params.append(tool_use_block_param)
        anthropic_messages.append(MessageParam(role='assistant', content=assistant_content_params))
      else:
        assert_never(m)
    return system_prompt, anthropic_messages
  @staticmethod
  async def_map_user_prompt(
    part: UserPromptPart,
  ) -> AsyncGenerator[ImageBlockParam | TextBlockParam | DocumentBlockParam]:
    if isinstance(part.content, str):
      yield TextBlockParam(text=part.content, type='text')
    else:
      for item in part.content:
        if isinstance(item, str):
          yield TextBlockParam(text=item, type='text')
        elif isinstance(item, BinaryContent):
          if item.is_image:
            yield ImageBlockParam(
              source={'data': io.BytesIO(item.data), 'media_type': item.media_type, 'type': 'base64'}, # type: ignore
              type='image',
            )
          elif item.media_type == 'application/pdf':
            yield DocumentBlockParam(
              source=Base64PDFSourceParam(
                data=io.BytesIO(item.data),
                media_type='application/pdf',
                type='base64',
              ),
              type='document',
            )
          else:
            raise RuntimeError('Only images and PDFs are supported for binary content')
        elif isinstance(item, ImageUrl):
          try:
            response = await cached_async_http_client().get(item.url)
            response.raise_for_status()
            yield ImageBlockParam(
              source={
                'data': io.BytesIO(response.content),
                'media_type': item.media_type,
                'type': 'base64',
              },
              type='image',
            )
          except ValueError:
            # Download the file if can't find the mime type.
            client = cached_async_http_client()
            response = await client.get(item.url, follow_redirects=True)
            response.raise_for_status()
            base64_encoded = base64.b64encode(response.content).decode('utf-8')
            if (mime_type := response.headers['Content-Type']) in (
              'image/jpeg',
              'image/png',
              'image/gif',
              'image/webp',
            ):
              yield ImageBlockParam(
                source={'data': base64_encoded, 'media_type': mime_type, 'type': 'base64'},
                type='image',
              )
            else: # pragma: no cover
              raise RuntimeError(f'Unsupported image type: {mime_type}')
        elif isinstance(item, DocumentUrl):
          response = await cached_async_http_client().get(item.url)
          response.raise_for_status()
          if item.media_type == 'application/pdf':
            yield DocumentBlockParam(
              source=Base64PDFSourceParam(
                data=io.BytesIO(response.content),
                media_type=item.media_type,
                type='base64',
              ),
              type='document',
            )
          elif item.media_type == 'text/plain':
            yield DocumentBlockParam(
              source=PlainTextSourceParam(data=response.text, media_type=item.media_type, type='text'),
              type='document',
            )
          else: # pragma: no cover
            raise RuntimeError(f'Unsupported media type: {item.media_type}')
        else:
          raise RuntimeError(f'Unsupported content type: {type(item)}')
  @staticmethod
  def_map_tool_definition(f: ToolDefinition) -> ToolParam:
    return {
      'name': f.name,
      'description': f.description,
      'input_schema': f.parameters_json_schema,
    }

```
  
---|---  
####  __init__
```
__init__(
  model_name: ,
  *,
  provider: (
    ["anthropic"] | [AsyncAnthropic]
  ) = "anthropic"
)

```

Initialize an Anthropic model.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`model_name` |  |  The name of the Anthropic model to use. List of model names available [here](https://docs.anthropic.com/en/docs/about-claude/models). |  _required_  
`provider` |  `Literal[](https://docs.python.org/3/library/typing.html#typing.Literal "typing.Literal")['anthropic'] | Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncAnthropic]` |  The provider to use for the Anthropic API. Can be either the string 'anthropic' or an instance of `Provider[AsyncAnthropic]`. If not provided, the other parameters will be used. |  `'anthropic'`  
Source code in `pydantic_ai_slim/pydantic_ai/models/anthropic.py`
```
```
| ```
def__init__(
  self,
  model_name: AnthropicModelName,
  *,
  provider: Literal['anthropic'] | Provider[AsyncAnthropic] = 'anthropic',
):
"""Initialize an Anthropic model.
  Args:
    model_name: The name of the Anthropic model to use. List of model names available
      [here](https://docs.anthropic.com/en/docs/about-claude/models).
    provider: The provider to use for the Anthropic API. Can be either the string 'anthropic' or an
      instance of `Provider[AsyncAnthropic]`. If not provided, the other parameters will be used.
  """
  self._model_name = model_name
  if isinstance(provider, str):
    provider = infer_provider(provider)
  self.client = provider.client

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider.
###  AnthropicStreamedResponse `dataclass`
Bases: 
Implementation of `StreamedResponse` for Anthropic models.
Source code in `pydantic_ai_slim/pydantic_ai/models/anthropic.py`
```
```
| ```
@dataclass
classAnthropicStreamedResponse(StreamedResponse):
"""Implementation of `StreamedResponse` for Anthropic models."""
  _model_name: AnthropicModelName
  _response: AsyncIterable[RawMessageStreamEvent]
  _timestamp: datetime
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    current_block: ContentBlock | None = None
    current_json: str = ''
    async for event in self._response:
      self._usage += _map_usage(event)
      if isinstance(event, RawContentBlockStartEvent):
        current_block = event.content_block
        if isinstance(current_block, TextBlock) and current_block.text:
          yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=current_block.text)
        elif isinstance(current_block, ToolUseBlock):
          maybe_event = self._parts_manager.handle_tool_call_delta(
            vendor_part_id=current_block.id,
            tool_name=current_block.name,
            args=cast(dict[str, Any], current_block.input),
            tool_call_id=current_block.id,
          )
          if maybe_event is not None:
            yield maybe_event
      elif isinstance(event, RawContentBlockDeltaEvent):
        if isinstance(event.delta, TextDelta):
          yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=event.delta.text)
        elif (
          current_block and event.delta.type == 'input_json_delta' and isinstance(current_block, ToolUseBlock)
        ):
          # Try to parse the JSON immediately, otherwise cache the value for later. This handles
          # cases where the JSON is not currently valid but will be valid once we stream more tokens.
          try:
            parsed_args = json_loads(current_json + event.delta.partial_json)
            current_json = ''
          except JSONDecodeError:
            current_json += event.delta.partial_json
            continue
          # For tool calls, we need to handle partial JSON updates
          maybe_event = self._parts_manager.handle_tool_call_delta(
            vendor_part_id=current_block.id,
            tool_name='',
            args=parsed_args,
            tool_call_id=current_block.id,
          )
          if maybe_event is not None:
            yield maybe_event
      elif isinstance(event, (RawContentBlockStopEvent, RawMessageStopEvent)):
        current_block = None
  @property
  defmodel_name(self) -> AnthropicModelName:
"""Get the model name of the response."""
    return self._model_name
  @property
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    return self._timestamp

```
  
---|---  
####  model_name `property`
```
model_name: 

```

Get the model name of the response.
####  timestamp `property`
```
timestamp: 

```

Get the timestamp of the response.

# `pydantic_ai.models`
Logic related to making requests to an LLM.
The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the specific LLM being used.
###  KnownModelName `module-attribute`
```
KnownModelName = [
  "anthropic:claude-3-7-sonnet-latest",
  "anthropic:claude-3-5-haiku-latest",
  "anthropic:claude-3-5-sonnet-latest",
  "anthropic:claude-3-opus-latest",
  "claude-3-7-sonnet-latest",
  "claude-3-5-haiku-latest",
  "bedrock:amazon.titan-tg1-large",
  "bedrock:amazon.titan-text-lite-v1",
  "bedrock:amazon.titan-text-express-v1",
  "bedrock:us.amazon.nova-pro-v1:0",
  "bedrock:us.amazon.nova-lite-v1:0",
  "bedrock:us.amazon.nova-micro-v1:0",
  "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0",
  "bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0",
  "bedrock:anthropic.claude-3-5-haiku-20241022-v1:0",
  "bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0",
  "bedrock:anthropic.claude-instant-v1",
  "bedrock:anthropic.claude-v2:1",
  "bedrock:anthropic.claude-v2",
  "bedrock:anthropic.claude-3-sonnet-20240229-v1:0",
  "bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0",
  "bedrock:anthropic.claude-3-haiku-20240307-v1:0",
  "bedrock:us.anthropic.claude-3-haiku-20240307-v1:0",
  "bedrock:anthropic.claude-3-opus-20240229-v1:0",
  "bedrock:us.anthropic.claude-3-opus-20240229-v1:0",
  "bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0",
  "bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0",
  "bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0",
  "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0",
  "bedrock:cohere.command-text-v14",
  "bedrock:cohere.command-r-v1:0",
  "bedrock:cohere.command-r-plus-v1:0",
  "bedrock:cohere.command-light-text-v14",
  "bedrock:meta.llama3-8b-instruct-v1:0",
  "bedrock:meta.llama3-70b-instruct-v1:0",
  "bedrock:meta.llama3-1-8b-instruct-v1:0",
  "bedrock:us.meta.llama3-1-8b-instruct-v1:0",
  "bedrock:meta.llama3-1-70b-instruct-v1:0",
  "bedrock:us.meta.llama3-1-70b-instruct-v1:0",
  "bedrock:meta.llama3-1-405b-instruct-v1:0",
  "bedrock:us.meta.llama3-2-11b-instruct-v1:0",
  "bedrock:us.meta.llama3-2-90b-instruct-v1:0",
  "bedrock:us.meta.llama3-2-1b-instruct-v1:0",
  "bedrock:us.meta.llama3-2-3b-instruct-v1:0",
  "bedrock:us.meta.llama3-3-70b-instruct-v1:0",
  "bedrock:mistral.mistral-7b-instruct-v0:2",
  "bedrock:mistral.mixtral-8x7b-instruct-v0:1",
  "bedrock:mistral.mistral-large-2402-v1:0",
  "bedrock:mistral.mistral-large-2407-v1:0",
  "claude-3-5-sonnet-latest",
  "claude-3-opus-latest",
  "cohere:c4ai-aya-expanse-32b",
  "cohere:c4ai-aya-expanse-8b",
  "cohere:command",
  "cohere:command-light",
  "cohere:command-light-nightly",
  "cohere:command-nightly",
  "cohere:command-r",
  "cohere:command-r-03-2024",
  "cohere:command-r-08-2024",
  "cohere:command-r-plus",
  "cohere:command-r-plus-04-2024",
  "cohere:command-r-plus-08-2024",
  "cohere:command-r7b-12-2024",
  "deepseek:deepseek-chat",
  "deepseek:deepseek-reasoner",
  "google-gla:gemini-1.0-pro",
  "google-gla:gemini-1.5-flash",
  "google-gla:gemini-1.5-flash-8b",
  "google-gla:gemini-1.5-pro",
  "google-gla:gemini-2.0-flash-exp",
  "google-gla:gemini-2.0-flash-thinking-exp-01-21",
  "google-gla:gemini-exp-1206",
  "google-gla:gemini-2.0-flash",
  "google-gla:gemini-2.0-flash-lite-preview-02-05",
  "google-gla:gemini-2.0-pro-exp-02-05",
  "google-vertex:gemini-1.0-pro",
  "google-vertex:gemini-1.5-flash",
  "google-vertex:gemini-1.5-flash-8b",
  "google-vertex:gemini-1.5-pro",
  "google-vertex:gemini-2.0-flash-exp",
  "google-vertex:gemini-2.0-flash-thinking-exp-01-21",
  "google-vertex:gemini-exp-1206",
  "google-vertex:gemini-2.0-flash",
  "google-vertex:gemini-2.0-flash-lite-preview-02-05",
  "google-vertex:gemini-2.0-pro-exp-02-05",
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-0125",
  "gpt-3.5-turbo-0301",
  "gpt-3.5-turbo-0613",
  "gpt-3.5-turbo-1106",
  "gpt-3.5-turbo-16k",
  "gpt-3.5-turbo-16k-0613",
  "gpt-4",
  "gpt-4-0125-preview",
  "gpt-4-0314",
  "gpt-4-0613",
  "gpt-4-1106-preview",
  "gpt-4-32k",
  "gpt-4-32k-0314",
  "gpt-4-32k-0613",
  "gpt-4-turbo",
  "gpt-4-turbo-2024-04-09",
  "gpt-4-turbo-preview",
  "gpt-4-vision-preview",
  "gpt-4o",
  "gpt-4o-2024-05-13",
  "gpt-4o-2024-08-06",
  "gpt-4o-2024-11-20",
  "gpt-4o-audio-preview",
  "gpt-4o-audio-preview-2024-10-01",
  "gpt-4o-audio-preview-2024-12-17",
  "gpt-4o-mini",
  "gpt-4o-mini-2024-07-18",
  "gpt-4o-mini-audio-preview",
  "gpt-4o-mini-audio-preview-2024-12-17",
  "gpt-4o-mini-search-preview",
  "gpt-4o-mini-search-preview-2025-03-11",
  "gpt-4o-search-preview",
  "gpt-4o-search-preview-2025-03-11",
  "groq:gemma2-9b-it",
  "groq:llama-3.1-8b-instant",
  "groq:llama-3.2-11b-vision-preview",
  "groq:llama-3.2-1b-preview",
  "groq:llama-3.2-3b-preview",
  "groq:llama-3.2-90b-vision-preview",
  "groq:llama-3.3-70b-specdec",
  "groq:llama-3.3-70b-versatile",
  "groq:llama3-70b-8192",
  "groq:llama3-8b-8192",
  "groq:mixtral-8x7b-32768",
  "mistral:codestral-latest",
  "mistral:mistral-large-latest",
  "mistral:mistral-moderation-latest",
  "mistral:mistral-small-latest",
  "o1",
  "o1-2024-12-17",
  "o1-mini",
  "o1-mini-2024-09-12",
  "o1-preview",
  "o1-preview-2024-09-12",
  "o3-mini",
  "o3-mini-2025-01-31",
  "openai:chatgpt-4o-latest",
  "openai:gpt-3.5-turbo",
  "openai:gpt-3.5-turbo-0125",
  "openai:gpt-3.5-turbo-0301",
  "openai:gpt-3.5-turbo-0613",
  "openai:gpt-3.5-turbo-1106",
  "openai:gpt-3.5-turbo-16k",
  "openai:gpt-3.5-turbo-16k-0613",
  "openai:gpt-4",
  "openai:gpt-4-0125-preview",
  "openai:gpt-4-0314",
  "openai:gpt-4-0613",
  "openai:gpt-4-1106-preview",
  "openai:gpt-4-32k",
  "openai:gpt-4-32k-0314",
  "openai:gpt-4-32k-0613",
  "openai:gpt-4-turbo",
  "openai:gpt-4-turbo-2024-04-09",
  "openai:gpt-4-turbo-preview",
  "openai:gpt-4-vision-preview",
  "openai:gpt-4o",
  "openai:gpt-4o-2024-05-13",
  "openai:gpt-4o-2024-08-06",
  "openai:gpt-4o-2024-11-20",
  "openai:gpt-4o-audio-preview",
  "openai:gpt-4o-audio-preview-2024-10-01",
  "openai:gpt-4o-audio-preview-2024-12-17",
  "openai:gpt-4o-mini",
  "openai:gpt-4o-mini-2024-07-18",
  "openai:gpt-4o-mini-audio-preview",
  "openai:gpt-4o-mini-audio-preview-2024-12-17",
  "openai:gpt-4o-mini-search-preview",
  "openai:gpt-4o-mini-search-preview-2025-03-11",
  "openai:gpt-4o-search-preview",
  "openai:gpt-4o-search-preview-2025-03-11",
  "openai:o1",
  "openai:o1-2024-12-17",
  "openai:o1-mini",
  "openai:o1-mini-2024-09-12",
  "openai:o1-preview",
  "openai:o1-preview-2024-09-12",
  "openai:o3-mini",
  "openai:o3-mini-2025-01-31",
  "test",
]

```

Known model names that can be used with the `model` parameter of [`Agent`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent).
`KnownModelName` is provided as a concise way to specify a model.
###  ModelRequestParameters `dataclass`
Configuration for an agent's request to a model, specifically related to tools and result handling.
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
@dataclass
classModelRequestParameters:
"""Configuration for an agent's request to a model, specifically related to tools and result handling."""
  function_tools: list[ToolDefinition]
  allow_text_result: bool
  result_tools: list[ToolDefinition]

```
  
---|---  
###  Model
Bases: 
Abstract class for a model.
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
classModel(ABC):
"""Abstract class for a model."""
  @abstractmethod
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, Usage]:
"""Make a request to the model."""
    raise NotImplementedError()
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
"""Make a request to the model and return a streaming response."""
    # This method is not required, but you need to implement it if you want to support streamed responses
    raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
    # yield is required to make this a generator for type checking
    # noinspection PyUnreachableCode
    yield # pragma: no cover
  @property
  @abstractmethod
  defmodel_name(self) -> str:
"""The model name."""
    raise NotImplementedError()
  @property
  @abstractmethod
  defsystem(self) -> str:
"""The system / model provider, ex: openai.
    Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute,
    so should use well-known values listed in
    https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system
    when applicable.
    """
    raise NotImplementedError()
  @property
  defbase_url(self) -> str | None:
"""The base URL for the provider API, if available."""
    return None

```
  
---|---  
####  request `abstractmethod` `async`
```
request(
  messages: [],
  model_settings:  | None,
  model_request_parameters: ,
) -> [, ]

```

Make a request to the model.
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
@abstractmethod
async defrequest(
  self,
  messages: list[ModelMessage],
  model_settings: ModelSettings | None,
  model_request_parameters: ModelRequestParameters,
) -> tuple[ModelResponse, Usage]:
"""Make a request to the model."""
  raise NotImplementedError()

```
  
---|---  
####  request_stream `async`
```
request_stream(
  messages: [],
  model_settings:  | None,
  model_request_parameters: ,
) -> []

```

Make a request to the model and return a streaming response.
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
@asynccontextmanager
async defrequest_stream(
  self,
  messages: list[ModelMessage],
  model_settings: ModelSettings | None,
  model_request_parameters: ModelRequestParameters,
) -> AsyncIterator[StreamedResponse]:
"""Make a request to the model and return a streaming response."""
  # This method is not required, but you need to implement it if you want to support streamed responses
  raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
  # yield is required to make this a generator for type checking
  # noinspection PyUnreachableCode
  yield # pragma: no cover

```
  
---|---  
####  model_name `abstractmethod` `property`
```
model_name: 

```

The model name.
####  system `abstractmethod` `property`
```
system: 

```

The system / model provider, ex: openai.
Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute, so should use well-known values listed in https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system when applicable.
####  base_url `property`
```
base_url:  | None

```

The base URL for the provider API, if available.
###  StreamedResponse `dataclass`
Bases: 
Streamed response from an LLM when calling a tool.
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
@dataclass
classStreamedResponse(ABC):
"""Streamed response from an LLM when calling a tool."""
  _parts_manager: ModelResponsePartsManager = field(default_factory=ModelResponsePartsManager, init=False)
  _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
  _usage: Usage = field(default_factory=Usage, init=False)
  def__aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
"""Stream the response as an async iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
    if self._event_iterator is None:
      self._event_iterator = self._get_event_iterator()
    return self._event_iterator
  @abstractmethod
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
"""Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.
    This method should be implemented by subclasses to translate the vendor-specific stream of events into
    pydantic_ai-format events.
    It should use the `_parts_manager` to handle deltas, and should update the `_usage` attributes as it goes.
    """
    raise NotImplementedError()
    # noinspection PyUnreachableCode
    yield
  defget(self) -> ModelResponse:
"""Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
    return ModelResponse(
      parts=self._parts_manager.get_parts(), model_name=self.model_name, timestamp=self.timestamp
    )
  defusage(self) -> Usage:
"""Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
    return self._usage
  @property
  @abstractmethod
  defmodel_name(self) -> str:
"""Get the model name of the response."""
    raise NotImplementedError()
  @property
  @abstractmethod
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    raise NotImplementedError()

```
  
---|---  
####  __aiter__
```
__aiter__() -> []

```

Stream the response as an async iterable of [`ModelResponseStreamEvent`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelResponseStreamEvent)s.
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
def__aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
"""Stream the response as an async iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
  if self._event_iterator is None:
    self._event_iterator = self._get_event_iterator()
  return self._event_iterator

```
  
---|---  
####  get
```
get() -> 

```

Build a [`ModelResponse`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelResponse) from the data received from the stream so far.
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
defget(self) -> ModelResponse:
"""Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
  return ModelResponse(
    parts=self._parts_manager.get_parts(), model_name=self.model_name, timestamp=self.timestamp
  )

```
  
---|---  
####  usage
```
usage() -> 

```

Get the usage of the response so far. This will not be the final usage until the stream is exhausted.
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
defusage(self) -> Usage:
"""Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
  return self._usage

```
  
---|---  
####  model_name `abstractmethod` `property`
```
model_name: 

```

Get the model name of the response.
####  timestamp `abstractmethod` `property`
```
timestamp: 

```

Get the timestamp of the response.
###  ALLOW_MODEL_REQUESTS `module-attribute`
```
ALLOW_MODEL_REQUESTS = True

```

Whether to allow requests to models.
This global setting allows you to disable request to most models, e.g. to make sure you don't accidentally make costly requests to a model during tests.
The testing models [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel) and [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) are no affected by this setting.
###  check_allow_model_requests
```
check_allow_model_requests() -> None

```

Check if model requests are allowed.
If you're defining your own models that have costs or latency associated with their use, you should call this in [`Model.request`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model.request) and [`Model.request_stream`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model.request_stream).
Raises:
Type | Description  
---|---  
|  If model requests are not allowed.  
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
defcheck_allow_model_requests() -> None:
"""Check if model requests are allowed.
  If you're defining your own models that have costs or latency associated with their use, you should call this in
  [`Model.request`][pydantic_ai.models.Model.request] and [`Model.request_stream`][pydantic_ai.models.Model.request_stream].
  Raises:
    RuntimeError: If model requests are not allowed.
  """
  if not ALLOW_MODEL_REQUESTS:
    raise RuntimeError('Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False')

```
  
---|---  
###  override_allow_model_requests
```
override_allow_model_requests(
  allow_model_requests: ,
) -> [None]

```

Context manager to temporarily override [`ALLOW_MODEL_REQUESTS`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.ALLOW_MODEL_REQUESTS).
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`allow_model_requests` |  Whether to allow model requests within the context. |  _required_  
Source code in `pydantic_ai_slim/pydantic_ai/models/__init__.py`
```
```
| ```
@contextmanager
defoverride_allow_model_requests(allow_model_requests: bool) -> Iterator[None]:
"""Context manager to temporarily override [`ALLOW_MODEL_REQUESTS`][pydantic_ai.models.ALLOW_MODEL_REQUESTS].
  Args:
    allow_model_requests: Whether to allow model requests within the context.
  """
  global ALLOW_MODEL_REQUESTS
  old_value = ALLOW_MODEL_REQUESTS
  ALLOW_MODEL_REQUESTS = allow_model_requests # pyright: ignore[reportConstantRedefinition]
  try:
    yield
  finally:
    ALLOW_MODEL_REQUESTS = old_value # pyright: ignore[reportConstantRedefinition]

```
  
---|---

# `pydantic_ai.models.bedrock`
## Setup
For details on how to set up authentication with this model, see [model configuration for Bedrock](https://ai.pydantic.dev/models/#bedrock).
###  LatestBedrockModelNames `module-attribute`
```
LatestBedrockModelNames = [
  "amazon.titan-tg1-large",
  "amazon.titan-text-lite-v1",
  "amazon.titan-text-express-v1",
  "us.amazon.nova-pro-v1:0",
  "us.amazon.nova-lite-v1:0",
  "us.amazon.nova-micro-v1:0",
  "anthropic.claude-3-5-sonnet-20241022-v2:0",
  "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
  "anthropic.claude-3-5-haiku-20241022-v1:0",
  "us.anthropic.claude-3-5-haiku-20241022-v1:0",
  "anthropic.claude-instant-v1",
  "anthropic.claude-v2:1",
  "anthropic.claude-v2",
  "anthropic.claude-3-sonnet-20240229-v1:0",
  "us.anthropic.claude-3-sonnet-20240229-v1:0",
  "anthropic.claude-3-haiku-20240307-v1:0",
  "us.anthropic.claude-3-haiku-20240307-v1:0",
  "anthropic.claude-3-opus-20240229-v1:0",
  "us.anthropic.claude-3-opus-20240229-v1:0",
  "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
  "anthropic.claude-3-7-sonnet-20250219-v1:0",
  "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
  "cohere.command-text-v14",
  "cohere.command-r-v1:0",
  "cohere.command-r-plus-v1:0",
  "cohere.command-light-text-v14",
  "meta.llama3-8b-instruct-v1:0",
  "meta.llama3-70b-instruct-v1:0",
  "meta.llama3-1-8b-instruct-v1:0",
  "us.meta.llama3-1-8b-instruct-v1:0",
  "meta.llama3-1-70b-instruct-v1:0",
  "us.meta.llama3-1-70b-instruct-v1:0",
  "meta.llama3-1-405b-instruct-v1:0",
  "us.meta.llama3-2-11b-instruct-v1:0",
  "us.meta.llama3-2-90b-instruct-v1:0",
  "us.meta.llama3-2-1b-instruct-v1:0",
  "us.meta.llama3-2-3b-instruct-v1:0",
  "us.meta.llama3-3-70b-instruct-v1:0",
  "mistral.mistral-7b-instruct-v0:2",
  "mistral.mixtral-8x7b-instruct-v0:1",
  "mistral.mistral-large-2402-v1:0",
  "mistral.mistral-large-2407-v1:0",
]

```

Latest Bedrock models.
###  BedrockModelName `module-attribute`
```
BedrockModelName = [, ]

```

Possible Bedrock model names.
Since Bedrock supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the Bedrock docs](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for a full list.
###  BedrockConverseModel `dataclass`
Bases: 
A model that uses the Bedrock Converse API.
Source code in `pydantic_ai_slim/pydantic_ai/models/bedrock.py`
```
```
| ```
@dataclass(init=False)
classBedrockConverseModel(Model):
"""A model that uses the Bedrock Converse API."""
  client: BedrockRuntimeClient
  _model_name: BedrockModelName = field(repr=False)
  _system: str = field(default='bedrock', repr=False)
  @property
  defmodel_name(self) -> str:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider, ex: openai."""
    return self._system
  def__init__(
    self,
    model_name: BedrockModelName,
    *,
    provider: Literal['bedrock'] | Provider[BaseClient] = 'bedrock',
  ):
"""Initialize a Bedrock model.
    Args:
      model_name: The name of the model to use.
      model_name: The name of the Bedrock model to use. List of model names available
        [here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).
      provider: The provider to use for authentication and API access. Can be either the string
        'bedrock' or an instance of `Provider[BaseClient]`. If not provided, a new provider will be
        created using the other parameters.
    """
    self._model_name = model_name
    if isinstance(provider, str):
      provider = infer_provider(provider)
    self.client = cast('BedrockRuntimeClient', provider.client)
  def_get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolTypeDef]:
    tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
    if model_request_parameters.result_tools:
      tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
    return tools
  @staticmethod
  def_map_tool_definition(f: ToolDefinition) -> ToolTypeDef:
    return {
      'toolSpec': {
        'name': f.name,
        'description': f.description,
        'inputSchema': {'json': f.parameters_json_schema},
      }
    }
  @property
  defbase_url(self) -> str:
    return str(self.client.meta.endpoint_url)
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, result.Usage]:
    response = await self._messages_create(messages, False, model_settings, model_request_parameters)
    return await self._process_response(response)
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
    response = await self._messages_create(messages, True, model_settings, model_request_parameters)
    yield BedrockStreamedResponse(_model_name=self.model_name, _event_stream=response)
  async def_process_response(self, response: ConverseResponseTypeDef) -> tuple[ModelResponse, result.Usage]:
    items: list[ModelResponsePart] = []
    if message := response['output'].get('message'):
      for item in message['content']:
        if text := item.get('text'):
          items.append(TextPart(content=text))
        else:
          tool_use = item.get('toolUse')
          assert tool_use is not None, f'Found a content that is not a text or tool use: {item}'
          items.append(
            ToolCallPart(
              tool_name=tool_use['name'],
              args=tool_use['input'],
              tool_call_id=tool_use['toolUseId'],
            ),
          )
    usage = result.Usage(
      request_tokens=response['usage']['inputTokens'],
      response_tokens=response['usage']['outputTokens'],
      total_tokens=response['usage']['totalTokens'],
    )
    return ModelResponse(items, model_name=self.model_name), usage
  @overload
  async def_messages_create(
    self,
    messages: list[ModelMessage],
    stream: Literal[True],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> EventStream[ConverseStreamOutputTypeDef]:
    pass
  @overload
  async def_messages_create(
    self,
    messages: list[ModelMessage],
    stream: Literal[False],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> ConverseResponseTypeDef:
    pass
  async def_messages_create(
    self,
    messages: list[ModelMessage],
    stream: bool,
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> ConverseResponseTypeDef | EventStream[ConverseStreamOutputTypeDef]:
    tools = self._get_tools(model_request_parameters)
    support_tools_choice = self.model_name.startswith(('anthropic', 'us.anthropic'))
    if not tools or not support_tools_choice:
      tool_choice: ToolChoiceTypeDef = {}
    elif not model_request_parameters.allow_text_result:
      tool_choice = {'any': {}}
    else:
      tool_choice = {'auto': {}}
    system_prompt, bedrock_messages = await self._map_message(messages)
    inference_config = self._map_inference_config(model_settings)
    params = {
      'modelId': self.model_name,
      'messages': bedrock_messages,
      'system': [{'text': system_prompt}],
      'inferenceConfig': inference_config,
      **(
        {'toolConfig': {'tools': tools, **({'toolChoice': tool_choice} if tool_choice else {})}}
        if tools
        else {}
      ),
    }
    if stream:
      model_response = await anyio.to_thread.run_sync(functools.partial(self.client.converse_stream, **params))
      model_response = model_response['stream']
    else:
      model_response = await anyio.to_thread.run_sync(functools.partial(self.client.converse, **params))
    return model_response
  @staticmethod
  def_map_inference_config(
    model_settings: ModelSettings | None,
  ) -> InferenceConfigurationTypeDef:
    model_settings = model_settings or {}
    inference_config: InferenceConfigurationTypeDef = {}
    if max_tokens := model_settings.get('max_tokens'):
      inference_config['maxTokens'] = max_tokens
    if temperature := model_settings.get('temperature'):
      inference_config['temperature'] = temperature
    if top_p := model_settings.get('top_p'):
      inference_config['topP'] = top_p
    # TODO(Marcelo): This is not included in model_settings yet.
    # if stop_sequences := model_settings.get('stop_sequences'):
    #   inference_config['stopSequences'] = stop_sequences
    return inference_config
  async def_map_message(self, messages: list[ModelMessage]) -> tuple[str, list[MessageUnionTypeDef]]:
"""Just maps a `pydantic_ai.Message` to the Bedrock `MessageUnionTypeDef`."""
    system_prompt: str = ''
    bedrock_messages: list[MessageUnionTypeDef] = []
    for m in messages:
      if isinstance(m, ModelRequest):
        for part in m.parts:
          if isinstance(part, SystemPromptPart):
            system_prompt += part.content
          elif isinstance(part, UserPromptPart):
            bedrock_messages.extend(await self._map_user_prompt(part))
          elif isinstance(part, ToolReturnPart):
            assert part.tool_call_id is not None
            bedrock_messages.append(
              {
                'role': 'user',
                'content': [
                  {
                    'toolResult': {
                      'toolUseId': part.tool_call_id,
                      'content': [{'text': part.model_response_str()}],
                      'status': 'success',
                    }
                  }
                ],
              }
            )
          elif isinstance(part, RetryPromptPart):
            # TODO(Marcelo): We need to add a test here.
            if part.tool_name is None: # pragma: no cover
              bedrock_messages.append({'role': 'user', 'content': [{'text': part.model_response()}]})
            else:
              assert part.tool_call_id is not None
              bedrock_messages.append(
                {
                  'role': 'user',
                  'content': [
                    {
                      'toolResult': {
                        'toolUseId': part.tool_call_id,
                        'content': [{'text': part.model_response()}],
                        'status': 'error',
                      }
                    }
                  ],
                }
              )
      elif isinstance(m, ModelResponse):
        content: list[ContentBlockOutputTypeDef] = []
        for item in m.parts:
          if isinstance(item, TextPart):
            content.append({'text': item.content})
          else:
            assert isinstance(item, ToolCallPart)
            content.append(self._map_tool_call(item))
        bedrock_messages.append({'role': 'assistant', 'content': content})
      else:
        assert_never(m)
    return system_prompt, bedrock_messages
  @staticmethod
  async def_map_user_prompt(part: UserPromptPart) -> list[MessageUnionTypeDef]:
    content: list[ContentBlockUnionTypeDef] = []
    if isinstance(part.content, str):
      content.append({'text': part.content})
    else:
      document_count = 0
      for item in part.content:
        if isinstance(item, str):
          content.append({'text': item})
        elif isinstance(item, BinaryContent):
          format = item.format
          if item.is_document:
            document_count += 1
            name = f'Document {document_count}'
            assert format in ('pdf', 'txt', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'md')
            content.append({'document': {'name': name, 'format': format, 'source': {'bytes': item.data}}})
          elif item.is_image:
            assert format in ('jpeg', 'png', 'gif', 'webp')
            content.append({'image': {'format': format, 'source': {'bytes': item.data}}})
          else:
            raise NotImplementedError('Binary content is not supported yet.')
        elif isinstance(item, (ImageUrl, DocumentUrl)):
          response = await cached_async_http_client().get(item.url)
          response.raise_for_status()
          if item.kind == 'image-url':
            format = item.media_type.split('/')[1]
            assert format in ('jpeg', 'png', 'gif', 'webp'), f'Unsupported image format: {format}'
            image: ImageBlockTypeDef = {'format': format, 'source': {'bytes': response.content}}
            content.append({'image': image})
          elif item.kind == 'document-url':
            document_count += 1
            name = f'Document {document_count}'
            data = response.content
            content.append({'document': {'name': name, 'format': item.format, 'source': {'bytes': data}}})
        elif isinstance(item, AudioUrl): # pragma: no cover
          raise NotImplementedError('Audio is not supported yet.')
        else:
          assert_never(item)
    return [{'role': 'user', 'content': content}]
  @staticmethod
  def_map_tool_call(t: ToolCallPart) -> ContentBlockOutputTypeDef:
    return {
      'toolUse': {'toolUseId': _utils.guard_tool_call_id(t=t), 'name': t.tool_name, 'input': t.args_as_dict()}
    }

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider, ex: openai.
####  __init__
```
__init__(
  model_name: ,
  *,
  provider: (
    ["bedrock"] | [BaseClient]
  ) = "bedrock"
)

```

Initialize a Bedrock model.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`model_name` |  |  The name of the model to use. |  _required_  
`model_name` |  |  The name of the Bedrock model to use. List of model names available [here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html). |  _required_  
`provider` |  `Literal[](https://docs.python.org/3/library/typing.html#typing.Literal "typing.Literal")['bedrock'] | Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[BaseClient]` |  The provider to use for authentication and API access. Can be either the string 'bedrock' or an instance of `Provider[BaseClient]`. If not provided, a new provider will be created using the other parameters. |  `'bedrock'`  
Source code in `pydantic_ai_slim/pydantic_ai/models/bedrock.py`
```
```
| ```
def__init__(
  self,
  model_name: BedrockModelName,
  *,
  provider: Literal['bedrock'] | Provider[BaseClient] = 'bedrock',
):
"""Initialize a Bedrock model.
  Args:
    model_name: The name of the model to use.
    model_name: The name of the Bedrock model to use. List of model names available
      [here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).
    provider: The provider to use for authentication and API access. Can be either the string
      'bedrock' or an instance of `Provider[BaseClient]`. If not provided, a new provider will be
      created using the other parameters.
  """
  self._model_name = model_name
  if isinstance(provider, str):
    provider = infer_provider(provider)
  self.client = cast('BedrockRuntimeClient', provider.client)

```
  
---|---  
###  BedrockStreamedResponse `dataclass`
Bases: 
Implementation of `StreamedResponse` for Bedrock models.
Source code in `pydantic_ai_slim/pydantic_ai/models/bedrock.py`
```
```
| ```
@dataclass
classBedrockStreamedResponse(StreamedResponse):
"""Implementation of `StreamedResponse` for Bedrock models."""
  _model_name: BedrockModelName
  _event_stream: EventStream[ConverseStreamOutputTypeDef]
  _timestamp: datetime = field(default_factory=_utils.now_utc)
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
"""Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.
    This method should be implemented by subclasses to translate the vendor-specific stream of events into
    pydantic_ai-format events.
    """
    chunk: ConverseStreamOutputTypeDef
    tool_id: str | None = None
    async for chunk in _AsyncIteratorWrapper(self._event_stream):
      # TODO(Marcelo): Switch this to `match` when we drop Python 3.9 support.
      if 'messageStart' in chunk:
        continue
      if 'messageStop' in chunk:
        continue
      if 'metadata' in chunk:
        if 'usage' in chunk['metadata']:
          self._usage += self._map_usage(chunk['metadata'])
        continue
      if 'contentBlockStart' in chunk:
        index = chunk['contentBlockStart']['contentBlockIndex']
        start = chunk['contentBlockStart']['start']
        if 'toolUse' in start:
          tool_use_start = start['toolUse']
          tool_id = tool_use_start['toolUseId']
          tool_name = tool_use_start['name']
          maybe_event = self._parts_manager.handle_tool_call_delta(
            vendor_part_id=index,
            tool_name=tool_name,
            args=None,
            tool_call_id=tool_id,
          )
          if maybe_event:
            yield maybe_event
      if 'contentBlockDelta' in chunk:
        index = chunk['contentBlockDelta']['contentBlockIndex']
        delta = chunk['contentBlockDelta']['delta']
        if 'text' in delta:
          yield self._parts_manager.handle_text_delta(vendor_part_id=index, content=delta['text'])
        if 'toolUse' in delta:
          tool_use = delta['toolUse']
          maybe_event = self._parts_manager.handle_tool_call_delta(
            vendor_part_id=index,
            tool_name=tool_use.get('name'),
            args=tool_use.get('input'),
            tool_call_id=tool_id,
          )
          if maybe_event:
            yield maybe_event
  @property
  deftimestamp(self) -> datetime:
    return self._timestamp
  @property
  defmodel_name(self) -> str:
"""Get the model name of the response."""
    return self._model_name
  def_map_usage(self, metadata: ConverseStreamMetadataEventTypeDef) -> result.Usage:
    return result.Usage(
      request_tokens=metadata['usage']['inputTokens'],
      response_tokens=metadata['usage']['outputTokens'],
      total_tokens=metadata['usage']['totalTokens'],
    )

```
  
---|---  
####  model_name `property`
```
model_name: 

```

Get the model name of the response.

# `pydantic_ai.models.cohere`
## Setup
For details on how to set up authentication with this model, see [model configuration for Cohere](https://ai.pydantic.dev/models/#cohere).
###  LatestCohereModelNames `module-attribute`
```
LatestCohereModelNames = [
  "c4ai-aya-expanse-32b",
  "c4ai-aya-expanse-8b",
  "command",
  "command-light",
  "command-light-nightly",
  "command-nightly",
  "command-r",
  "command-r-03-2024",
  "command-r-08-2024",
  "command-r-plus",
  "command-r-plus-04-2024",
  "command-r-plus-08-2024",
  "command-r7b-12-2024",
]

```

Latest Cohere models.
###  CohereModelName `module-attribute`
```
CohereModelName = [, ]

```

Possible Cohere model names.
Since Cohere supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [Cohere's docs](https://docs.cohere.com/v2/docs/models) for a list of all available models.
###  CohereModelSettings
Bases: 
Settings used for a Cohere model request.
Source code in `pydantic_ai_slim/pydantic_ai/models/cohere.py`
```
```
| ```
classCohereModelSettings(ModelSettings):
"""Settings used for a Cohere model request."""

```
  
---|---  
###  CohereModel `dataclass`
Bases: 
A model that uses the Cohere API.
Internally, this uses the [Cohere Python client](https://github.com/cohere-ai/cohere-python) to interact with the API.
Apart from `__init__`, all methods are private or match those of the base class.
Source code in `pydantic_ai_slim/pydantic_ai/models/cohere.py`
```
```
| ```
@dataclass(init=False)
classCohereModel(Model):
"""A model that uses the Cohere API.
  Internally, this uses the [Cohere Python client](
  https://github.com/cohere-ai/cohere-python) to interact with the API.
  Apart from `__init__`, all methods are private or match those of the base class.
  """
  client: AsyncClientV2 = field(repr=False)
  _model_name: CohereModelName = field(repr=False)
  _system: str = field(default='cohere', repr=False)
  def__init__(
    self,
    model_name: CohereModelName,
    *,
    provider: Literal['cohere'] | Provider[AsyncClientV2] = 'cohere',
  ):
"""Initialize an Cohere model.
    Args:
      model_name: The name of the Cohere model to use. List of model names
        available [here](https://docs.cohere.com/docs/models#command).
      provider: The provider to use for authentication and API access. Can be either the string
        'cohere' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be
        created using the other parameters.
    """
    self._model_name: CohereModelName = model_name
    if isinstance(provider, str):
      provider = infer_provider(provider)
    self.client = provider.client
  @property
  defbase_url(self) -> str:
    client_wrapper = self.client._client_wrapper # type: ignore
    return str(client_wrapper.get_base_url())
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, result.Usage]:
    check_allow_model_requests()
    response = await self._chat(messages, cast(CohereModelSettings, model_settings or {}), model_request_parameters)
    return self._process_response(response), _map_usage(response)
  @property
  defmodel_name(self) -> CohereModelName:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider."""
    return self._system
  async def_chat(
    self,
    messages: list[ModelMessage],
    model_settings: CohereModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> ChatResponse:
    tools = self._get_tools(model_request_parameters)
    cohere_messages = list(chain(*(self._map_message(m) for m in messages)))
    try:
      return await self.client.chat(
        model=self._model_name,
        messages=cohere_messages,
        tools=tools or OMIT,
        max_tokens=model_settings.get('max_tokens', OMIT),
        temperature=model_settings.get('temperature', OMIT),
        p=model_settings.get('top_p', OMIT),
        seed=model_settings.get('seed', OMIT),
        presence_penalty=model_settings.get('presence_penalty', OMIT),
        frequency_penalty=model_settings.get('frequency_penalty', OMIT),
      )
    except ApiError as e:
      if (status_code := e.status_code) and status_code >= 400:
        raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) frome
      raise
  def_process_response(self, response: ChatResponse) -> ModelResponse:
"""Process a non-streamed response, and prepare a message to return."""
    parts: list[ModelResponsePart] = []
    if response.message.content is not None and len(response.message.content)  0:
      # While Cohere's API returns a list, it only does that for future proofing
      # and currently only one item is being returned.
      choice = response.message.content[0]
      parts.append(TextPart(choice.text))
    for c in response.message.tool_calls or []:
      if c.function and c.function.name and c.function.arguments:
        parts.append(
          ToolCallPart(
            tool_name=c.function.name,
            args=c.function.arguments,
            tool_call_id=c.id or _generate_tool_call_id(),
          )
        )
    return ModelResponse(parts=parts, model_name=self._model_name)
  def_map_message(self, message: ModelMessage) -> Iterable[ChatMessageV2]:
"""Just maps a `pydantic_ai.Message` to a `cohere.ChatMessageV2`."""
    if isinstance(message, ModelRequest):
      yield from self._map_user_message(message)
    elif isinstance(message, ModelResponse):
      texts: list[str] = []
      tool_calls: list[ToolCallV2] = []
      for item in message.parts:
        if isinstance(item, TextPart):
          texts.append(item.content)
        elif isinstance(item, ToolCallPart):
          tool_calls.append(self._map_tool_call(item))
        else:
          assert_never(item)
      message_param = AssistantChatMessageV2(role='assistant')
      if texts:
        message_param.content = [TextAssistantMessageContentItem(text='\n\n'.join(texts))]
      if tool_calls:
        message_param.tool_calls = tool_calls
      yield message_param
    else:
      assert_never(message)
  def_get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolV2]:
    tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
    if model_request_parameters.result_tools:
      tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
    return tools
  @staticmethod
  def_map_tool_call(t: ToolCallPart) -> ToolCallV2:
    return ToolCallV2(
      id=_guard_tool_call_id(t=t),
      type='function',
      function=ToolCallV2Function(
        name=t.tool_name,
        arguments=t.args_as_json_str(),
      ),
    )
  @staticmethod
  def_map_tool_definition(f: ToolDefinition) -> ToolV2:
    return ToolV2(
      type='function',
      function=ToolV2Function(
        name=f.name,
        description=f.description,
        parameters=f.parameters_json_schema,
      ),
    )
  @classmethod
  def_map_user_message(cls, message: ModelRequest) -> Iterable[ChatMessageV2]:
    for part in message.parts:
      if isinstance(part, SystemPromptPart):
        yield SystemChatMessageV2(role='system', content=part.content)
      elif isinstance(part, UserPromptPart):
        if isinstance(part.content, str):
          yield UserChatMessageV2(role='user', content=part.content)
        else:
          raise RuntimeError('Cohere does not yet support multi-modal inputs.')
      elif isinstance(part, ToolReturnPart):
        yield ToolChatMessageV2(
          role='tool',
          tool_call_id=_guard_tool_call_id(t=part),
          content=part.model_response_str(),
        )
      elif isinstance(part, RetryPromptPart):
        if part.tool_name is None:
          yield UserChatMessageV2(role='user', content=part.model_response())
        else:
          yield ToolChatMessageV2(
            role='tool',
            tool_call_id=_guard_tool_call_id(t=part),
            content=part.model_response(),
          )
      else:
        assert_never(part)

```
  
---|---  
####  __init__
```
__init__(
  model_name: ,
  *,
  provider: (
    ["cohere"] | [AsyncClientV2]
  ) = "cohere"
)

```

Initialize an Cohere model.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`model_name` |  |  The name of the Cohere model to use. List of model names available [here](https://docs.cohere.com/docs/models#command). |  _required_  
`provider` |  `Literal[](https://docs.python.org/3/library/typing.html#typing.Literal "typing.Literal")['cohere'] | Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncClientV2]` |  The provider to use for authentication and API access. Can be either the string 'cohere' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be created using the other parameters. |  `'cohere'`  
Source code in `pydantic_ai_slim/pydantic_ai/models/cohere.py`
```
```
| ```
def__init__(
  self,
  model_name: CohereModelName,
  *,
  provider: Literal['cohere'] | Provider[AsyncClientV2] = 'cohere',
):
"""Initialize an Cohere model.
  Args:
    model_name: The name of the Cohere model to use. List of model names
      available [here](https://docs.cohere.com/docs/models#command).
    provider: The provider to use for authentication and API access. Can be either the string
      'cohere' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be
      created using the other parameters.
  """
  self._model_name: CohereModelName = model_name
  if isinstance(provider, str):
    provider = infer_provider(provider)
  self.client = provider.client

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider.

# pydantic_ai.models.fallback
###  FallbackModel `dataclass`
Bases: 
A model that uses one or more fallback models upon failure.
Apart from `__init__`, all methods are private or match those of the base class.
Source code in `pydantic_ai_slim/pydantic_ai/models/fallback.py`
```
```
| ```
@dataclass(init=False)
classFallbackModel(Model):
"""A model that uses one or more fallback models upon failure.
  Apart from `__init__`, all methods are private or match those of the base class.
  """
  models: list[Model]
  _model_name: str = field(repr=False)
  _fallback_on: Callable[[Exception], bool]
  def__init__(
    self,
    default_model: Model | KnownModelName,
    *fallback_models: Model | KnownModelName,
    fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (ModelHTTPError,),
  ):
"""Initialize a fallback model instance.
    Args:
      default_model: The name or instance of the default model to use.
      fallback_models: The names or instances of the fallback models to use upon failure.
      fallback_on: A callable or tuple of exceptions that should trigger a fallback.
    """
    self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]
    if isinstance(fallback_on, tuple):
      self._fallback_on = _default_fallback_condition_factory(fallback_on)
    else:
      self._fallback_on = fallback_on
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, Usage]:
"""Try each model in sequence until one succeeds.
    In case of failure, raise a FallbackExceptionGroup with all exceptions.
    """
    exceptions: list[Exception] = []
    for model in self.models:
      try:
        response, usage = await model.request(messages, model_settings, model_request_parameters)
      except Exception as exc:
        if self._fallback_on(exc):
          exceptions.append(exc)
          continue
        raise exc
      self._set_span_attributes(model)
      return response, usage
    raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
"""Try each model in sequence until one succeeds."""
    exceptions: list[Exception] = []
    for model in self.models:
      async with AsyncExitStack() as stack:
        try:
          response = await stack.enter_async_context(
            model.request_stream(messages, model_settings, model_request_parameters)
          )
        except Exception as exc:
          if self._fallback_on(exc):
            exceptions.append(exc)
            continue
          raise exc
        self._set_span_attributes(model)
        yield response
        return
    raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)
  def_set_span_attributes(self, model: Model):
    with suppress(Exception):
      span = get_current_span()
      if span.is_recording():
        attributes = getattr(span, 'attributes', {})
        if attributes.get('gen_ai.request.model') == self.model_name:
          span.set_attributes(InstrumentedModel.model_attributes(model))
  @property
  defmodel_name(self) -> str:
"""The model name."""
    return f'fallback:{",".join(model.model_nameformodelinself.models)}'
  @property
  defsystem(self) -> str:
    return f'fallback:{",".join(model.systemformodelinself.models)}'
  @property
  defbase_url(self) -> str | None:
    return self.models[0].base_url

```
  
---|---  
####  __init__
```
__init__(
  default_model:  | ,
  *fallback_models:  | ,
  fallback_on: (
    [[], ]
    | [[], ...]
  ) = (,)
)

```

Initialize a fallback model instance.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`default_model` |  |  The name or instance of the default model to use. |  _required_  
`fallback_models` |  |  The names or instances of the fallback models to use upon failure.  
`fallback_on` |  `Callable[](https://docs.python.org/3/library/typing.html#typing.Callable "typing.Callable")[[Exception[](https://docs.python.org/3/library/exceptions.html#Exception)], bool[](https://docs.python.org/3/library/functions.html#bool)] | tuple[](https://docs.python.org/3/library/stdtypes.html#tuple)[type[](https://docs.python.org/3/library/functions.html#type)[Exception[](https://docs.python.org/3/library/exceptions.html#Exception)], ...]` |  A callable or tuple of exceptions that should trigger a fallback. |   
Source code in `pydantic_ai_slim/pydantic_ai/models/fallback.py`
```
```
| ```
def__init__(
  self,
  default_model: Model | KnownModelName,
  *fallback_models: Model | KnownModelName,
  fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (ModelHTTPError,),
):
"""Initialize a fallback model instance.
  Args:
    default_model: The name or instance of the default model to use.
    fallback_models: The names or instances of the fallback models to use upon failure.
    fallback_on: A callable or tuple of exceptions that should trigger a fallback.
  """
  self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]
  if isinstance(fallback_on, tuple):
    self._fallback_on = _default_fallback_condition_factory(fallback_on)
  else:
    self._fallback_on = fallback_on

```
  
---|---  
####  request `async`
```
request(
  messages: [],
  model_settings:  | None,
  model_request_parameters: ,
) -> [, ]

```

Try each model in sequence until one succeeds.
In case of failure, raise a FallbackExceptionGroup with all exceptions.
Source code in `pydantic_ai_slim/pydantic_ai/models/fallback.py`
```
```
| ```
async defrequest(
  self,
  messages: list[ModelMessage],
  model_settings: ModelSettings | None,
  model_request_parameters: ModelRequestParameters,
) -> tuple[ModelResponse, Usage]:
"""Try each model in sequence until one succeeds.
  In case of failure, raise a FallbackExceptionGroup with all exceptions.
  """
  exceptions: list[Exception] = []
  for model in self.models:
    try:
      response, usage = await model.request(messages, model_settings, model_request_parameters)
    except Exception as exc:
      if self._fallback_on(exc):
        exceptions.append(exc)
        continue
      raise exc
    self._set_span_attributes(model)
    return response, usage
  raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

```
  
---|---  
####  request_stream `async`
```
request_stream(
  messages: [],
  model_settings:  | None,
  model_request_parameters: ,
) -> []

```

Try each model in sequence until one succeeds.
Source code in `pydantic_ai_slim/pydantic_ai/models/fallback.py`
```
```
| ```
@asynccontextmanager
async defrequest_stream(
  self,
  messages: list[ModelMessage],
  model_settings: ModelSettings | None,
  model_request_parameters: ModelRequestParameters,
) -> AsyncIterator[StreamedResponse]:
"""Try each model in sequence until one succeeds."""
  exceptions: list[Exception] = []
  for model in self.models:
    async with AsyncExitStack() as stack:
      try:
        response = await stack.enter_async_context(
          model.request_stream(messages, model_settings, model_request_parameters)
        )
      except Exception as exc:
        if self._fallback_on(exc):
          exceptions.append(exc)
          continue
        raise exc
      self._set_span_attributes(model)
      yield response
      return
  raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.

# `pydantic_ai.models.function`
A model controlled by a local function.
[`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) is similar to [`TestModel`](https://ai.pydantic.dev/api/models/test/), but allows greater control over the model's behavior.
Its primary use case is for more advanced unit testing than is possible with `TestModel`.
Here's a minimal example:
function_model_usage.py```
frompydantic_aiimport Agent
frompydantic_ai.messagesimport ModelMessage, ModelResponse, TextPart
frompydantic_ai.models.functionimport FunctionModel, AgentInfo
my_agent = Agent('openai:gpt-4o')

async defmodel_function(
  messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
  print(messages)
"""

    ModelRequest(
      parts=[
        UserPromptPart(
          content='Testing my agent...',
          timestamp=datetime.datetime(...),
          part_kind='user-prompt',


      kind='request',


  """
  print(info)
"""
  AgentInfo(
    function_tools=[], allow_text_result=True, result_tools=[], model_settings=None

  """
  return ModelResponse(parts=[TextPart('hello world')])

async deftest_my_agent():
"""Unit test for my_agent, to be run by pytest."""
  with my_agent.override(model=FunctionModel(model_function)):
    result = await my_agent.run('Testing my agent...')
    assert result.data == 'hello world'

```

See [Unit testing with `FunctionModel`](https://ai.pydantic.dev/testing-evals/#unit-testing-with-functionmodel) for detailed documentation.
###  FunctionModel `dataclass`
Bases: 
A model controlled by a local function.
Apart from `__init__`, all methods are private or match those of the base class.
Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`
```
```
| ```
@dataclass(init=False)
classFunctionModel(Model):
"""A model controlled by a local function.
  Apart from `__init__`, all methods are private or match those of the base class.
  """
  function: FunctionDef | None = None
  stream_function: StreamFunctionDef | None = None
  _model_name: str = field(repr=False)
  _system: str = field(default='function', repr=False)
  @overload
  def__init__(self, function: FunctionDef, *, model_name: str | None = None) -> None: ...
  @overload
  def__init__(self, *, stream_function: StreamFunctionDef, model_name: str | None = None) -> None: ...
  @overload
  def__init__(
    self, function: FunctionDef, *, stream_function: StreamFunctionDef, model_name: str | None = None
  ) -> None: ...
  def__init__(
    self,
    function: FunctionDef | None = None,
    *,
    stream_function: StreamFunctionDef | None = None,
    model_name: str | None = None,
  ):
"""Initialize a `FunctionModel`.
    Either `function` or `stream_function` must be provided, providing both is allowed.
    Args:
      function: The function to call for non-streamed requests.
      stream_function: The function to call for streamed requests.
      model_name: The name of the model. If not provided, a name is generated from the function names.
    """
    if function is None and stream_function is None:
      raise TypeError('Either `function` or `stream_function` must be provided')
    self.function = function
    self.stream_function = stream_function
    function_name = self.function.__name__ if self.function is not None else ''
    stream_function_name = self.stream_function.__name__ if self.stream_function is not None else ''
    self._model_name = model_name or f'function:{function_name}:{stream_function_name}'
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, usage.Usage]:
    agent_info = AgentInfo(
      model_request_parameters.function_tools,
      model_request_parameters.allow_text_result,
      model_request_parameters.result_tools,
      model_settings,
    )
    assert self.function is not None, 'FunctionModel must receive a `function` to support non-streamed requests'
    if inspect.iscoroutinefunction(self.function):
      response = await self.function(messages, agent_info)
    else:
      response_ = await _utils.run_in_executor(self.function, messages, agent_info)
      assert isinstance(response_, ModelResponse), response_
      response = response_
    response.model_name = self._model_name
    # TODO is `messages` right here? Should it just be new messages?
    return response, _estimate_usage(chain(messages, [response]))
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
    agent_info = AgentInfo(
      model_request_parameters.function_tools,
      model_request_parameters.allow_text_result,
      model_request_parameters.result_tools,
      model_settings,
    )
    assert self.stream_function is not None, (
      'FunctionModel must receive a `stream_function` to support streamed requests'
    )
    response_stream = PeekableAsyncStream(self.stream_function(messages, agent_info))
    first = await response_stream.peek()
    if isinstance(first, _utils.Unset):
      raise ValueError('Stream function must return at least one item')
    yield FunctionStreamedResponse(_model_name=self._model_name, _iter=response_stream)
  @property
  defmodel_name(self) -> str:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider."""
    return self._system

```
  
---|---  
####  __init__
```
__init__(
  function: , *, model_name:  | None = None
) -> None

```

```
__init__(
  *,
  stream_function: ,
  model_name:  | None = None
) -> None

```

```
__init__(
  function: ,
  *,
  stream_function: ,
  model_name:  | None = None
) -> None

```

```
__init__(
  function:  | None = None,
  *,
  stream_function:  | None = None,
  model_name:  | None = None
)

```

Initialize a `FunctionModel`.
Either `function` or `stream_function` must be provided, providing both is allowed.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`function` |  `FunctionDef[](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionDef "pydantic_ai.models.function.FunctionDef") | None` |  The function to call for non-streamed requests. |  `None`  
`stream_function` |  `StreamFunctionDef[](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.StreamFunctionDef "pydantic_ai.models.function.StreamFunctionDef") | None` |  The function to call for streamed requests. |  `None`  
`model_name` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The name of the model. If not provided, a name is generated from the function names. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`
```
```
| ```
def__init__(
  self,
  function: FunctionDef | None = None,
  *,
  stream_function: StreamFunctionDef | None = None,
  model_name: str | None = None,
):
"""Initialize a `FunctionModel`.
  Either `function` or `stream_function` must be provided, providing both is allowed.
  Args:
    function: The function to call for non-streamed requests.
    stream_function: The function to call for streamed requests.
    model_name: The name of the model. If not provided, a name is generated from the function names.
  """
  if function is None and stream_function is None:
    raise TypeError('Either `function` or `stream_function` must be provided')
  self.function = function
  self.stream_function = stream_function
  function_name = self.function.__name__ if self.function is not None else ''
  stream_function_name = self.stream_function.__name__ if self.stream_function is not None else ''
  self._model_name = model_name or f'function:{function_name}:{stream_function_name}'

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider.
###  AgentInfo `dataclass`
Information about an agent.
This is passed as the second to functions used within [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel).
Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`
```
```
| ```
@dataclass(frozen=True)
classAgentInfo:
"""Information about an agent.
  This is passed as the second to functions used within [`FunctionModel`][pydantic_ai.models.function.FunctionModel].
  """
  function_tools: list[ToolDefinition]
"""The function tools available on this agent.
  These are the tools registered via the [`tool`][pydantic_ai.Agent.tool] and
  [`tool_plain`][pydantic_ai.Agent.tool_plain] decorators.
  """
  allow_text_result: bool
"""Whether a plain text result is allowed."""
  result_tools: list[ToolDefinition]
"""The tools that can called as the final result of the run."""
  model_settings: ModelSettings | None
"""The model settings passed to the run call."""

```
  
---|---  
####  function_tools `instance-attribute`
```
function_tools: []

```

The function tools available on this agent.
These are the tools registered via the [`tool`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool) and [`tool_plain`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool_plain) decorators.
####  allow_text_result `instance-attribute`
```
allow_text_result: 

```

Whether a plain text result is allowed.
####  result_tools `instance-attribute`
```
result_tools: []

```

The tools that can called as the final result of the run.
####  model_settings `instance-attribute`
```
model_settings:  | None

```

The model settings passed to the run call.
###  DeltaToolCall `dataclass`
Incremental change to a tool call.
Used to describe a chunk when streaming structured responses.
Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`
```
```
| ```
@dataclass
classDeltaToolCall:
"""Incremental change to a tool call.
  Used to describe a chunk when streaming structured responses.
  """
  name: str | None = None
"""Incremental change to the name of the tool."""
  json_args: str | None = None
"""Incremental change to the arguments as JSON"""
  tool_call_id: str | None = None
"""Incremental change to the tool call ID."""

```
  
---|---  
####  name `class-attribute` `instance-attribute`
```
name:  | None = None

```

Incremental change to the name of the tool.
####  json_args `class-attribute` `instance-attribute`
```
json_args:  | None = None

```

Incremental change to the arguments as JSON
####  tool_call_id `class-attribute` `instance-attribute`
```
tool_call_id:  | None = None

```

Incremental change to the tool call ID.
###  DeltaToolCalls `module-attribute`
```
DeltaToolCalls:  = [, ]

```

A mapping of tool call IDs to incremental changes.
###  FunctionDef `module-attribute`
```
FunctionDef:  = [
  [[], ],
  [, []],
]

```

A function used to generate a non-streamed response.
###  StreamFunctionDef `module-attribute`
```
StreamFunctionDef:  = [
  [[], ],
  [[, ]],
]

```

A function used to generate a streamed response.
While this is defined as having return type of `AsyncIterator[Union[str, DeltaToolCalls]]`, it should really be considered as `Union[AsyncIterator[str], AsyncIterator[DeltaToolCalls]`,
E.g. you need to yield all text or all `DeltaToolCalls`, not mix them.
###  FunctionStreamedResponse `dataclass`
Bases: 
Implementation of `StreamedResponse` for [FunctionModel](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel).
Source code in `pydantic_ai_slim/pydantic_ai/models/function.py`
```
```
| ```
@dataclass
classFunctionStreamedResponse(StreamedResponse):
"""Implementation of `StreamedResponse` for [FunctionModel][pydantic_ai.models.function.FunctionModel]."""
  _model_name: str
  _iter: AsyncIterator[str | DeltaToolCalls]
  _timestamp: datetime = field(default_factory=_utils.now_utc)
  def__post_init__(self):
    self._usage += _estimate_usage([])
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    async for item in self._iter:
      if isinstance(item, str):
        response_tokens = _estimate_string_tokens(item)
        self._usage += usage.Usage(response_tokens=response_tokens, total_tokens=response_tokens)
        yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=item)
      else:
        delta_tool_calls = item
        for dtc_index, delta_tool_call in delta_tool_calls.items():
          if delta_tool_call.json_args:
            response_tokens = _estimate_string_tokens(delta_tool_call.json_args)
            self._usage += usage.Usage(response_tokens=response_tokens, total_tokens=response_tokens)
          maybe_event = self._parts_manager.handle_tool_call_delta(
            vendor_part_id=dtc_index,
            tool_name=delta_tool_call.name,
            args=delta_tool_call.json_args,
            tool_call_id=delta_tool_call.tool_call_id,
          )
          if maybe_event is not None:
            yield maybe_event
  @property
  defmodel_name(self) -> str:
"""Get the model name of the response."""
    return self._model_name
  @property
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    return self._timestamp

```
  
---|---  
####  model_name `property`
```
model_name: 

```

Get the model name of the response.
####  timestamp `property`
```
timestamp: 

```

Get the timestamp of the response.

# `pydantic_ai.models.gemini`
Custom interface to the `generativelanguage.googleapis.com` API using [HTTPX](https://www.python-httpx.org/) and [Pydantic](https://docs.pydantic.dev/latest/).
The Google SDK for interacting with the `generativelanguage.googleapis.com` API [`google-generativeai`](https://ai.google.dev/gemini-api/docs/quickstart?lang=python) reads like it was written by a Java developer who thought they knew everything about OOP, spent 30 minutes trying to learn Python, gave up and decided to build the library to prove how horrible Python is. It also doesn't use httpx for HTTP requests, and tries to implement tool calling itself, but doesn't use Pydantic or equivalent for validation.
We therefore implement support for the API directly.
Despite these shortcomings, the Gemini model is actually quite powerful and very fast.
## Setup
For details on how to set up authentication with this model, see [model configuration for Gemini](https://ai.pydantic.dev/models/#gemini).
###  LatestGeminiModelNames `module-attribute`
```
LatestGeminiModelNames = [
  "gemini-1.5-flash",
  "gemini-1.5-flash-8b",
  "gemini-1.5-pro",
  "gemini-1.0-pro",
  "gemini-2.0-flash-exp",
  "gemini-2.0-flash-thinking-exp-01-21",
  "gemini-exp-1206",
  "gemini-2.0-flash",
  "gemini-2.0-flash-lite-preview-02-05",
  "gemini-2.0-pro-exp-02-05",
]

```

Latest Gemini models.
###  GeminiModelName `module-attribute`
```
GeminiModelName = [, ]

```

Possible Gemini model names.
Since Gemini supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the Gemini API docs](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations) for a full list.
###  GeminiModelSettings
Bases: 
Settings used for a Gemini model request.
Source code in `pydantic_ai_slim/pydantic_ai/models/gemini.py`
```
```
| ```
classGeminiModelSettings(ModelSettings):
"""Settings used for a Gemini model request."""
  gemini_safety_settings: list[GeminiSafetySettings]

```
  
---|---  
###  GeminiModel `dataclass`
Bases: 
A model that uses Gemini via `generativelanguage.googleapis.com` API.
This is implemented from scratch rather than using a dedicated SDK, good API documentation is available [here](https://ai.google.dev/api).
Apart from `__init__`, all methods are private or match those of the base class.
Source code in `pydantic_ai_slim/pydantic_ai/models/gemini.py`
```
```
| ```
@dataclass(init=False)
classGeminiModel(Model):
"""A model that uses Gemini via `generativelanguage.googleapis.com` API.
  This is implemented from scratch rather than using a dedicated SDK, good API documentation is
  available [here](https://ai.google.dev/api).
  Apart from `__init__`, all methods are private or match those of the base class.
  """
  client: httpx.AsyncClient = field(repr=False)
  _model_name: GeminiModelName = field(repr=False)
  _provider: Literal['google-gla', 'google-vertex'] | Provider[httpx.AsyncClient] | None = field(repr=False)
  _auth: AuthProtocol | None = field(repr=False)
  _url: str | None = field(repr=False)
  _system: str = field(default='gemini', repr=False)
  def__init__(
    self,
    model_name: GeminiModelName,
    *,
    provider: Literal['google-gla', 'google-vertex'] | Provider[httpx.AsyncClient] = 'google-gla',
  ):
"""Initialize a Gemini model.
    Args:
      model_name: The name of the model to use.
      provider: The provider to use for authentication and API access. Can be either the string
        'google-gla' or 'google-vertex' or an instance of `Provider[httpx.AsyncClient]`.
        If not provided, a new provider will be created using the other parameters.
    """
    self._model_name = model_name
    self._provider = provider
    if isinstance(provider, str):
      provider = infer_provider(provider)
    self._system = provider.name
    self.client = provider.client
    self._url = str(self.client.base_url)
  @property
  defbase_url(self) -> str:
    assert self._url is not None, 'URL not initialized'
    return self._url
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, usage.Usage]:
    check_allow_model_requests()
    async with self._make_request(
      messages, False, cast(GeminiModelSettings, model_settings or {}), model_request_parameters
    ) as http_response:
      response = _gemini_response_ta.validate_json(await http_response.aread())
    return self._process_response(response), _metadata_as_usage(response)
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
    check_allow_model_requests()
    async with self._make_request(
      messages, True, cast(GeminiModelSettings, model_settings or {}), model_request_parameters
    ) as http_response:
      yield await self._process_streamed_response(http_response)
  @property
  defmodel_name(self) -> GeminiModelName:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider."""
    return self._system
  def_get_tools(self, model_request_parameters: ModelRequestParameters) -> _GeminiTools | None:
    tools = [_function_from_abstract_tool(t) for t in model_request_parameters.function_tools]
    if model_request_parameters.result_tools:
      tools += [_function_from_abstract_tool(t) for t in model_request_parameters.result_tools]
    return _GeminiTools(function_declarations=tools) if tools else None
  def_get_tool_config(
    self, model_request_parameters: ModelRequestParameters, tools: _GeminiTools | None
  ) -> _GeminiToolConfig | None:
    if model_request_parameters.allow_text_result:
      return None
    elif tools:
      return _tool_config([t['name'] for t in tools['function_declarations']])
    else:
      return _tool_config([])
  @asynccontextmanager
  async def_make_request(
    self,
    messages: list[ModelMessage],
    streamed: bool,
    model_settings: GeminiModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[HTTPResponse]:
    tools = self._get_tools(model_request_parameters)
    tool_config = self._get_tool_config(model_request_parameters, tools)
    sys_prompt_parts, contents = await self._message_to_gemini_content(messages)
    request_data = _GeminiRequest(contents=contents)
    if sys_prompt_parts:
      request_data['system_instruction'] = _GeminiTextContent(role='user', parts=sys_prompt_parts)
    if tools is not None:
      request_data['tools'] = tools
    if tool_config is not None:
      request_data['tool_config'] = tool_config
    generation_config: _GeminiGenerationConfig = {}
    if model_settings:
      if (max_tokens := model_settings.get('max_tokens')) is not None:
        generation_config['max_output_tokens'] = max_tokens
      if (temperature := model_settings.get('temperature')) is not None:
        generation_config['temperature'] = temperature
      if (top_p := model_settings.get('top_p')) is not None:
        generation_config['top_p'] = top_p
      if (presence_penalty := model_settings.get('presence_penalty')) is not None:
        generation_config['presence_penalty'] = presence_penalty
      if (frequency_penalty := model_settings.get('frequency_penalty')) is not None:
        generation_config['frequency_penalty'] = frequency_penalty
      if (gemini_safety_settings := model_settings.get('gemini_safety_settings')) != []:
        request_data['safety_settings'] = gemini_safety_settings
    if generation_config:
      request_data['generation_config'] = generation_config
    headers = {'Content-Type': 'application/json', 'User-Agent': get_user_agent()}
    url = f'/{self._model_name}:{"streamGenerateContent"ifstreamedelse"generateContent"}'
    request_json = _gemini_request_ta.dump_json(request_data, by_alias=True)
    async with self.client.stream(
      'POST',
      url,
      content=request_json,
      headers=headers,
      timeout=model_settings.get('timeout', USE_CLIENT_DEFAULT),
    ) as r:
      if (status_code := r.status_code) != 200:
        await r.aread()
        if status_code >= 400:
          raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=r.text)
        raise UnexpectedModelBehavior(f'Unexpected response from gemini {status_code}', r.text)
      yield r
  def_process_response(self, response: _GeminiResponse) -> ModelResponse:
    if len(response['candidates']) != 1:
      raise UnexpectedModelBehavior('Expected exactly one candidate in Gemini response')
    if 'content' not in response['candidates'][0]:
      if response['candidates'][0].get('finish_reason') == 'SAFETY':
        raise UnexpectedModelBehavior('Safety settings triggered', str(response))
      else:
        raise UnexpectedModelBehavior('Content field missing from Gemini response', str(response))
    parts = response['candidates'][0]['content']['parts']
    return _process_response_from_parts(parts, model_name=response.get('model_version', self._model_name))
  async def_process_streamed_response(self, http_response: HTTPResponse) -> StreamedResponse:
"""Process a streamed response, and prepare a streaming response to return."""
    aiter_bytes = http_response.aiter_bytes()
    start_response: _GeminiResponse | None = None
    content = bytearray()
    async for chunk in aiter_bytes:
      content.extend(chunk)
      responses = _gemini_streamed_response_ta.validate_json(
        _ensure_decodeable(content),
        experimental_allow_partial='trailing-strings',
      )
      if responses:
        last = responses[-1]
        if last['candidates'] and last['candidates'][0].get('content', {}).get('parts'):
          start_response = last
          break
    if start_response is None:
      raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')
    return GeminiStreamedResponse(_model_name=self._model_name, _content=content, _stream=aiter_bytes)
  @classmethod
  async def_message_to_gemini_content(
    cls, messages: list[ModelMessage]
  ) -> tuple[list[_GeminiTextPart], list[_GeminiContent]]:
    sys_prompt_parts: list[_GeminiTextPart] = []
    contents: list[_GeminiContent] = []
    for m in messages:
      if isinstance(m, ModelRequest):
        message_parts: list[_GeminiPartUnion] = []
        for part in m.parts:
          if isinstance(part, SystemPromptPart):
            sys_prompt_parts.append(_GeminiTextPart(text=part.content))
          elif isinstance(part, UserPromptPart):
            message_parts.extend(await cls._map_user_prompt(part))
          elif isinstance(part, ToolReturnPart):
            message_parts.append(_response_part_from_response(part.tool_name, part.model_response_object()))
          elif isinstance(part, RetryPromptPart):
            if part.tool_name is None:
              message_parts.append(_GeminiTextPart(text=part.model_response()))
            else:
              response = {'call_error': part.model_response()}
              message_parts.append(_response_part_from_response(part.tool_name, response))
          else:
            assert_never(part)
        if message_parts:
          contents.append(_GeminiContent(role='user', parts=message_parts))
      elif isinstance(m, ModelResponse):
        contents.append(_content_model_response(m))
      else:
        assert_never(m)
    return sys_prompt_parts, contents
  @staticmethod
  async def_map_user_prompt(part: UserPromptPart) -> list[_GeminiPartUnion]:
    if isinstance(part.content, str):
      return [{'text': part.content}]
    else:
      content: list[_GeminiPartUnion] = []
      for item in part.content:
        if isinstance(item, str):
          content.append({'text': item})
        elif isinstance(item, BinaryContent):
          base64_encoded = base64.b64encode(item.data).decode('utf-8')
          content.append(
            _GeminiInlineDataPart(inline_data={'data': base64_encoded, 'mime_type': item.media_type})
          )
        elif isinstance(item, (AudioUrl, ImageUrl, DocumentUrl)):
          client = cached_async_http_client()
          response = await client.get(item.url, follow_redirects=True)
          response.raise_for_status()
          mime_type = response.headers['Content-Type'].split(';')[0]
          inline_data = _GeminiInlineDataPart(
            inline_data={'data': base64.b64encode(response.content).decode('utf-8'), 'mime_type': mime_type}
          )
          content.append(inline_data)
        else:
          assert_never(item)
    return content

```
  
---|---  
####  __init__
```
__init__(
  model_name: ,
  *,
  provider: (
    ["google-gla", "google-vertex"]
    | [AsyncClient]
  ) = "google-gla"
)

```

Initialize a Gemini model.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`model_name` |  |  The name of the model to use. |  _required_  
`provider` |  `Literal[](https://docs.python.org/3/library/typing.html#typing.Literal "typing.Literal")['google-gla', 'google-vertex'] | Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncClient]` |  The provider to use for authentication and API access. Can be either the string 'google-gla' or 'google-vertex' or an instance of `Provider[httpx.AsyncClient]`. If not provided, a new provider will be created using the other parameters. |  `'google-gla'`  
Source code in `pydantic_ai_slim/pydantic_ai/models/gemini.py`
```
```
| ```
def__init__(
  self,
  model_name: GeminiModelName,
  *,
  provider: Literal['google-gla', 'google-vertex'] | Provider[httpx.AsyncClient] = 'google-gla',
):
"""Initialize a Gemini model.
  Args:
    model_name: The name of the model to use.
    provider: The provider to use for authentication and API access. Can be either the string
      'google-gla' or 'google-vertex' or an instance of `Provider[httpx.AsyncClient]`.
      If not provided, a new provider will be created using the other parameters.
  """
  self._model_name = model_name
  self._provider = provider
  if isinstance(provider, str):
    provider = infer_provider(provider)
  self._system = provider.name
  self.client = provider.client
  self._url = str(self.client.base_url)

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider.
###  AuthProtocol
Bases: 
Abstract definition for Gemini authentication.
Source code in `pydantic_ai_slim/pydantic_ai/models/gemini.py`
```
```
| ```
classAuthProtocol(Protocol):
"""Abstract definition for Gemini authentication."""
  async defheaders(self) -> dict[str, str]: ...

```
  
---|---  
###  ApiKeyAuth `dataclass`
Authentication using an API key for the `X-Goog-Api-Key` header.
Source code in `pydantic_ai_slim/pydantic_ai/models/gemini.py`
```
```
| ```
@dataclass
classApiKeyAuth:
"""Authentication using an API key for the `X-Goog-Api-Key` header."""
  api_key: str
  async defheaders(self) -> dict[str, str]:
    # https://cloud.google.com/docs/authentication/api-keys-use#using-with-rest
    return {'X-Goog-Api-Key': self.api_key}

```
  
---|---  
###  GeminiStreamedResponse `dataclass`
Bases: 
Implementation of `StreamedResponse` for the Gemini model.
Source code in `pydantic_ai_slim/pydantic_ai/models/gemini.py`
```
```
| ```
@dataclass
classGeminiStreamedResponse(StreamedResponse):
"""Implementation of `StreamedResponse` for the Gemini model."""
  _model_name: GeminiModelName
  _content: bytearray
  _stream: AsyncIterator[bytes]
  _timestamp: datetime = field(default_factory=_utils.now_utc, init=False)
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    async for gemini_response in self._get_gemini_responses():
      candidate = gemini_response['candidates'][0]
      if 'content' not in candidate:
        raise UnexpectedModelBehavior('Streamed response has no content field')
      gemini_part: _GeminiPartUnion
      for gemini_part in candidate['content']['parts']:
        if 'text' in gemini_part:
          # Using vendor_part_id=None means we can produce multiple text parts if their deltas are sprinkled
          # amongst the tool call deltas
          yield self._parts_manager.handle_text_delta(vendor_part_id=None, content=gemini_part['text'])
        elif 'function_call' in gemini_part:
          # Here, we assume all function_call parts are complete and don't have deltas.
          # We do this by assigning a unique randomly generated "vendor_part_id".
          # We need to confirm whether this is actually true, but if it isn't, we can still handle it properly
          # it would just be a bit more complicated. And we'd need to confirm the intended semantics.
          maybe_event = self._parts_manager.handle_tool_call_delta(
            vendor_part_id=uuid4(),
            tool_name=gemini_part['function_call']['name'],
            args=gemini_part['function_call']['args'],
            tool_call_id=None,
          )
          if maybe_event is not None:
            yield maybe_event
        else:
          assert 'function_response' in gemini_part, f'Unexpected part: {gemini_part}'
  async def_get_gemini_responses(self) -> AsyncIterator[_GeminiResponse]:
    # This method exists to ensure we only yield completed items, so we don't need to worry about
    # partial gemini responses, which would make everything more complicated
    gemini_responses: list[_GeminiResponse] = []
    current_gemini_response_index = 0
    # Right now, there are some circumstances where we will have information that could be yielded sooner than it is
    # But changing that would make things a lot more complicated.
    async for chunk in self._stream:
      self._content.extend(chunk)
      gemini_responses = _gemini_streamed_response_ta.validate_json(
        _ensure_decodeable(self._content),
        experimental_allow_partial='trailing-strings',
      )
      # The idea: yield only up to the latest response, which might still be partial.
      # Note that if the latest response is complete, we could yield it immediately, but there's not a good
      # allow_partial API to determine if the last item in the list is complete.
      responses_to_yield = gemini_responses[:-1]
      for r in responses_to_yield[current_gemini_response_index:]:
        current_gemini_response_index += 1
        self._usage += _metadata_as_usage(r)
        yield r
    # Now yield the final response, which should be complete
    if gemini_responses:
      r = gemini_responses[-1]
      self._usage += _metadata_as_usage(r)
      yield r
  @property
  defmodel_name(self) -> GeminiModelName:
"""Get the model name of the response."""
    return self._model_name
  @property
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    return self._timestamp

```
  
---|---  
####  model_name `property`
```
model_name: 

```

Get the model name of the response.
####  timestamp `property`
```
timestamp: 

```

Get the timestamp of the response.
###  GeminiSafetySettings
Bases: 
Safety settings options for Gemini model request.
See [Gemini API docs](https://ai.google.dev/gemini-api/docs/safety-settings) for safety category and threshold descriptions. For an example on how to use `GeminiSafetySettings`, see [here](https://ai.pydantic.dev/agents/#model-specific-settings).
Source code in `pydantic_ai_slim/pydantic_ai/models/gemini.py`
```
```
| ```
classGeminiSafetySettings(TypedDict):
"""Safety settings options for Gemini model request.
  See [Gemini API docs](https://ai.google.dev/gemini-api/docs/safety-settings) for safety category and threshold descriptions.
  For an example on how to use `GeminiSafetySettings`, see [here](../../agents.md#model-specific-settings).
  """
  category: Literal[
    'HARM_CATEGORY_UNSPECIFIED',
    'HARM_CATEGORY_HARASSMENT',
    'HARM_CATEGORY_HATE_SPEECH',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT',
    'HARM_CATEGORY_DANGEROUS_CONTENT',
    'HARM_CATEGORY_CIVIC_INTEGRITY',
  ]
"""
  Safety settings category.
  """
  threshold: Literal[
    'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
    'BLOCK_LOW_AND_ABOVE',
    'BLOCK_MEDIUM_AND_ABOVE',
    'BLOCK_ONLY_HIGH',
    'BLOCK_NONE',
    'OFF',
  ]
"""
  Safety settings threshold.
  """

```
  
---|---  
####  category `instance-attribute`
```
category: [
  "HARM_CATEGORY_UNSPECIFIED",
  "HARM_CATEGORY_HARASSMENT",
  "HARM_CATEGORY_HATE_SPEECH",
  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
  "HARM_CATEGORY_DANGEROUS_CONTENT",
  "HARM_CATEGORY_CIVIC_INTEGRITY",
]

```

Safety settings category.
####  threshold `instance-attribute`
```
threshold: [
  "HARM_BLOCK_THRESHOLD_UNSPECIFIED",
  "BLOCK_LOW_AND_ABOVE",
  "BLOCK_MEDIUM_AND_ABOVE",
  "BLOCK_ONLY_HIGH",
  "BLOCK_NONE",
  "OFF",
]

```

Safety settings threshold.

# `pydantic_ai.models.groq`
## Setup
For details on how to set up authentication with this model, see [model configuration for Groq](https://ai.pydantic.dev/models/#groq).
###  LatestGroqModelNames `module-attribute`
```
LatestGroqModelNames = [
  "llama-3.3-70b-versatile",
  "llama-3.3-70b-specdec",
  "llama-3.1-8b-instant",
  "llama-3.2-1b-preview",
  "llama-3.2-3b-preview",
  "llama-3.2-11b-vision-preview",
  "llama-3.2-90b-vision-preview",
  "llama3-70b-8192",
  "llama3-8b-8192",
  "mixtral-8x7b-32768",
  "gemma2-9b-it",
]

```

Latest Groq models.
###  GroqModelName `module-attribute`
```
GroqModelName = [, ]

```

Possible Groq model names.
Since Groq supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the Groq docs](https://console.groq.com/docs/models) for a full list.
###  GroqModelSettings
Bases: 
Settings used for a Groq model request.
Source code in `pydantic_ai_slim/pydantic_ai/models/groq.py`
```
```
| ```
classGroqModelSettings(ModelSettings):
"""Settings used for a Groq model request."""

```
  
---|---  
###  GroqModel `dataclass`
Bases: 
A model that uses the Groq API.
Internally, this uses the [Groq Python client](https://github.com/groq/groq-python) to interact with the API.
Apart from `__init__`, all methods are private or match those of the base class.
Source code in `pydantic_ai_slim/pydantic_ai/models/groq.py`
```
```
| ```
@dataclass(init=False)
classGroqModel(Model):
"""A model that uses the Groq API.
  Internally, this uses the [Groq Python client](https://github.com/groq/groq-python) to interact with the API.
  Apart from `__init__`, all methods are private or match those of the base class.
  """
  client: AsyncGroq = field(repr=False)
  _model_name: GroqModelName = field(repr=False)
  _system: str = field(default='groq', repr=False)
  def__init__(self, model_name: GroqModelName, *, provider: Literal['groq'] | Provider[AsyncGroq] = 'groq'):
"""Initialize a Groq model.
    Args:
      model_name: The name of the Groq model to use. List of model names available
        [here](https://console.groq.com/docs/models).
      provider: The provider to use for authentication and API access. Can be either the string
        'groq' or an instance of `Provider[AsyncGroq]`. If not provided, a new provider will be
        created using the other parameters.
    """
    self._model_name = model_name
    if isinstance(provider, str):
      provider = infer_provider(provider)
    self.client = provider.client
  @property
  defbase_url(self) -> str:
    return str(self.client.base_url)
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, usage.Usage]:
    check_allow_model_requests()
    response = await self._completions_create(
      messages, False, cast(GroqModelSettings, model_settings or {}), model_request_parameters
    )
    return self._process_response(response), _map_usage(response)
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
    check_allow_model_requests()
    response = await self._completions_create(
      messages, True, cast(GroqModelSettings, model_settings or {}), model_request_parameters
    )
    async with response:
      yield await self._process_streamed_response(response)
  @property
  defmodel_name(self) -> GroqModelName:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider."""
    return self._system
  @overload
  async def_completions_create(
    self,
    messages: list[ModelMessage],
    stream: Literal[True],
    model_settings: GroqModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncStream[chat.ChatCompletionChunk]:
    pass
  @overload
  async def_completions_create(
    self,
    messages: list[ModelMessage],
    stream: Literal[False],
    model_settings: GroqModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> chat.ChatCompletion:
    pass
  async def_completions_create(
    self,
    messages: list[ModelMessage],
    stream: bool,
    model_settings: GroqModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> chat.ChatCompletion | AsyncStream[chat.ChatCompletionChunk]:
    tools = self._get_tools(model_request_parameters)
    # standalone function to make it easier to override
    if not tools:
      tool_choice: Literal['none', 'required', 'auto'] | None = None
    elif not model_request_parameters.allow_text_result:
      tool_choice = 'required'
    else:
      tool_choice = 'auto'
    groq_messages = list(chain(*(self._map_message(m) for m in messages)))
    try:
      return await self.client.chat.completions.create(
        model=str(self._model_name),
        messages=groq_messages,
        n=1,
        parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
        tools=tools or NOT_GIVEN,
        tool_choice=tool_choice or NOT_GIVEN,
        stream=stream,
        max_tokens=model_settings.get('max_tokens', NOT_GIVEN),
        temperature=model_settings.get('temperature', NOT_GIVEN),
        top_p=model_settings.get('top_p', NOT_GIVEN),
        timeout=model_settings.get('timeout', NOT_GIVEN),
        seed=model_settings.get('seed', NOT_GIVEN),
        presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
        frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
        logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
      )
    except APIStatusError as e:
      if (status_code := e.status_code) >= 400:
        raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) frome
      raise
  def_process_response(self, response: chat.ChatCompletion) -> ModelResponse:
"""Process a non-streamed response, and prepare a message to return."""
    timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
    choice = response.choices[0]
    items: list[ModelResponsePart] = []
    if choice.message.content is not None:
      items.append(TextPart(content=choice.message.content))
    if choice.message.tool_calls is not None:
      for c in choice.message.tool_calls:
        items.append(ToolCallPart(tool_name=c.function.name, args=c.function.arguments, tool_call_id=c.id))
    return ModelResponse(items, model_name=response.model, timestamp=timestamp)
  async def_process_streamed_response(self, response: AsyncStream[chat.ChatCompletionChunk]) -> GroqStreamedResponse:
"""Process a streamed response, and prepare a streaming response to return."""
    peekable_response = _utils.PeekableAsyncStream(response)
    first_chunk = await peekable_response.peek()
    if isinstance(first_chunk, _utils.Unset):
      raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')
    return GroqStreamedResponse(
      _response=peekable_response,
      _model_name=self._model_name,
      _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
    )
  def_get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
    tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
    if model_request_parameters.result_tools:
      tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
    return tools
  def_map_message(self, message: ModelMessage) -> Iterable[chat.ChatCompletionMessageParam]:
"""Just maps a `pydantic_ai.Message` to a `groq.types.ChatCompletionMessageParam`."""
    if isinstance(message, ModelRequest):
      yield from self._map_user_message(message)
    elif isinstance(message, ModelResponse):
      texts: list[str] = []
      tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
      for item in message.parts:
        if isinstance(item, TextPart):
          texts.append(item.content)
        elif isinstance(item, ToolCallPart):
          tool_calls.append(self._map_tool_call(item))
        else:
          assert_never(item)
      message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
      if texts:
        # Note: model responses from this model should only have one text item, so the following
        # shouldn't merge multiple texts into one unless you switch models between runs:
        message_param['content'] = '\n\n'.join(texts)
      if tool_calls:
        message_param['tool_calls'] = tool_calls
      yield message_param
    else:
      assert_never(message)
  @staticmethod
  def_map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
    return chat.ChatCompletionMessageToolCallParam(
      id=_guard_tool_call_id(t=t),
      type='function',
      function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
    )
  @staticmethod
  def_map_tool_definition(f: ToolDefinition) -> chat.ChatCompletionToolParam:
    return {
      'type': 'function',
      'function': {
        'name': f.name,
        'description': f.description,
        'parameters': f.parameters_json_schema,
      },
    }
  @classmethod
  def_map_user_message(cls, message: ModelRequest) -> Iterable[chat.ChatCompletionMessageParam]:
    for part in message.parts:
      if isinstance(part, SystemPromptPart):
        yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
      elif isinstance(part, UserPromptPart):
        yield cls._map_user_prompt(part)
      elif isinstance(part, ToolReturnPart):
        yield chat.ChatCompletionToolMessageParam(
          role='tool',
          tool_call_id=_guard_tool_call_id(t=part),
          content=part.model_response_str(),
        )
      elif isinstance(part, RetryPromptPart):
        if part.tool_name is None:
          yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
        else:
          yield chat.ChatCompletionToolMessageParam(
            role='tool',
            tool_call_id=_guard_tool_call_id(t=part),
            content=part.model_response(),
          )
  @staticmethod
  def_map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
    content: str | list[chat.ChatCompletionContentPartParam]
    if isinstance(part.content, str):
      content = part.content
    else:
      content = []
      for item in part.content:
        if isinstance(item, str):
          content.append(chat.ChatCompletionContentPartTextParam(text=item, type='text'))
        elif isinstance(item, ImageUrl):
          image_url = ImageURL(url=item.url)
          content.append(chat.ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
        elif isinstance(item, BinaryContent):
          base64_encoded = base64.b64encode(item.data).decode('utf-8')
          if item.is_image:
            image_url = ImageURL(url=f'data:{item.media_type};base64,{base64_encoded}')
            content.append(chat.ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
          else:
            raise RuntimeError('Only images are supported for binary content in Groq.')
        elif isinstance(item, DocumentUrl): # pragma: no cover
          raise RuntimeError('DocumentUrl is not supported in Groq.')
        else: # pragma: no cover
          raise RuntimeError(f'Unsupported content type: {type(item)}')
    return chat.ChatCompletionUserMessageParam(role='user', content=content)

```
  
---|---  
####  __init__
```
__init__(
  model_name: ,
  *,
  provider: ["groq"] | [AsyncGroq] = "groq"
)

```

Initialize a Groq model.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`model_name` |  |  The name of the Groq model to use. List of model names available [here](https://console.groq.com/docs/models). |  _required_  
`provider` |  `Literal[](https://docs.python.org/3/library/typing.html#typing.Literal "typing.Literal")['groq'] | Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncGroq]` |  The provider to use for authentication and API access. Can be either the string 'groq' or an instance of `Provider[AsyncGroq]`. If not provided, a new provider will be created using the other parameters. |  `'groq'`  
Source code in `pydantic_ai_slim/pydantic_ai/models/groq.py`
```
```
| ```
def__init__(self, model_name: GroqModelName, *, provider: Literal['groq'] | Provider[AsyncGroq] = 'groq'):
"""Initialize a Groq model.
  Args:
    model_name: The name of the Groq model to use. List of model names available
      [here](https://console.groq.com/docs/models).
    provider: The provider to use for authentication and API access. Can be either the string
      'groq' or an instance of `Provider[AsyncGroq]`. If not provided, a new provider will be
      created using the other parameters.
  """
  self._model_name = model_name
  if isinstance(provider, str):
    provider = infer_provider(provider)
  self.client = provider.client

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider.
###  GroqStreamedResponse `dataclass`
Bases: 
Implementation of `StreamedResponse` for Groq models.
Source code in `pydantic_ai_slim/pydantic_ai/models/groq.py`
```
```
| ```
@dataclass
classGroqStreamedResponse(StreamedResponse):
"""Implementation of `StreamedResponse` for Groq models."""
  _model_name: GroqModelName
  _response: AsyncIterable[chat.ChatCompletionChunk]
  _timestamp: datetime
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    async for chunk in self._response:
      self._usage += _map_usage(chunk)
      try:
        choice = chunk.choices[0]
      except IndexError:
        continue
      # Handle the text part of the response
      content = choice.delta.content
      if content is not None:
        yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=content)
      # Handle the tool calls
      for dtc in choice.delta.tool_calls or []:
        maybe_event = self._parts_manager.handle_tool_call_delta(
          vendor_part_id=dtc.index,
          tool_name=dtc.function and dtc.function.name,
          args=dtc.function and dtc.function.arguments,
          tool_call_id=dtc.id,
        )
        if maybe_event is not None:
          yield maybe_event
  @property
  defmodel_name(self) -> GroqModelName:
"""Get the model name of the response."""
    return self._model_name
  @property
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    return self._timestamp

```
  
---|---  
####  model_name `property`
```
model_name: 

```

Get the model name of the response.
####  timestamp `property`
```
timestamp: 

```

Get the timestamp of the response.

# `pydantic_ai.models.mistral`
## Setup
For details on how to set up authentication with this model, see [model configuration for Mistral](https://ai.pydantic.dev/models/#mistral).
###  LatestMistralModelNames `module-attribute`
```
LatestMistralModelNames = [
  "mistral-large-latest",
  "mistral-small-latest",
  "codestral-latest",
  "mistral-moderation-latest",
]

```

Latest Mistral models.
###  MistralModelName `module-attribute`
```
MistralModelName = [, ]

```

Possible Mistral model names.
Since Mistral supports a variety of date-stamped models, we explicitly list the most popular models but allow any name in the type hints. Since [the Mistral docs](https://docs.mistral.ai/getting-started/models/models_overview/) for a full list.
###  MistralModelSettings
Bases: 
Settings used for a Mistral model request.
Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`
```
```
| ```
classMistralModelSettings(ModelSettings):
"""Settings used for a Mistral model request."""

```
  
---|---  
###  MistralModel `dataclass`
Bases: 
A model that uses Mistral.
Internally, this uses the [Mistral Python client](https://github.com/mistralai/client-python) to interact with the API.
[API Documentation](https://docs.mistral.ai/)
Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`
```
```
| ```
@dataclass(init=False)
classMistralModel(Model):
"""A model that uses Mistral.
  Internally, this uses the [Mistral Python client](https://github.com/mistralai/client-python) to interact with the API.
  [API Documentation](https://docs.mistral.ai/)
  """
  client: Mistral = field(repr=False)
  json_mode_schema_prompt: str = """Answer in JSON Object, respect the format:\n```\n{schema}\n```\n"""
  _model_name: MistralModelName = field(repr=False)
  _system: str = field(default='mistral_ai', repr=False)
  def__init__(
    self,
    model_name: MistralModelName,
    *,
    provider: Literal['mistral'] | Provider[Mistral] = 'mistral',
    json_mode_schema_prompt: str = """Answer in JSON Object, respect the format:\n```\n{schema}\n```\n""",
  ):
"""Initialize a Mistral model.
    Args:
      model_name: The name of the model to use.
      provider: The provider to use for authentication and API access. Can be either the string
        'mistral' or an instance of `Provider[Mistral]`. If not provided, a new provider will be
        created using the other parameters.
      json_mode_schema_prompt: The prompt to show when the model expects a JSON object as input.
    """
    self._model_name = model_name
    self.json_mode_schema_prompt = json_mode_schema_prompt
    if isinstance(provider, str):
      provider = infer_provider(provider)
    self.client = provider.client
  @property
  defbase_url(self) -> str:
    return self.client.sdk_configuration.get_server_details()[0]
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, Usage]:
"""Make a non-streaming request to the model from Pydantic AI call."""
    check_allow_model_requests()
    response = await self._completions_create(
      messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
    )
    return self._process_response(response), _map_usage(response)
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
"""Make a streaming request to the model from Pydantic AI call."""
    check_allow_model_requests()
    response = await self._stream_completions_create(
      messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
    )
    async with response:
      yield await self._process_streamed_response(model_request_parameters.result_tools, response)
  @property
  defmodel_name(self) -> MistralModelName:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider."""
    return self._system
  async def_completions_create(
    self,
    messages: list[ModelMessage],
    model_settings: MistralModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> MistralChatCompletionResponse:
"""Make a non-streaming request to the model."""
    try:
      response = await self.client.chat.complete_async(
        model=str(self._model_name),
        messages=list(chain(*(self._map_message(m) for m in messages))),
        n=1,
        tools=self._map_function_and_result_tools_definition(model_request_parameters) or UNSET,
        tool_choice=self._get_tool_choice(model_request_parameters),
        stream=False,
        max_tokens=model_settings.get('max_tokens', UNSET),
        temperature=model_settings.get('temperature', UNSET),
        top_p=model_settings.get('top_p', 1),
        timeout_ms=self._get_timeout_ms(model_settings.get('timeout')),
        random_seed=model_settings.get('seed', UNSET),
      )
    except SDKError as e:
      if (status_code := e.status_code) >= 400:
        raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) frome
      raise
    assert response, 'A unexpected empty response from Mistral.'
    return response
  async def_stream_completions_create(
    self,
    messages: list[ModelMessage],
    model_settings: MistralModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> MistralEventStreamAsync[MistralCompletionEvent]:
"""Create a streaming completion request to the Mistral model."""
    response: MistralEventStreamAsync[MistralCompletionEvent] | None
    mistral_messages = list(chain(*(self._map_message(m) for m in messages)))
    if (
      model_request_parameters.result_tools
      and model_request_parameters.function_tools
      or model_request_parameters.function_tools
    ):
      # Function Calling
      response = await self.client.chat.stream_async(
        model=str(self._model_name),
        messages=mistral_messages,
        n=1,
        tools=self._map_function_and_result_tools_definition(model_request_parameters) or UNSET,
        tool_choice=self._get_tool_choice(model_request_parameters),
        temperature=model_settings.get('temperature', UNSET),
        top_p=model_settings.get('top_p', 1),
        max_tokens=model_settings.get('max_tokens', UNSET),
        timeout_ms=self._get_timeout_ms(model_settings.get('timeout')),
        presence_penalty=model_settings.get('presence_penalty'),
        frequency_penalty=model_settings.get('frequency_penalty'),
      )
    elif model_request_parameters.result_tools:
      # Json Mode
      parameters_json_schemas = [tool.parameters_json_schema for tool in model_request_parameters.result_tools]
      user_output_format_message = self._generate_user_output_format(parameters_json_schemas)
      mistral_messages.append(user_output_format_message)
      response = await self.client.chat.stream_async(
        model=str(self._model_name),
        messages=mistral_messages,
        response_format={'type': 'json_object'},
        stream=True,
      )
    else:
      # Stream Mode
      response = await self.client.chat.stream_async(
        model=str(self._model_name),
        messages=mistral_messages,
        stream=True,
      )
    assert response, 'A unexpected empty response from Mistral.'
    return response
  def_get_tool_choice(self, model_request_parameters: ModelRequestParameters) -> MistralToolChoiceEnum | None:
"""Get tool choice for the model.
    - "auto": Default mode. Model decides if it uses the tool or not.
    - "any": Select any tool.
    - "none": Prevents tool use.
    - "required": Forces tool use.
    """
    if not model_request_parameters.function_tools and not model_request_parameters.result_tools:
      return None
    elif not model_request_parameters.allow_text_result:
      return 'required'
    else:
      return 'auto'
  def_map_function_and_result_tools_definition(
    self, model_request_parameters: ModelRequestParameters
  ) -> list[MistralTool] | None:
"""Map function and result tools to MistralTool format.
    Returns None if both function_tools and result_tools are empty.
    """
    all_tools: list[ToolDefinition] = (
      model_request_parameters.function_tools + model_request_parameters.result_tools
    )
    tools = [
      MistralTool(
        function=MistralFunction(name=r.name, parameters=r.parameters_json_schema, description=r.description)
      )
      for r in all_tools
    ]
    return tools if tools else None
  def_process_response(self, response: MistralChatCompletionResponse) -> ModelResponse:
"""Process a non-streamed response, and prepare a message to return."""
    assert response.choices, 'Unexpected empty response choice.'
    if response.created:
      timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
    else:
      timestamp = _now_utc()
    choice = response.choices[0]
    content = choice.message.content
    tool_calls = choice.message.tool_calls
    parts: list[ModelResponsePart] = []
    if text := _map_content(content):
      parts.append(TextPart(content=text))
    if isinstance(tool_calls, list):
      for tool_call in tool_calls:
        tool = self._map_mistral_to_pydantic_tool_call(tool_call=tool_call)
        parts.append(tool)
    return ModelResponse(parts, model_name=response.model, timestamp=timestamp)
  async def_process_streamed_response(
    self,
    result_tools: list[ToolDefinition],
    response: MistralEventStreamAsync[MistralCompletionEvent],
  ) -> StreamedResponse:
"""Process a streamed response, and prepare a streaming response to return."""
    peekable_response = _utils.PeekableAsyncStream(response)
    first_chunk = await peekable_response.peek()
    if isinstance(first_chunk, _utils.Unset):
      raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')
    if first_chunk.data.created:
      timestamp = datetime.fromtimestamp(first_chunk.data.created, tz=timezone.utc)
    else:
      timestamp = datetime.now(tz=timezone.utc)
    return MistralStreamedResponse(
      _response=peekable_response,
      _model_name=self._model_name,
      _timestamp=timestamp,
      _result_tools={c.name: c for c in result_tools},
    )
  @staticmethod
  def_map_mistral_to_pydantic_tool_call(tool_call: MistralToolCall) -> ToolCallPart:
"""Maps a MistralToolCall to a ToolCall."""
    tool_call_id = tool_call.id or _generate_tool_call_id()
    func_call = tool_call.function
    return ToolCallPart(func_call.name, func_call.arguments, tool_call_id)
  @staticmethod
  def_map_tool_call(t: ToolCallPart) -> MistralToolCall:
"""Maps a pydantic-ai ToolCall to a MistralToolCall."""
    return MistralToolCall(
      id=_utils.guard_tool_call_id(t=t),
      type='function',
      function=MistralFunctionCall(name=t.tool_name, arguments=t.args),
    )
  def_generate_user_output_format(self, schemas: list[dict[str, Any]]) -> MistralUserMessage:
"""Get a message with an example of the expected output format."""
    examples: list[dict[str, Any]] = []
    for schema in schemas:
      typed_dict_definition: dict[str, Any] = {}
      for key, value in schema.get('properties', {}).items():
        typed_dict_definition[key] = self._get_python_type(value)
      examples.append(typed_dict_definition)
    example_schema = examples[0] if len(examples) == 1 else examples
    return MistralUserMessage(content=self.json_mode_schema_prompt.format(schema=example_schema))
  @classmethod
  def_get_python_type(cls, value: dict[str, Any]) -> str:
"""Return a string representation of the Python type for a single JSON schema property.
    This function handles recursion for nested arrays/objects and `anyOf`.
    """
    # 1) Handle anyOf first, because it's a different schema structure
    if any_of := value.get('anyOf'):
      # Simplistic approach: pick the first option in anyOf
      # (In reality, you'd possibly want to merge or union types)
      return f'Optional[{cls._get_python_type(any_of[0])}]'
    # 2) If we have a top-level "type" field
    value_type = value.get('type')
    if not value_type:
      # No explicit type; fallback
      return 'Any'
    # 3) Direct simple type mapping (string, integer, float, bool, None)
    if value_type in SIMPLE_JSON_TYPE_MAPPING and value_type != 'array' and value_type != 'object':
      return SIMPLE_JSON_TYPE_MAPPING[value_type]
    # 4) Array: Recursively get the item type
    if value_type == 'array':
      items = value.get('items', {})
      return f'list[{cls._get_python_type(items)}]'
    # 5) Object: Check for additionalProperties
    if value_type == 'object':
      additional_properties = value.get('additionalProperties', {})
      additional_properties_type = additional_properties.get('type')
      if (
        additional_properties_type in SIMPLE_JSON_TYPE_MAPPING
        and additional_properties_type != 'array'
        and additional_properties_type != 'object'
      ):
        # dict[str, bool/int/float/etc...]
        return f'dict[str, {SIMPLE_JSON_TYPE_MAPPING[additional_properties_type]}]'
      elif additional_properties_type == 'array':
        array_items = additional_properties.get('items', {})
        return f'dict[str, list[{cls._get_python_type(array_items)}]]'
      elif additional_properties_type == 'object':
        # nested dictionary of unknown shape
        return 'dict[str, dict[str, Any]]'
      else:
        # If no additionalProperties type or something else, default to a generic dict
        return 'dict[str, Any]'
    # 6) Fallback
    return 'Any'
  @staticmethod
  def_get_timeout_ms(timeout: Timeout | float | None) -> int | None:
"""Convert a timeout to milliseconds."""
    if timeout is None:
      return None
    if isinstance(timeout, float):
      return int(1000 * timeout)
    raise NotImplementedError('Timeout object is not yet supported for MistralModel.')
  @classmethod
  def_map_user_message(cls, message: ModelRequest) -> Iterable[MistralMessages]:
    for part in message.parts:
      if isinstance(part, SystemPromptPart):
        yield MistralSystemMessage(content=part.content)
      elif isinstance(part, UserPromptPart):
        yield cls._map_user_prompt(part)
      elif isinstance(part, ToolReturnPart):
        yield MistralToolMessage(
          tool_call_id=part.tool_call_id,
          content=part.model_response_str(),
        )
      elif isinstance(part, RetryPromptPart):
        if part.tool_name is None:
          yield MistralUserMessage(content=part.model_response())
        else:
          yield MistralToolMessage(
            tool_call_id=part.tool_call_id,
            content=part.model_response(),
          )
      else:
        assert_never(part)
  @classmethod
  def_map_message(cls, message: ModelMessage) -> Iterable[MistralMessages]:
"""Just maps a `pydantic_ai.Message` to a `MistralMessage`."""
    if isinstance(message, ModelRequest):
      yield from cls._map_user_message(message)
    elif isinstance(message, ModelResponse):
      content_chunks: list[MistralContentChunk] = []
      tool_calls: list[MistralToolCall] = []
      for part in message.parts:
        if isinstance(part, TextPart):
          content_chunks.append(MistralTextChunk(text=part.content))
        elif isinstance(part, ToolCallPart):
          tool_calls.append(cls._map_tool_call(part))
        else:
          assert_never(part)
      yield MistralAssistantMessage(content=content_chunks, tool_calls=tool_calls)
    else:
      assert_never(message)
  @staticmethod
  def_map_user_prompt(part: UserPromptPart) -> MistralUserMessage:
    content: str | list[MistralContentChunk]
    if isinstance(part.content, str):
      content = part.content
    else:
      content = []
      for item in part.content:
        if isinstance(item, str):
          content.append(MistralTextChunk(text=item))
        elif isinstance(item, ImageUrl):
          content.append(MistralImageURLChunk(image_url=MistralImageURL(url=item.url)))
        elif isinstance(item, BinaryContent):
          base64_encoded = base64.b64encode(item.data).decode('utf-8')
          if item.is_image:
            image_url = MistralImageURL(url=f'data:{item.media_type};base64,{base64_encoded}')
            content.append(MistralImageURLChunk(image_url=image_url, type='image_url'))
          else:
            raise RuntimeError('Only image binary content is supported for Mistral.')
        elif isinstance(item, DocumentUrl):
          raise RuntimeError('DocumentUrl is not supported in Mistral.')
        else: # pragma: no cover
          raise RuntimeError(f'Unsupported content type: {type(item)}')
    return MistralUserMessage(content=content)

```
  
---|---  
####  __init__
```
__init__(
  model_name: ,
  *,
  provider: (
    ["mistral"] | [Mistral]
  ) = "mistral",
  json_mode_schema_prompt:  = "Answer in JSON Object, respect the format:\n```\n{schema}\n```\n"
)

```

Initialize a Mistral model.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`model_name` |  |  The name of the model to use. |  _required_  
`provider` |  `Literal[](https://docs.python.org/3/library/typing.html#typing.Literal "typing.Literal")['mistral'] | Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[Mistral]` |  The provider to use for authentication and API access. Can be either the string 'mistral' or an instance of `Provider[Mistral]`. If not provided, a new provider will be created using the other parameters. |  `'mistral'`  
`json_mode_schema_prompt` |  The prompt to show when the model expects a JSON object as input. |  `'Answer in JSON Object, respect the format:\n```\n{schema}\n```\n'`  
Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`
```
```
| ```
def__init__(
  self,
  model_name: MistralModelName,
  *,
  provider: Literal['mistral'] | Provider[Mistral] = 'mistral',
  json_mode_schema_prompt: str = """Answer in JSON Object, respect the format:\n```\n{schema}\n```\n""",
):
"""Initialize a Mistral model.
  Args:
    model_name: The name of the model to use.
    provider: The provider to use for authentication and API access. Can be either the string
      'mistral' or an instance of `Provider[Mistral]`. If not provided, a new provider will be
      created using the other parameters.
    json_mode_schema_prompt: The prompt to show when the model expects a JSON object as input.
  """
  self._model_name = model_name
  self.json_mode_schema_prompt = json_mode_schema_prompt
  if isinstance(provider, str):
    provider = infer_provider(provider)
  self.client = provider.client

```
  
---|---  
####  request `async`
```
request(
  messages: [],
  model_settings:  | None,
  model_request_parameters: ,
) -> [, ]

```

Make a non-streaming request to the model from Pydantic AI call.
Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`
```
```
| ```
async defrequest(
  self,
  messages: list[ModelMessage],
  model_settings: ModelSettings | None,
  model_request_parameters: ModelRequestParameters,
) -> tuple[ModelResponse, Usage]:
"""Make a non-streaming request to the model from Pydantic AI call."""
  check_allow_model_requests()
  response = await self._completions_create(
    messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
  )
  return self._process_response(response), _map_usage(response)

```
  
---|---  
####  request_stream `async`
```
request_stream(
  messages: [],
  model_settings:  | None,
  model_request_parameters: ,
) -> []

```

Make a streaming request to the model from Pydantic AI call.
Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`
```
```
| ```
@asynccontextmanager
async defrequest_stream(
  self,
  messages: list[ModelMessage],
  model_settings: ModelSettings | None,
  model_request_parameters: ModelRequestParameters,
) -> AsyncIterator[StreamedResponse]:
"""Make a streaming request to the model from Pydantic AI call."""
  check_allow_model_requests()
  response = await self._stream_completions_create(
    messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
  )
  async with response:
    yield await self._process_streamed_response(model_request_parameters.result_tools, response)

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider.
###  MistralStreamedResponse `dataclass`
Bases: 
Implementation of `StreamedResponse` for Mistral models.
Source code in `pydantic_ai_slim/pydantic_ai/models/mistral.py`
```
```
| ```
@dataclass
classMistralStreamedResponse(StreamedResponse):
"""Implementation of `StreamedResponse` for Mistral models."""
  _model_name: MistralModelName
  _response: AsyncIterable[MistralCompletionEvent]
  _timestamp: datetime
  _result_tools: dict[str, ToolDefinition]
  _delta_content: str = field(default='', init=False)
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    chunk: MistralCompletionEvent
    async for chunk in self._response:
      self._usage += _map_usage(chunk.data)
      try:
        choice = chunk.data.choices[0]
      except IndexError:
        continue
      # Handle the text part of the response
      content = choice.delta.content
      text = _map_content(content)
      if text:
        # Attempt to produce a result tool call from the received text
        if self._result_tools:
          self._delta_content += text
          maybe_tool_call_part = self._try_get_result_tool_from_text(self._delta_content, self._result_tools)
          if maybe_tool_call_part:
            yield self._parts_manager.handle_tool_call_part(
              vendor_part_id='result',
              tool_name=maybe_tool_call_part.tool_name,
              args=maybe_tool_call_part.args_as_dict(),
              tool_call_id=maybe_tool_call_part.tool_call_id,
            )
        else:
          yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=text)
      # Handle the explicit tool calls
      for index, dtc in enumerate(choice.delta.tool_calls or []):
        # It seems that mistral just sends full tool calls, so we just use them directly, rather than building
        yield self._parts_manager.handle_tool_call_part(
          vendor_part_id=index, tool_name=dtc.function.name, args=dtc.function.arguments, tool_call_id=dtc.id
        )
  @property
  defmodel_name(self) -> MistralModelName:
"""Get the model name of the response."""
    return self._model_name
  @property
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    return self._timestamp
  @staticmethod
  def_try_get_result_tool_from_text(text: str, result_tools: dict[str, ToolDefinition]) -> ToolCallPart | None:
    output_json: dict[str, Any] | None = pydantic_core.from_json(text, allow_partial='trailing-strings')
    if output_json:
      for result_tool in result_tools.values():
        # NOTE: Additional verification to prevent JSON validation to crash in `_result.py`
        # Ensures required parameters in the JSON schema are respected, especially for stream-based return types.
        # Example with BaseModel and required fields.
        if not MistralStreamedResponse._validate_required_json_schema(
          output_json, result_tool.parameters_json_schema
        ):
          continue
        # The following part_id will be thrown away
        return ToolCallPart(tool_name=result_tool.name, args=output_json)
  @staticmethod
  def_validate_required_json_schema(json_dict: dict[str, Any], json_schema: dict[str, Any]) -> bool:
"""Validate that all required parameters in the JSON schema are present in the JSON dictionary."""
    required_params = json_schema.get('required', [])
    properties = json_schema.get('properties', {})
    for param in required_params:
      if param not in json_dict:
        return False
      param_schema = properties.get(param, {})
      param_type = param_schema.get('type')
      param_items_type = param_schema.get('items', {}).get('type')
      if param_type == 'array' and param_items_type:
        if not isinstance(json_dict[param], list):
          return False
        for item in json_dict[param]:
          if not isinstance(item, VALID_JSON_TYPE_MAPPING[param_items_type]):
            return False
      elif param_type and not isinstance(json_dict[param], VALID_JSON_TYPE_MAPPING[param_type]):
        return False
      if isinstance(json_dict[param], dict) and 'properties' in param_schema:
        nested_schema = param_schema
        if not MistralStreamedResponse._validate_required_json_schema(json_dict[param], nested_schema):
          return False
    return True

```
  
---|---  
####  model_name `property`
```
model_name: 

```

Get the model name of the response.
####  timestamp `property`
```
timestamp: 

```

Get the timestamp of the response.

# `pydantic_ai.models.openai`
## Setup
For details on how to set up authentication with this model, see [model configuration for OpenAI](https://ai.pydantic.dev/models/#openai).
###  OpenAIModelName `module-attribute`
```
OpenAIModelName = [, ChatModel]

```

Possible OpenAI model names.
Since OpenAI supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints. See [the OpenAI docs](https://platform.openai.com/docs/models) for a full list.
Using this more broad type for the model name instead of the ChatModel definition allows this model to be used more easily with other model types (ie, Ollama, Deepseek).
###  OpenAIModelSettings
Bases: 
Settings used for an OpenAI model request.
Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`
```
```
| ```
classOpenAIModelSettings(ModelSettings, total=False):
"""Settings used for an OpenAI model request."""
  openai_reasoning_effort: chat.ChatCompletionReasoningEffort
"""
  Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).
  Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
  result in faster responses and fewer tokens used on reasoning in a response.
  """
  user: str
"""A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.
  See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
  """

```
  
---|---  
####  openai_reasoning_effort `instance-attribute`
```
openai_reasoning_effort: ChatCompletionReasoningEffort

```

Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning). Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.
####  user `instance-attribute`
```
user: 

```

A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.
See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
###  OpenAIModel `dataclass`
Bases: 
A model that uses the OpenAI API.
Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.
Apart from `__init__`, all methods are private or match those of the base class.
Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`
```
```
| ```
@dataclass(init=False)
classOpenAIModel(Model):
"""A model that uses the OpenAI API.
  Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.
  Apart from `__init__`, all methods are private or match those of the base class.
  """
  client: AsyncOpenAI = field(repr=False)
  system_prompt_role: OpenAISystemPromptRole | None = field(default=None)
  _model_name: OpenAIModelName = field(repr=False)
  _system: str = field(default='openai', repr=False)
  def__init__(
    self,
    model_name: OpenAIModelName,
    *,
    provider: Literal['openai', 'deepseek', 'azure'] | Provider[AsyncOpenAI] = 'openai',
    system_prompt_role: OpenAISystemPromptRole | None = None,
  ):
"""Initialize an OpenAI model.
    Args:
      model_name: The name of the OpenAI model to use. List of model names available
        [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
        (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
      provider: The provider to use. Defaults to `'openai'`.
      system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
        In the future, this may be inferred from the model name.
    """
    self._model_name = model_name
    if isinstance(provider, str):
      provider = infer_provider(provider)
    self.client = provider.client
    self.system_prompt_role = system_prompt_role
  @property
  defbase_url(self) -> str:
    return str(self.client.base_url)
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, usage.Usage]:
    check_allow_model_requests()
    response = await self._completions_create(
      messages, False, cast(OpenAIModelSettings, model_settings or {}), model_request_parameters
    )
    return self._process_response(response), _map_usage(response)
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
    check_allow_model_requests()
    response = await self._completions_create(
      messages, True, cast(OpenAIModelSettings, model_settings or {}), model_request_parameters
    )
    async with response:
      yield await self._process_streamed_response(response)
  @property
  defmodel_name(self) -> OpenAIModelName:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider."""
    return self._system
  @overload
  async def_completions_create(
    self,
    messages: list[ModelMessage],
    stream: Literal[True],
    model_settings: OpenAIModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncStream[ChatCompletionChunk]:
    pass
  @overload
  async def_completions_create(
    self,
    messages: list[ModelMessage],
    stream: Literal[False],
    model_settings: OpenAIModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> chat.ChatCompletion:
    pass
  async def_completions_create(
    self,
    messages: list[ModelMessage],
    stream: bool,
    model_settings: OpenAIModelSettings,
    model_request_parameters: ModelRequestParameters,
  ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
    tools = self._get_tools(model_request_parameters)
    # standalone function to make it easier to override
    if not tools:
      tool_choice: Literal['none', 'required', 'auto'] | None = None
    elif not model_request_parameters.allow_text_result:
      tool_choice = 'required'
    else:
      tool_choice = 'auto'
    openai_messages: list[chat.ChatCompletionMessageParam] = []
    for m in messages:
      async for msg in self._map_message(m):
        openai_messages.append(msg)
    try:
      return await self.client.chat.completions.create(
        model=self._model_name,
        messages=openai_messages,
        n=1,
        parallel_tool_calls=model_settings.get('parallel_tool_calls', NOT_GIVEN),
        tools=tools or NOT_GIVEN,
        tool_choice=tool_choice or NOT_GIVEN,
        stream=stream,
        stream_options={'include_usage': True} if stream else NOT_GIVEN,
        max_completion_tokens=model_settings.get('max_tokens', NOT_GIVEN),
        temperature=model_settings.get('temperature', NOT_GIVEN),
        top_p=model_settings.get('top_p', NOT_GIVEN),
        timeout=model_settings.get('timeout', NOT_GIVEN),
        seed=model_settings.get('seed', NOT_GIVEN),
        presence_penalty=model_settings.get('presence_penalty', NOT_GIVEN),
        frequency_penalty=model_settings.get('frequency_penalty', NOT_GIVEN),
        logit_bias=model_settings.get('logit_bias', NOT_GIVEN),
        reasoning_effort=model_settings.get('openai_reasoning_effort', NOT_GIVEN),
        user=model_settings.get('user', NOT_GIVEN),
      )
    except APIStatusError as e:
      if (status_code := e.status_code) >= 400:
        raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) frome
      raise
  def_process_response(self, response: chat.ChatCompletion) -> ModelResponse:
"""Process a non-streamed response, and prepare a message to return."""
    timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
    choice = response.choices[0]
    items: list[ModelResponsePart] = []
    if choice.message.content is not None:
      items.append(TextPart(choice.message.content))
    if choice.message.tool_calls is not None:
      for c in choice.message.tool_calls:
        items.append(ToolCallPart(c.function.name, c.function.arguments, c.id))
    return ModelResponse(items, model_name=response.model, timestamp=timestamp)
  async def_process_streamed_response(self, response: AsyncStream[ChatCompletionChunk]) -> OpenAIStreamedResponse:
"""Process a streamed response, and prepare a streaming response to return."""
    peekable_response = _utils.PeekableAsyncStream(response)
    first_chunk = await peekable_response.peek()
    if isinstance(first_chunk, _utils.Unset):
      raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')
    return OpenAIStreamedResponse(
      _model_name=self._model_name,
      _response=peekable_response,
      _timestamp=datetime.fromtimestamp(first_chunk.created, tz=timezone.utc),
    )
  def_get_tools(self, model_request_parameters: ModelRequestParameters) -> list[chat.ChatCompletionToolParam]:
    tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
    if model_request_parameters.result_tools:
      tools += [self._map_tool_definition(r) for r in model_request_parameters.result_tools]
    return tools
  async def_map_message(self, message: ModelMessage) -> AsyncIterable[chat.ChatCompletionMessageParam]:
"""Just maps a `pydantic_ai.Message` to a `openai.types.ChatCompletionMessageParam`."""
    if isinstance(message, ModelRequest):
      async for item in self._map_user_message(message):
        yield item
    elif isinstance(message, ModelResponse):
      texts: list[str] = []
      tool_calls: list[chat.ChatCompletionMessageToolCallParam] = []
      for item in message.parts:
        if isinstance(item, TextPart):
          texts.append(item.content)
        elif isinstance(item, ToolCallPart):
          tool_calls.append(self._map_tool_call(item))
        else:
          assert_never(item)
      message_param = chat.ChatCompletionAssistantMessageParam(role='assistant')
      if texts:
        # Note: model responses from this model should only have one text item, so the following
        # shouldn't merge multiple texts into one unless you switch models between runs:
        message_param['content'] = '\n\n'.join(texts)
      if tool_calls:
        message_param['tool_calls'] = tool_calls
      yield message_param
    else:
      assert_never(message)
  @staticmethod
  def_map_tool_call(t: ToolCallPart) -> chat.ChatCompletionMessageToolCallParam:
    return chat.ChatCompletionMessageToolCallParam(
      id=_guard_tool_call_id(t=t),
      type='function',
      function={'name': t.tool_name, 'arguments': t.args_as_json_str()},
    )
  @staticmethod
  def_map_tool_definition(f: ToolDefinition) -> chat.ChatCompletionToolParam:
    return {
      'type': 'function',
      'function': {
        'name': f.name,
        'description': f.description,
        'parameters': f.parameters_json_schema,
      },
    }
  async def_map_user_message(self, message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
    for part in message.parts:
      if isinstance(part, SystemPromptPart):
        if self.system_prompt_role == 'developer':
          yield chat.ChatCompletionDeveloperMessageParam(role='developer', content=part.content)
        elif self.system_prompt_role == 'user':
          yield chat.ChatCompletionUserMessageParam(role='user', content=part.content)
        else:
          yield chat.ChatCompletionSystemMessageParam(role='system', content=part.content)
      elif isinstance(part, UserPromptPart):
        yield await self._map_user_prompt(part)
      elif isinstance(part, ToolReturnPart):
        yield chat.ChatCompletionToolMessageParam(
          role='tool',
          tool_call_id=_guard_tool_call_id(t=part),
          content=part.model_response_str(),
        )
      elif isinstance(part, RetryPromptPart):
        if part.tool_name is None:
          yield chat.ChatCompletionUserMessageParam(role='user', content=part.model_response())
        else:
          yield chat.ChatCompletionToolMessageParam(
            role='tool',
            tool_call_id=_guard_tool_call_id(t=part),
            content=part.model_response(),
          )
      else:
        assert_never(part)
  @staticmethod
  async def_map_user_prompt(part: UserPromptPart) -> chat.ChatCompletionUserMessageParam:
    content: str | list[ChatCompletionContentPartParam]
    if isinstance(part.content, str):
      content = part.content
    else:
      content = []
      for item in part.content:
        if isinstance(item, str):
          content.append(ChatCompletionContentPartTextParam(text=item, type='text'))
        elif isinstance(item, ImageUrl):
          image_url = ImageURL(url=item.url)
          content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
        elif isinstance(item, BinaryContent):
          base64_encoded = base64.b64encode(item.data).decode('utf-8')
          if item.is_image:
            image_url = ImageURL(url=f'data:{item.media_type};base64,{base64_encoded}')
            content.append(ChatCompletionContentPartImageParam(image_url=image_url, type='image_url'))
          elif item.is_audio:
            assert item.format in ('wav', 'mp3')
            audio = InputAudio(data=base64_encoded, format=item.format)
            content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
          else: # pragma: no cover
            raise RuntimeError(f'Unsupported binary content type: {item.media_type}')
        elif isinstance(item, AudioUrl): # pragma: no cover
          client = cached_async_http_client()
          response = await client.get(item.url)
          response.raise_for_status()
          base64_encoded = base64.b64encode(response.content).decode('utf-8')
          audio = InputAudio(data=base64_encoded, format=response.headers.get('content-type'))
          content.append(ChatCompletionContentPartInputAudioParam(input_audio=audio, type='input_audio'))
        elif isinstance(item, DocumentUrl): # pragma: no cover
          raise NotImplementedError('DocumentUrl is not supported for OpenAI')
          # The following implementation should have worked, but it seems we have the following error:
          # pydantic_ai.exceptions.ModelHTTPError: status_code: 400, model_name: gpt-4o, body:
          # {
          #  'message': "Unknown parameter: 'messages[1].content[1].file.data'.",
          #  'type': 'invalid_request_error',
          #  'param': 'messages[1].content[1].file.data',
          #  'code': 'unknown_parameter'
          # }
          #
          # client = cached_async_http_client()
          # response = await client.get(item.url)
          # response.raise_for_status()
          # base64_encoded = base64.b64encode(response.content).decode('utf-8')
          # media_type = response.headers.get('content-type').split(';')[0]
          # file_data = f'data:{media_type};base64,{base64_encoded}'
          # file = File(file={'file_data': file_data, 'file_name': item.url, 'file_id': item.url}, type='file')
          # content.append(file)
        else:
          assert_never(item)
    return chat.ChatCompletionUserMessageParam(role='user', content=content)

```
  
---|---  
####  __init__
```
__init__(
  model_name: ,
  *,
  provider: (
    ["openai", "deepseek", "azure"]
    | [AsyncOpenAI]
  ) = "openai",
  system_prompt_role: OpenAISystemPromptRole | None = None
)

```

Initialize an OpenAI model.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`model_name` |  |  The name of the OpenAI model to use. List of model names available [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7) (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API). |  _required_  
`provider` |  `Literal[](https://docs.python.org/3/library/typing.html#typing.Literal "typing.Literal")['openai', 'deepseek', 'azure'] | Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncOpenAI]` |  The provider to use. Defaults to `'openai'`. |  `'openai'`  
`system_prompt_role` |  `OpenAISystemPromptRole | None` |  The role to use for the system prompt message. If not provided, defaults to `'system'`. In the future, this may be inferred from the model name. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`
```
```
| ```
def__init__(
  self,
  model_name: OpenAIModelName,
  *,
  provider: Literal['openai', 'deepseek', 'azure'] | Provider[AsyncOpenAI] = 'openai',
  system_prompt_role: OpenAISystemPromptRole | None = None,
):
"""Initialize an OpenAI model.
  Args:
    model_name: The name of the OpenAI model to use. List of model names available
      [here](https://github.com/openai/openai-python/blob/v1.54.3/src/openai/types/chat_model.py#L7)
      (Unfortunately, despite being ask to do so, OpenAI do not provide `.inv` files for their API).
    provider: The provider to use. Defaults to `'openai'`.
    system_prompt_role: The role to use for the system prompt message. If not provided, defaults to `'system'`.
      In the future, this may be inferred from the model name.
  """
  self._model_name = model_name
  if isinstance(provider, str):
    provider = infer_provider(provider)
  self.client = provider.client
  self.system_prompt_role = system_prompt_role

```
  
---|---  
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider.
###  OpenAIStreamedResponse `dataclass`
Bases: 
Implementation of `StreamedResponse` for OpenAI models.
Source code in `pydantic_ai_slim/pydantic_ai/models/openai.py`
```
```
| ```
@dataclass
classOpenAIStreamedResponse(StreamedResponse):
"""Implementation of `StreamedResponse` for OpenAI models."""
  _model_name: OpenAIModelName
  _response: AsyncIterable[ChatCompletionChunk]
  _timestamp: datetime
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    async for chunk in self._response:
      self._usage += _map_usage(chunk)
      try:
        choice = chunk.choices[0]
      except IndexError:
        continue
      # Handle the text part of the response
      content = choice.delta.content
      if content is not None:
        yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=content)
      for dtc in choice.delta.tool_calls or []:
        maybe_event = self._parts_manager.handle_tool_call_delta(
          vendor_part_id=dtc.index,
          tool_name=dtc.function and dtc.function.name,
          args=dtc.function and dtc.function.arguments,
          tool_call_id=dtc.id,
        )
        if maybe_event is not None:
          yield maybe_event
  @property
  defmodel_name(self) -> OpenAIModelName:
"""Get the model name of the response."""
    return self._model_name
  @property
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    return self._timestamp

```
  
---|---  
####  model_name `property`
```
model_name: 

```

Get the model name of the response.
####  timestamp `property`
```
timestamp: 

```

Get the timestamp of the response.

# `pydantic_ai.models.test`
Utility model for quickly testing apps built with PydanticAI.
Here's a minimal example:
test_model_usage.py```
frompydantic_aiimport Agent
frompydantic_ai.models.testimport TestModel
my_agent = Agent('openai:gpt-4o', system_prompt='...')

async deftest_my_agent():
"""Unit test for my_agent, to be run by pytest."""
  m = TestModel()
  with my_agent.override(model=m):
    result = await my_agent.run('Testing my agent...')
    assert result.data == 'success (no tool calls)'
  assert m.last_model_request_parameters.function_tools == []

```

See [Unit testing with `TestModel`](https://ai.pydantic.dev/testing-evals/#unit-testing-with-testmodel) for detailed documentation.
###  TestModel `dataclass`
Bases: 
A model specifically for testing purposes.
This will (by default) call all tools in the agent, then return a tool response if possible, otherwise a plain response.
How useful this model is will vary significantly.
Apart from `__init__` derived by the `dataclass` decorator, all methods are private or match those of the base class.
Source code in `pydantic_ai_slim/pydantic_ai/models/test.py`
```
```
| ```
@dataclass
classTestModel(Model):
"""A model specifically for testing purposes.
  This will (by default) call all tools in the agent, then return a tool response if possible,
  otherwise a plain response.
  How useful this model is will vary significantly.
  Apart from `__init__` derived by the `dataclass` decorator, all methods are private or match those
  of the base class.
  """
  # NOTE: Avoid test discovery by pytest.
  __test__ = False
  call_tools: list[str] | Literal['all'] = 'all'
"""List of tools to call. If `'all'`, all tools will be called."""
  custom_result_text: str | None = None
"""If set, this text is returned as the final result."""
  custom_result_args: Any | None = None
"""If set, these args will be passed to the result tool."""
  seed: int = 0
"""Seed for generating random data."""
  last_model_request_parameters: ModelRequestParameters | None = field(default=None, init=False)
"""The last ModelRequestParameters passed to the model in a request.
  The ModelRequestParameters contains information about the function and result tools available during request handling.
  This is set when a request is made, so will reflect the function tools from the last step of the last run.
  """
  _model_name: str = field(default='test', repr=False)
  _system: str = field(default='test', repr=False)
  async defrequest(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> tuple[ModelResponse, Usage]:
    self.last_model_request_parameters = model_request_parameters
    model_response = self._request(messages, model_settings, model_request_parameters)
    usage = _estimate_usage([*messages, model_response])
    return model_response, usage
  @asynccontextmanager
  async defrequest_stream(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> AsyncIterator[StreamedResponse]:
    self.last_model_request_parameters = model_request_parameters
    model_response = self._request(messages, model_settings, model_request_parameters)
    yield TestStreamedResponse(
      _model_name=self._model_name, _structured_response=model_response, _messages=messages
    )
  @property
  defmodel_name(self) -> str:
"""The model name."""
    return self._model_name
  @property
  defsystem(self) -> str:
"""The system / model provider."""
    return self._system
  defgen_tool_args(self, tool_def: ToolDefinition) -> Any:
    return _JsonSchemaTestData(tool_def.parameters_json_schema, self.seed).generate()
  def_get_tool_calls(self, model_request_parameters: ModelRequestParameters) -> list[tuple[str, ToolDefinition]]:
    if self.call_tools == 'all':
      return [(r.name, r) for r in model_request_parameters.function_tools]
    else:
      function_tools_lookup = {t.name: t for t in model_request_parameters.function_tools}
      tools_to_call = (function_tools_lookup[name] for name in self.call_tools)
      return [(r.name, r) for r in tools_to_call]
  def_get_result(self, model_request_parameters: ModelRequestParameters) -> _TextResult | _FunctionToolResult:
    if self.custom_result_text is not None:
      assert model_request_parameters.allow_text_result, (
        'Plain response not allowed, but `custom_result_text` is set.'
      )
      assert self.custom_result_args is None, 'Cannot set both `custom_result_text` and `custom_result_args`.'
      return _TextResult(self.custom_result_text)
    elif self.custom_result_args is not None:
      assert model_request_parameters.result_tools is not None, (
        'No result tools provided, but `custom_result_args` is set.'
      )
      result_tool = model_request_parameters.result_tools[0]
      if k := result_tool.outer_typed_dict_key:
        return _FunctionToolResult({k: self.custom_result_args})
      else:
        return _FunctionToolResult(self.custom_result_args)
    elif model_request_parameters.allow_text_result:
      return _TextResult(None)
    elif model_request_parameters.result_tools:
      return _FunctionToolResult(None)
    else:
      return _TextResult(None)
  def_request(
    self,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
  ) -> ModelResponse:
    tool_calls = self._get_tool_calls(model_request_parameters)
    result = self._get_result(model_request_parameters)
    result_tools = model_request_parameters.result_tools
    # if there are tools, the first thing we want to do is call all of them
    if tool_calls and not any(isinstance(m, ModelResponse) for m in messages):
      return ModelResponse(
        parts=[ToolCallPart(name, self.gen_tool_args(args)) for name, args in tool_calls],
        model_name=self._model_name,
      )
    if messages:
      last_message = messages[-1]
      assert isinstance(last_message, ModelRequest), 'Expected last message to be a `ModelRequest`.'
      # check if there are any retry prompts, if so retry them
      new_retry_names = {p.tool_name for p in last_message.parts if isinstance(p, RetryPromptPart)}
      if new_retry_names:
        # Handle retries for both function tools and result tools
        # Check function tools first
        retry_parts: list[ModelResponsePart] = [
          ToolCallPart(name, self.gen_tool_args(args)) for name, args in tool_calls if name in new_retry_names
        ]
        # Check result tools
        if result_tools:
          retry_parts.extend(
            [
              ToolCallPart(
                tool.name,
                result.value
                if isinstance(result, _FunctionToolResult) and result.value is not None
                else self.gen_tool_args(tool),
              )
              for tool in result_tools
              if tool.name in new_retry_names
            ]
          )
        return ModelResponse(parts=retry_parts, model_name=self._model_name)
    if isinstance(result, _TextResult):
      if (response_text := result.value) is None:
        # build up details of tool responses
        output: dict[str, Any] = {}
        for message in messages:
          if isinstance(message, ModelRequest):
            for part in message.parts:
              if isinstance(part, ToolReturnPart):
                output[part.tool_name] = part.content
        if output:
          return ModelResponse(
            parts=[TextPart(pydantic_core.to_json(output).decode())], model_name=self._model_name
          )
        else:
          return ModelResponse(parts=[TextPart('success (no tool calls)')], model_name=self._model_name)
      else:
        return ModelResponse(parts=[TextPart(response_text)], model_name=self._model_name)
    else:
      assert result_tools, 'No result tools provided'
      custom_result_args = result.value
      result_tool = result_tools[self.seed % len(result_tools)]
      if custom_result_args is not None:
        return ModelResponse(
          parts=[ToolCallPart(result_tool.name, custom_result_args)], model_name=self._model_name
        )
      else:
        response_args = self.gen_tool_args(result_tool)
        return ModelResponse(parts=[ToolCallPart(result_tool.name, response_args)], model_name=self._model_name)

```
  
---|---  
####  call_tools `class-attribute` `instance-attribute`
```
call_tools: [] | ['all'] = 'all'

```

List of tools to call. If `'all'`, all tools will be called.
####  custom_result_text `class-attribute` `instance-attribute`
```
custom_result_text:  | None = None

```

If set, this text is returned as the final result.
####  custom_result_args `class-attribute` `instance-attribute`
```
custom_result_args:  | None = None

```

If set, these args will be passed to the result tool.
####  seed `class-attribute` `instance-attribute`
```
seed:  = 0

```

Seed for generating random data.
####  last_model_request_parameters `class-attribute` `instance-attribute`
```
last_model_request_parameters: (
   | None
) = (default=None, =False)

```

The last ModelRequestParameters passed to the model in a request.
The ModelRequestParameters contains information about the function and result tools available during request handling.
This is set when a request is made, so will reflect the function tools from the last step of the last run.
####  model_name `property`
```
model_name: 

```

The model name.
####  system `property`
```
system: 

```

The system / model provider.
###  TestStreamedResponse `dataclass`
Bases: 
A structured response that streams test data.
Source code in `pydantic_ai_slim/pydantic_ai/models/test.py`
```
```
| ```
@dataclass
classTestStreamedResponse(StreamedResponse):
"""A structured response that streams test data."""
  _model_name: str
  _structured_response: ModelResponse
  _messages: InitVar[Iterable[ModelMessage]]
  _timestamp: datetime = field(default_factory=_utils.now_utc, init=False)
  def__post_init__(self, _messages: Iterable[ModelMessage]):
    self._usage = _estimate_usage(_messages)
  async def_get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    for i, part in enumerate(self._structured_response.parts):
      if isinstance(part, TextPart):
        text = part.content
        *words, last_word = text.split(' ')
        words = [f'{word} ' for word in words]
        words.append(last_word)
        if len(words) == 1 and len(text)  2:
          mid = len(text) // 2
          words = [text[:mid], text[mid:]]
        self._usage += _get_string_usage('')
        yield self._parts_manager.handle_text_delta(vendor_part_id=i, content='')
        for word in words:
          self._usage += _get_string_usage(word)
          yield self._parts_manager.handle_text_delta(vendor_part_id=i, content=word)
      else:
        yield self._parts_manager.handle_tool_call_part(
          vendor_part_id=i, tool_name=part.tool_name, args=part.args, tool_call_id=part.tool_call_id
        )
  @property
  defmodel_name(self) -> str:
"""Get the model name of the response."""
    return self._model_name
  @property
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    return self._timestamp

```
  
---|---  
####  model_name `property`
```
model_name: 

```

Get the model name of the response.
####  timestamp `property`
```
timestamp: 

```

Get the timestamp of the response.

# 404 - Not found

# `pydantic_ai.providers`
Bases: , `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[InterfaceClient]`
Abstract class for a provider.
The provider is in charge of providing an authenticated client to the API.
Each provider only supports a specific interface. A interface can be supported by multiple providers.
For example, the OpenAIModel interface can be supported by the OpenAIProvider and the DeepSeekProvider.
Source code in `pydantic_ai_slim/pydantic_ai/providers/__init__.py`
```
```
| ```
classProvider(ABC, Generic[InterfaceClient]):
"""Abstract class for a provider.
  The provider is in charge of providing an authenticated client to the API.
  Each provider only supports a specific interface. A interface can be supported by multiple providers.
  For example, the OpenAIModel interface can be supported by the OpenAIProvider and the DeepSeekProvider.
  """
  _client: InterfaceClient
  @property
  @abstractmethod
  defname(self) -> str:
"""The provider name."""
    raise NotImplementedError()
  @property
  @abstractmethod
  defbase_url(self) -> str:
"""The base URL for the provider API."""
    raise NotImplementedError()
  @property
  @abstractmethod
  defclient(self) -> InterfaceClient:
"""The client for the provider."""
    raise NotImplementedError()

```
  
---|---  
###  name `abstractmethod` `property`
```
name: 

```

The provider name.
###  base_url `abstractmethod` `property`
```
base_url: 

```

The base URL for the provider API.
###  client `abstractmethod` `property`
```
client: InterfaceClient

```

The client for the provider.
###  GoogleVertexProvider
Bases: `Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncClient]`
Provider for Vertex AI API.
Source code in `pydantic_ai_slim/pydantic_ai/providers/google_vertex.py`
```
```
| ```
classGoogleVertexProvider(Provider[httpx.AsyncClient]):
"""Provider for Vertex AI API."""
  @property
  defname(self) -> str:
    return 'google-vertex'
  @property
  defbase_url(self) -> str:
    return (
      f'https://{self.region}-aiplatform.googleapis.com/v1'
      f'/projects/{self.project_id}'
      f'/locations/{self.region}'
      f'/publishers/{self.model_publisher}/models/'
    )
  @property
  defclient(self) -> httpx.AsyncClient:
    return self._client
  @overload
  def__init__(
    self,
    *,
    service_account_file: Path | str | None = None,
    project_id: str | None = None,
    region: VertexAiRegion = 'us-central1',
    model_publisher: str = 'google',
    http_client: httpx.AsyncClient | None = None,
  ) -> None: ...
  @overload
  def__init__(
    self,
    *,
    service_account_info: Mapping[str, str] | None = None,
    project_id: str | None = None,
    region: VertexAiRegion = 'us-central1',
    model_publisher: str = 'google',
    http_client: httpx.AsyncClient | None = None,
  ) -> None: ...
  def__init__(
    self,
    *,
    service_account_file: Path | str | None = None,
    service_account_info: Mapping[str, str] | None = None,
    project_id: str | None = None,
    region: VertexAiRegion = 'us-central1',
    model_publisher: str = 'google',
    http_client: httpx.AsyncClient | None = None,
  ) -> None:
"""Create a new Vertex AI provider.
    Args:
      service_account_file: Path to a service account file.
        If not provided, the service_account_info or default environment credentials will be used.
      service_account_info: The loaded service_account_file contents.
        If not provided, the service_account_file or default environment credentials will be used.
      project_id: The project ID to use, if not provided it will be taken from the credentials.
      region: The region to make requests to.
      model_publisher: The model publisher to use, I couldn't find a good list of available publishers,
        and from trial and error it seems non-google models don't work with the `generateContent` and
        `streamGenerateContent` functions, hence only `google` is currently supported.
        Please create an issue or PR if you know how to use other publishers.
      http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    if service_account_file and service_account_info:
      raise ValueError('Only one of `service_account_file` or `service_account_info` can be provided.')
    self._client = http_client or cached_async_http_client(provider='google-vertex')
    self.service_account_file = service_account_file
    self.service_account_info = service_account_info
    self.project_id = project_id
    self.region = region
    self.model_publisher = model_publisher
    self._client.auth = _VertexAIAuth(service_account_file, service_account_info, project_id, region)
    self._client.base_url = self.base_url

```
  
---|---  
####  __init__
```
__init__(
  *,
  service_account_file:  |  | None = None,
  project_id:  | None = None,
  region: VertexAiRegion = "us-central1",
  model_publisher:  = "google",
  http_client: AsyncClient | None = None
) -> None

```

```
__init__(
  *,
  service_account_info: [, ] | None = None,
  project_id:  | None = None,
  region: VertexAiRegion = "us-central1",
  model_publisher:  = "google",
  http_client: AsyncClient | None = None
) -> None

```

```
__init__(
  *,
  service_account_file:  |  | None = None,
  service_account_info: [, ] | None = None,
  project_id:  | None = None,
  region: VertexAiRegion = "us-central1",
  model_publisher:  = "google",
  http_client: AsyncClient | None = None
) -> None

```

Create a new Vertex AI provider.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`service_account_file` |  `Path[](https://docs.python.org/3/library/pathlib.html#pathlib.Path "pathlib.Path") | str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  Path to a service account file. If not provided, the service_account_info or default environment credentials will be used. |  `None`  
`service_account_info` |  `Mapping[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping "collections.abc.Mapping")[str[](https://docs.python.org/3/library/stdtypes.html#str), str[](https://docs.python.org/3/library/stdtypes.html#str)] | None` |  The loaded service_account_file contents. If not provided, the service_account_file or default environment credentials will be used. |  `None`  
`project_id` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The project ID to use, if not provided it will be taken from the credentials. |  `None`  
`region` |  `VertexAiRegion` |  The region to make requests to. |  `'us-central1'`  
`model_publisher` |  The model publisher to use, I couldn't find a good list of available publishers, and from trial and error it seems non-google models don't work with the `generateContent` and `streamGenerateContent` functions, hence only `google` is currently supported. Please create an issue or PR if you know how to use other publishers. |  `'google'`  
`http_client` |  `AsyncClient | None` |  An existing `httpx.AsyncClient` to use for making HTTP requests. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/providers/google_vertex.py`
```
```
| ```
def__init__(
  self,
  *,
  service_account_file: Path | str | None = None,
  service_account_info: Mapping[str, str] | None = None,
  project_id: str | None = None,
  region: VertexAiRegion = 'us-central1',
  model_publisher: str = 'google',
  http_client: httpx.AsyncClient | None = None,
) -> None:
"""Create a new Vertex AI provider.
  Args:
    service_account_file: Path to a service account file.
      If not provided, the service_account_info or default environment credentials will be used.
    service_account_info: The loaded service_account_file contents.
      If not provided, the service_account_file or default environment credentials will be used.
    project_id: The project ID to use, if not provided it will be taken from the credentials.
    region: The region to make requests to.
    model_publisher: The model publisher to use, I couldn't find a good list of available publishers,
      and from trial and error it seems non-google models don't work with the `generateContent` and
      `streamGenerateContent` functions, hence only `google` is currently supported.
      Please create an issue or PR if you know how to use other publishers.
    http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
  """
  if service_account_file and service_account_info:
    raise ValueError('Only one of `service_account_file` or `service_account_info` can be provided.')
  self._client = http_client or cached_async_http_client(provider='google-vertex')
  self.service_account_file = service_account_file
  self.service_account_info = service_account_info
  self.project_id = project_id
  self.region = region
  self.model_publisher = model_publisher
  self._client.auth = _VertexAIAuth(service_account_file, service_account_info, project_id, region)
  self._client.base_url = self.base_url

```
  
---|---  
###  OpenAIProvider
Bases: `Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncOpenAI]`
Provider for OpenAI API.
Source code in `pydantic_ai_slim/pydantic_ai/providers/openai.py`
```
```
| ```
classOpenAIProvider(Provider[AsyncOpenAI]):
"""Provider for OpenAI API."""
  @property
  defname(self) -> str:
    return 'openai' # pragma: no cover
  @property
  defbase_url(self) -> str:
    return str(self.client.base_url)
  @property
  defclient(self) -> AsyncOpenAI:
    return self._client
  def__init__(
    self,
    base_url: str | None = None,
    api_key: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
  ) -> None:
"""Create a new OpenAI provider.
    Args:
      base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
        will be used if available. Otherwise, defaults to OpenAI's base url.
      api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
        will be used if available.
      openai_client: An existing
        [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
        client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
      http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
    # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
    if api_key is None and 'OPENAI_API_KEY' not in os.environ and base_url is not None and openai_client is None:
      api_key = 'api-key-not-set'
    if openai_client is not None:
      assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
      assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
      assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
      self._client = openai_client
    elif http_client is not None:
      self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
    else:
      http_client = cached_async_http_client(provider='openai')
      self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

```
  
---|---  
####  __init__
```
__init__(
  base_url:  | None = None,
  api_key:  | None = None,
  openai_client: AsyncOpenAI | None = None,
  http_client: AsyncClient | None = None,
) -> None

```

Create a new OpenAI provider.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`base_url` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable will be used if available. Otherwise, defaults to OpenAI's base url. |  `None`  
`api_key` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable will be used if available. |  `None`  
`openai_client` |  `AsyncOpenAI | None` |  An existing [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage) client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`. |  `None`  
`http_client` |  `AsyncClient | None` |  An existing `httpx.AsyncClient` to use for making HTTP requests. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/providers/openai.py`
```
```
| ```
def__init__(
  self,
  base_url: str | None = None,
  api_key: str | None = None,
  openai_client: AsyncOpenAI | None = None,
  http_client: httpx.AsyncClient | None = None,
) -> None:
"""Create a new OpenAI provider.
  Args:
    base_url: The base url for the OpenAI requests. If not provided, the `OPENAI_BASE_URL` environment variable
      will be used if available. Otherwise, defaults to OpenAI's base url.
    api_key: The API key to use for authentication, if not provided, the `OPENAI_API_KEY` environment variable
      will be used if available.
    openai_client: An existing
      [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
      client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
    http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
  """
  # This is a workaround for the OpenAI client requiring an API key, whilst locally served,
  # openai compatible models do not always need an API key, but a placeholder (non-empty) key is required.
  if api_key is None and 'OPENAI_API_KEY' not in os.environ and base_url is not None and openai_client is None:
    api_key = 'api-key-not-set'
  if openai_client is not None:
    assert base_url is None, 'Cannot provide both `openai_client` and `base_url`'
    assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
    assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
    self._client = openai_client
  elif http_client is not None:
    self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
  else:
    http_client = cached_async_http_client(provider='openai')
    self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_client)

```
  
---|---  
###  DeepSeekProvider
Bases: `Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncOpenAI]`
Provider for DeepSeek API.
Source code in `pydantic_ai_slim/pydantic_ai/providers/deepseek.py`
```
```
| ```
classDeepSeekProvider(Provider[AsyncOpenAI]):
"""Provider for DeepSeek API."""
  @property
  defname(self) -> str:
    return 'deepseek'
  @property
  defbase_url(self) -> str:
    return 'https://api.deepseek.com'
  @property
  defclient(self) -> AsyncOpenAI:
    return self._client
  @overload
  def__init__(self) -> None: ...
  @overload
  def__init__(self, *, api_key: str) -> None: ...
  @overload
  def__init__(self, *, api_key: str, http_client: AsyncHTTPClient) -> None: ...
  @overload
  def__init__(self, *, openai_client: AsyncOpenAI | None = None) -> None: ...
  def__init__(
    self,
    *,
    api_key: str | None = None,
    openai_client: AsyncOpenAI | None = None,
    http_client: AsyncHTTPClient | None = None,
  ) -> None:
    api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
    if not api_key and openai_client is None:
      raise UserError(
        'Set the `DEEPSEEK_API_KEY` environment variable or pass it via `DeepSeekProvider(api_key=...)`'
        'to use the DeepSeek provider.'
      )
    if openai_client is not None:
      self._client = openai_client
    elif http_client is not None:
      self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)
    else:
      http_client = cached_async_http_client(provider='deepseek')
      self._client = AsyncOpenAI(base_url=self.base_url, api_key=api_key, http_client=http_client)

```
  
---|---  
###  BedrockProvider
Bases: `Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[BaseClient]`
Provider for AWS Bedrock.
Source code in `pydantic_ai_slim/pydantic_ai/providers/bedrock.py`
```
```
| ```
classBedrockProvider(Provider[BaseClient]):
"""Provider for AWS Bedrock."""
  @property
  defname(self) -> str:
    return 'bedrock'
  @property
  defbase_url(self) -> str:
    return self._client.meta.endpoint_url
  @property
  defclient(self) -> BaseClient:
    return self._client
  @overload
  def__init__(self, *, bedrock_client: BaseClient) -> None: ...
  @overload
  def__init__(
    self,
    *,
    region_name: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
  ) -> None: ...
  def__init__(
    self,
    *,
    bedrock_client: BaseClient | None = None,
    region_name: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
  ) -> None:
"""Initialize the Bedrock provider.
    Args:
      bedrock_client: A boto3 client for Bedrock Runtime. If provided, other arguments are ignored.
      region_name: The AWS region name.
      aws_access_key_id: The AWS access key ID.
      aws_secret_access_key: The AWS secret access key.
      aws_session_token: The AWS session token.
    """
    if bedrock_client is not None:
      self._client = bedrock_client
    else:
      try:
        self._client = boto3.client( # type: ignore[reportUnknownMemberType]
          'bedrock-runtime',
          aws_access_key_id=aws_access_key_id,
          aws_secret_access_key=aws_secret_access_key,
          aws_session_token=aws_session_token,
          region_name=region_name,
        )
      except NoRegionError as exc: # pragma: no cover
        raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') fromexc

```
  
---|---  
####  __init__
```
__init__(*, bedrock_client: BaseClient) -> None

```

```
__init__(
  *,
  region_name:  | None = None,
  aws_access_key_id:  | None = None,
  aws_secret_access_key:  | None = None,
  aws_session_token:  | None = None
) -> None

```

```
__init__(
  *,
  bedrock_client: BaseClient | None = None,
  region_name:  | None = None,
  aws_access_key_id:  | None = None,
  aws_secret_access_key:  | None = None,
  aws_session_token:  | None = None
) -> None

```

Initialize the Bedrock provider.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`bedrock_client` |  `BaseClient | None` |  A boto3 client for Bedrock Runtime. If provided, other arguments are ignored. |  `None`  
`region_name` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The AWS region name. |  `None`  
`aws_access_key_id` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The AWS access key ID. |  `None`  
`aws_secret_access_key` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The AWS secret access key. |  `None`  
`aws_session_token` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The AWS session token. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/providers/bedrock.py`
```
```
| ```
def__init__(
  self,
  *,
  bedrock_client: BaseClient | None = None,
  region_name: str | None = None,
  aws_access_key_id: str | None = None,
  aws_secret_access_key: str | None = None,
  aws_session_token: str | None = None,
) -> None:
"""Initialize the Bedrock provider.
  Args:
    bedrock_client: A boto3 client for Bedrock Runtime. If provided, other arguments are ignored.
    region_name: The AWS region name.
    aws_access_key_id: The AWS access key ID.
    aws_secret_access_key: The AWS secret access key.
    aws_session_token: The AWS session token.
  """
  if bedrock_client is not None:
    self._client = bedrock_client
  else:
    try:
      self._client = boto3.client( # type: ignore[reportUnknownMemberType]
        'bedrock-runtime',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
        region_name=region_name,
      )
    except NoRegionError as exc: # pragma: no cover
      raise UserError('You must provide a `region_name` or a boto3 client for Bedrock Runtime.') fromexc

```
  
---|---  
###  GroqProvider
Bases: `Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncGroq]`
Provider for Groq API.
Source code in `pydantic_ai_slim/pydantic_ai/providers/groq.py`
```
```
| ```
classGroqProvider(Provider[AsyncGroq]):
"""Provider for Groq API."""
  @property
  defname(self) -> str:
    return 'groq'
  @property
  defbase_url(self) -> str:
    return os.environ.get('GROQ_BASE_URL', 'https://api.groq.com')
  @property
  defclient(self) -> AsyncGroq:
    return self._client
  @overload
  def__init__(self, *, groq_client: AsyncGroq | None = None) -> None: ...
  @overload
  def__init__(self, *, api_key: str | None = None, http_client: AsyncHTTPClient | None = None) -> None: ...
  def__init__(
    self,
    *,
    api_key: str | None = None,
    groq_client: AsyncGroq | None = None,
    http_client: AsyncHTTPClient | None = None,
  ) -> None:
"""Create a new Groq provider.
    Args:
      api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
        will be used if available.
      groq_client: An existing
        [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
        client to use. If provided, `api_key` and `http_client` must be `None`.
      http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
    """
    if groq_client is not None:
      assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
      assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
      self._client = groq_client
    else:
      api_key = api_key or os.environ.get('GROQ_API_KEY')
      if not api_key:
        raise UserError(
          'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
          'to use the Groq provider.'
        )
      elif http_client is not None:
        self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)
      else:
        http_client = cached_async_http_client(provider='groq')
        self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)

```
  
---|---  
####  __init__
```
__init__(*, groq_client: AsyncGroq | None = None) -> None

```

```
__init__(
  *,
  api_key:  | None = None,
  http_client: AsyncClient | None = None
) -> None

```

```
__init__(
  *,
  api_key:  | None = None,
  groq_client: AsyncGroq | None = None,
  http_client: AsyncClient | None = None
) -> None

```

Create a new Groq provider.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`api_key` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable will be used if available. |  `None`  
`groq_client` |  `AsyncGroq | None` |  An existing [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage) client to use. If provided, `api_key` and `http_client` must be `None`. |  `None`  
`http_client` |  `AsyncClient | None` |  An existing `AsyncHTTPClient` to use for making HTTP requests. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/providers/groq.py`
```
```
| ```
def__init__(
  self,
  *,
  api_key: str | None = None,
  groq_client: AsyncGroq | None = None,
  http_client: AsyncHTTPClient | None = None,
) -> None:
"""Create a new Groq provider.
  Args:
    api_key: The API key to use for authentication, if not provided, the `GROQ_API_KEY` environment variable
      will be used if available.
    groq_client: An existing
      [`AsyncGroq`](https://github.com/groq/groq-python?tab=readme-ov-file#async-usage)
      client to use. If provided, `api_key` and `http_client` must be `None`.
    http_client: An existing `AsyncHTTPClient` to use for making HTTP requests.
  """
  if groq_client is not None:
    assert http_client is None, 'Cannot provide both `groq_client` and `http_client`'
    assert api_key is None, 'Cannot provide both `groq_client` and `api_key`'
    self._client = groq_client
  else:
    api_key = api_key or os.environ.get('GROQ_API_KEY')
    if not api_key:
      raise UserError(
        'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
        'to use the Groq provider.'
      )
    elif http_client is not None:
      self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)
    else:
      http_client = cached_async_http_client(provider='groq')
      self._client = AsyncGroq(base_url=self.base_url, api_key=api_key, http_client=http_client)

```
  
---|---  
###  AzureProvider
Bases: `Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncOpenAI]`
Provider for Azure OpenAI API.
See <https://azure.microsoft.com/en-us/products/ai-foundry> for more information.
Source code in `pydantic_ai_slim/pydantic_ai/providers/azure.py`
```
```
| ```
classAzureProvider(Provider[AsyncOpenAI]):
"""Provider for Azure OpenAI API.
  See <https://azure.microsoft.com/en-us/products/ai-foundry> for more information.
  """
  @property
  defname(self) -> str:
    return 'azure'
  @property
  defbase_url(self) -> str:
    assert self._base_url is not None
    return self._base_url
  @property
  defclient(self) -> AsyncOpenAI:
    return self._client
  @overload
  def__init__(self, *, openai_client: AsyncAzureOpenAI) -> None: ...
  @overload
  def__init__(
    self,
    *,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    http_client: httpx.AsyncClient | None = None,
  ) -> None: ...
  def__init__(
    self,
    *,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
    api_key: str | None = None,
    openai_client: AsyncAzureOpenAI | None = None,
    http_client: httpx.AsyncClient | None = None,
  ) -> None:
"""Create a new Azure provider.
    Args:
      azure_endpoint: The Azure endpoint to use for authentication, if not provided, the `AZURE_OPENAI_ENDPOINT`
        environment variable will be used if available.
      api_version: The API version to use for authentication, if not provided, the `OPENAI_API_VERSION`
        environment variable will be used if available.
      api_key: The API key to use for authentication, if not provided, the `AZURE_OPENAI_API_KEY` environment variable
        will be used if available.
      openai_client: An existing
        [`AsyncAzureOpenAI`](https://github.com/openai/openai-python#microsoft-azure-openai)
        client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
      http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    if openai_client is not None:
      assert azure_endpoint is None, 'Cannot provide both `openai_client` and `azure_endpoint`'
      assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
      assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
      self._base_url = str(openai_client.base_url)
      self._client = openai_client
    else:
      azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
      if not azure_endpoint: # pragma: no cover
        raise UserError(
          'Must provide one of the `azure_endpoint` argument or the `AZURE_OPENAI_ENDPOINT` environment variable'
        )
      if not api_key and 'OPENAI_API_KEY' not in os.environ: # pragma: no cover
        raise UserError(
          'Must provide one of the `api_key` argument or the `OPENAI_API_KEY` environment variable'
        )
      if not api_version and 'OPENAI_API_VERSION' not in os.environ: # pragma: no cover
        raise UserError(
          'Must provide one of the `api_version` argument or the `OPENAI_API_VERSION` environment variable'
        )
      http_client = http_client or cached_async_http_client(provider='azure')
      self._client = AsyncAzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        http_client=http_client,
      )
      self._base_url = str(self._client.base_url)

```
  
---|---  
####  __init__
```
__init__(*, openai_client: AsyncAzureOpenAI) -> None

```

```
__init__(
  *,
  azure_endpoint:  | None = None,
  api_version:  | None = None,
  api_key:  | None = None,
  http_client: AsyncClient | None = None
) -> None

```

```
__init__(
  *,
  azure_endpoint:  | None = None,
  api_version:  | None = None,
  api_key:  | None = None,
  openai_client: AsyncAzureOpenAI | None = None,
  http_client: AsyncClient | None = None
) -> None

```

Create a new Azure provider.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`azure_endpoint` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The Azure endpoint to use for authentication, if not provided, the `AZURE_OPENAI_ENDPOINT` environment variable will be used if available. |  `None`  
`api_version` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The API version to use for authentication, if not provided, the `OPENAI_API_VERSION` environment variable will be used if available. |  `None`  
`api_key` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The API key to use for authentication, if not provided, the `AZURE_OPENAI_API_KEY` environment variable will be used if available. |  `None`  
`openai_client` |  `AsyncAzureOpenAI | None` |  An existing [`AsyncAzureOpenAI`](https://github.com/openai/openai-python#microsoft-azure-openai) client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`. |  `None`  
`http_client` |  `AsyncClient | None` |  An existing `httpx.AsyncClient` to use for making HTTP requests. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/providers/azure.py`
```
```
| ```
def__init__(
  self,
  *,
  azure_endpoint: str | None = None,
  api_version: str | None = None,
  api_key: str | None = None,
  openai_client: AsyncAzureOpenAI | None = None,
  http_client: httpx.AsyncClient | None = None,
) -> None:
"""Create a new Azure provider.
  Args:
    azure_endpoint: The Azure endpoint to use for authentication, if not provided, the `AZURE_OPENAI_ENDPOINT`
      environment variable will be used if available.
    api_version: The API version to use for authentication, if not provided, the `OPENAI_API_VERSION`
      environment variable will be used if available.
    api_key: The API key to use for authentication, if not provided, the `AZURE_OPENAI_API_KEY` environment variable
      will be used if available.
    openai_client: An existing
      [`AsyncAzureOpenAI`](https://github.com/openai/openai-python#microsoft-azure-openai)
      client to use. If provided, `base_url`, `api_key`, and `http_client` must be `None`.
    http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
  """
  if openai_client is not None:
    assert azure_endpoint is None, 'Cannot provide both `openai_client` and `azure_endpoint`'
    assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
    assert api_key is None, 'Cannot provide both `openai_client` and `api_key`'
    self._base_url = str(openai_client.base_url)
    self._client = openai_client
  else:
    azure_endpoint = azure_endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
    if not azure_endpoint: # pragma: no cover
      raise UserError(
        'Must provide one of the `azure_endpoint` argument or the `AZURE_OPENAI_ENDPOINT` environment variable'
      )
    if not api_key and 'OPENAI_API_KEY' not in os.environ: # pragma: no cover
      raise UserError(
        'Must provide one of the `api_key` argument or the `OPENAI_API_KEY` environment variable'
      )
    if not api_version and 'OPENAI_API_VERSION' not in os.environ: # pragma: no cover
      raise UserError(
        'Must provide one of the `api_version` argument or the `OPENAI_API_VERSION` environment variable'
      )
    http_client = http_client or cached_async_http_client(provider='azure')
    self._client = AsyncAzureOpenAI(
      azure_endpoint=azure_endpoint,
      api_key=api_key,
      api_version=api_version,
      http_client=http_client,
    )
    self._base_url = str(self._client.base_url)

```
  
---|---  
###  CohereProvider
Bases: `Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[AsyncClientV2]`
Provider for Cohere API.
Source code in `pydantic_ai_slim/pydantic_ai/providers/cohere.py`
```
```
| ```
classCohereProvider(Provider[AsyncClientV2]):
"""Provider for Cohere API."""
  @property
  defname(self) -> str:
    return 'cohere'
  @property
  defbase_url(self) -> str:
    client_wrapper = self.client._client_wrapper # type: ignore
    return str(client_wrapper.get_base_url())
  @property
  defclient(self) -> AsyncClientV2:
    return self._client
  def__init__(
    self,
    *,
    api_key: str | None = None,
    cohere_client: AsyncClientV2 | None = None,
    http_client: AsyncHTTPClient | None = None,
  ) -> None:
"""Create a new Cohere provider.
    Args:
      api_key: The API key to use for authentication, if not provided, the `CO_API_KEY` environment variable
        will be used if available.
      cohere_client: An existing
        [AsyncClientV2](https://github.com/cohere-ai/cohere-python)
        client to use. If provided, `api_key` and `http_client` must be `None`.
      http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
    """
    if cohere_client is not None:
      assert http_client is None, 'Cannot provide both `cohere_client` and `http_client`'
      assert api_key is None, 'Cannot provide both `cohere_client` and `api_key`'
      self._client = cohere_client
    else:
      api_key = api_key or os.environ.get('CO_API_KEY')
      if not api_key:
        raise UserError(
          'Set the `CO_API_KEY` environment variable or pass it via `CohereProvider(api_key=...)`'
          'to use the Cohere provider.'
        )
      base_url = os.environ.get('CO_BASE_URL')
      if http_client is not None:
        self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)
      else:
        http_client = cached_async_http_client(provider='cohere')
        self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)

```
  
---|---  
####  __init__
```
__init__(
  *,
  api_key:  | None = None,
  cohere_client: AsyncClientV2 | None = None,
  http_client: AsyncClient | None = None
) -> None

```

Create a new Cohere provider.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`api_key` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The API key to use for authentication, if not provided, the `CO_API_KEY` environment variable will be used if available. |  `None`  
`cohere_client` |  `AsyncClientV2 | None` |  An existing [AsyncClientV2](https://github.com/cohere-ai/cohere-python) client to use. If provided, `api_key` and `http_client` must be `None`. |  `None`  
`http_client` |  `AsyncClient | None` |  An existing `httpx.AsyncClient` to use for making HTTP requests. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/providers/cohere.py`
```
```
| ```
def__init__(
  self,
  *,
  api_key: str | None = None,
  cohere_client: AsyncClientV2 | None = None,
  http_client: AsyncHTTPClient | None = None,
) -> None:
"""Create a new Cohere provider.
  Args:
    api_key: The API key to use for authentication, if not provided, the `CO_API_KEY` environment variable
      will be used if available.
    cohere_client: An existing
      [AsyncClientV2](https://github.com/cohere-ai/cohere-python)
      client to use. If provided, `api_key` and `http_client` must be `None`.
    http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
  """
  if cohere_client is not None:
    assert http_client is None, 'Cannot provide both `cohere_client` and `http_client`'
    assert api_key is None, 'Cannot provide both `cohere_client` and `api_key`'
    self._client = cohere_client
  else:
    api_key = api_key or os.environ.get('CO_API_KEY')
    if not api_key:
      raise UserError(
        'Set the `CO_API_KEY` environment variable or pass it via `CohereProvider(api_key=...)`'
        'to use the Cohere provider.'
      )
    base_url = os.environ.get('CO_BASE_URL')
    if http_client is not None:
      self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)
    else:
      http_client = cached_async_http_client(provider='cohere')
      self._client = AsyncClientV2(api_key=api_key, httpx_client=http_client, base_url=base_url)

```
  
---|---  
###  MistralProvider
Bases: `Provider[](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider "pydantic_ai.providers.Provider")[Mistral]`
Provider for Mistral API.
Source code in `pydantic_ai_slim/pydantic_ai/providers/mistral.py`
```
```
| ```
classMistralProvider(Provider[Mistral]):
"""Provider for Mistral API."""
  @property
  defname(self) -> str:
    return 'mistral'
  @property
  defbase_url(self) -> str:
    return self.client.sdk_configuration.get_server_details()[0]
  @property
  defclient(self) -> Mistral:
    return self._client
  @overload
  def__init__(self, *, mistral_client: Mistral | None = None) -> None: ...
  @overload
  def__init__(self, *, api_key: str | None = None, http_client: AsyncHTTPClient | None = None) -> None: ...
  def__init__(
    self,
    *,
    api_key: str | None = None,
    mistral_client: Mistral | None = None,
    http_client: AsyncHTTPClient | None = None,
  ) -> None:
"""Create a new Mistral provider.
    Args:
      api_key: The API key to use for authentication, if not provided, the `MISTRAL_API_KEY` environment variable
        will be used if available.
      mistral_client: An existing `Mistral` client to use, if provided, `api_key` and `http_client` must be `None`.
      http_client: An existing async client to use for making HTTP requests.
    """
    if mistral_client is not None:
      assert http_client is None, 'Cannot provide both `mistral_client` and `http_client`'
      assert api_key is None, 'Cannot provide both `mistral_client` and `api_key`'
      self._client = mistral_client
    else:
      api_key = api_key or os.environ.get('MISTRAL_API_KEY')
      if not api_key:
        raise UserError(
          'Set the `MISTRAL_API_KEY` environment variable or pass it via `MistralProvider(api_key=...)`'
          'to use the Mistral provider.'
        )
      elif http_client is not None:
        self._client = Mistral(api_key=api_key, async_client=http_client)
      else:
        http_client = cached_async_http_client(provider='mistral')
        self._client = Mistral(api_key=api_key, async_client=http_client)

```
  
---|---  
####  __init__
```
__init__(*, mistral_client: Mistral | None = None) -> None

```

```
__init__(
  *,
  api_key:  | None = None,
  http_client: AsyncClient | None = None
) -> None

```

```
__init__(
  *,
  api_key:  | None = None,
  mistral_client: Mistral | None = None,
  http_client: AsyncClient | None = None
) -> None

```

Create a new Mistral provider.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`api_key` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The API key to use for authentication, if not provided, the `MISTRAL_API_KEY` environment variable will be used if available. |  `None`  
`mistral_client` |  `Mistral | None` |  An existing `Mistral` client to use, if provided, `api_key` and `http_client` must be `None`. |  `None`  
`http_client` |  `AsyncClient | None` |  An existing async client to use for making HTTP requests. |  `None`  
Source code in `pydantic_ai_slim/pydantic_ai/providers/mistral.py`
```
```
| ```
def__init__(
  self,
  *,
  api_key: str | None = None,
  mistral_client: Mistral | None = None,
  http_client: AsyncHTTPClient | None = None,
) -> None:
"""Create a new Mistral provider.
  Args:
    api_key: The API key to use for authentication, if not provided, the `MISTRAL_API_KEY` environment variable
      will be used if available.
    mistral_client: An existing `Mistral` client to use, if provided, `api_key` and `http_client` must be `None`.
    http_client: An existing async client to use for making HTTP requests.
  """
  if mistral_client is not None:
    assert http_client is None, 'Cannot provide both `mistral_client` and `http_client`'
    assert api_key is None, 'Cannot provide both `mistral_client` and `api_key`'
    self._client = mistral_client
  else:
    api_key = api_key or os.environ.get('MISTRAL_API_KEY')
    if not api_key:
      raise UserError(
        'Set the `MISTRAL_API_KEY` environment variable or pass it via `MistralProvider(api_key=...)`'
        'to use the Mistral provider.'
      )
    elif http_client is not None:
      self._client = Mistral(api_key=api_key, async_client=http_client)
    else:
      http_client = cached_async_http_client(provider='mistral')
      self._client = Mistral(api_key=api_key, async_client=http_client)

```
  
---|---

# `pydantic_graph.exceptions`
###  GraphSetupError
Bases: 
Error caused by an incorrectly configured graph.
Source code in `pydantic_graph/pydantic_graph/exceptions.py`
```
```
| ```
classGraphSetupError(TypeError):
"""Error caused by an incorrectly configured graph."""
  message: str
"""Description of the mistake."""
  def__init__(self, message: str):
    self.message = message
    super().__init__(message)

```
  
---|---  
####  message `instance-attribute`
```
message:  = message

```

Description of the mistake.
###  GraphRuntimeError
Bases: 
Error caused by an issue during graph execution.
Source code in `pydantic_graph/pydantic_graph/exceptions.py`
```
```
| ```
classGraphRuntimeError(RuntimeError):
"""Error caused by an issue during graph execution."""
  message: str
"""The error message."""
  def__init__(self, message: str):
    self.message = message
    super().__init__(message)

```
  
---|---  
####  message `instance-attribute`
```
message:  = message

```

The error message.
###  GraphNodeStatusError
Bases: 
Error caused by trying to run a node that already has status `'running'`, `'success'`, or `'error'`.
Source code in `pydantic_graph/pydantic_graph/exceptions.py`
```
```
| ```
classGraphNodeStatusError(GraphRuntimeError):
"""Error caused by trying to run a node that already has status `'running'`, `'success'`, or `'error'`."""
  def__init__(self, actual_status: 'SnapshotStatus'):
    self.actual_status = actual_status
    super().__init__(f"Incorrect snapshot status {actual_status!r}, must be 'created' or 'pending'.")
  @classmethod
  defcheck(cls, status: 'SnapshotStatus') -> None:
"""Check if the status is valid."""
    if status not in {'created', 'pending'}:
      raise cls(status)

```
  
---|---  
####  check `classmethod`
```
check(status: ) -> None

```

Check if the status is valid.
Source code in `pydantic_graph/pydantic_graph/exceptions.py`
```
```
| ```
@classmethod
defcheck(cls, status: 'SnapshotStatus') -> None:
"""Check if the status is valid."""
  if status not in {'created', 'pending'}:
    raise cls(status)

```
  
---|---

[ Skip to content ](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph)
# `pydantic_graph`
###  Graph `dataclass`
Bases: `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]`
Definition of a graph.
In `pydantic-graph`, a graph is a collection of nodes that can be run in sequence. The nodes define their outgoing edges — e.g. which nodes may be run next, and thereby the structure of the graph.
Here's a very simple example of a graph which increments a number by 1, but makes sure the number is never 42 at the end.
never_42.py```
from__future__import annotations
fromdataclassesimport dataclass
frompydantic_graphimport BaseNode, End, Graph, GraphRunContext
@dataclass
classMyState:
  number: int
@dataclass
classIncrement(BaseNode[MyState]):
  async defrun(self, ctx: GraphRunContext) -> Check42:
    ctx.state.number += 1
    return Check42()
@dataclass
classCheck42(BaseNode[MyState, None, int]):
  async defrun(self, ctx: GraphRunContext) -> Increment | End[int]:
    if ctx.state.number == 42:
      return Increment()
    else:
      return End(ctx.state.number)
never_42_graph = Graph(nodes=(Increment, Check42))

```

_(This example is complete, it can be run "as is")_
See [`run`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.run) For an example of running graph, and [`mermaid_code`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_code) for an example of generating a mermaid diagram from the graph.
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
@dataclass(init=False)
classGraph(Generic[StateT, DepsT, RunEndT]):
"""Definition of a graph.
  In `pydantic-graph`, a graph is a collection of nodes that can be run in sequence. The nodes define
  their outgoing edges — e.g. which nodes may be run next, and thereby the structure of the graph.
  Here's a very simple example of a graph which increments a number by 1, but makes sure the number is never
  42 at the end.
  ```py {title="never_42.py" noqa="I001" py="3.10"}
  from __future__ import annotations
  from dataclasses import dataclass
  from pydantic_graph import BaseNode, End, Graph, GraphRunContext
  @dataclass
  class MyState:
    number: int
  @dataclass
  class Increment(BaseNode[MyState]):
    async def run(self, ctx: GraphRunContext) -> Check42:
      ctx.state.number += 1
      return Check42()
  @dataclass
  class Check42(BaseNode[MyState, None, int]):
    async def run(self, ctx: GraphRunContext) -> Increment | End[int]:
      if ctx.state.number == 42:
        return Increment()
      else:
        return End(ctx.state.number)
  never_42_graph = Graph(nodes=(Increment, Check42))
  ```
  _(This example is complete, it can be run "as is")_
  See [`run`][pydantic_graph.graph.Graph.run] For an example of running graph, and
  [`mermaid_code`][pydantic_graph.graph.Graph.mermaid_code] for an example of generating a mermaid diagram
  from the graph.
  """
  name: str | None
  node_defs: dict[str, NodeDef[StateT, DepsT, RunEndT]]
  _state_type: type[StateT] | _utils.Unset = field(repr=False)
  _run_end_type: type[RunEndT] | _utils.Unset = field(repr=False)
  auto_instrument: bool = field(repr=False)
  def__init__(
    self,
    *,
    nodes: Sequence[type[BaseNode[StateT, DepsT, RunEndT]]],
    name: str | None = None,
    state_type: type[StateT] | _utils.Unset = _utils.UNSET,
    run_end_type: type[RunEndT] | _utils.Unset = _utils.UNSET,
    auto_instrument: bool = True,
  ):
"""Create a graph from a sequence of nodes.
    Args:
      nodes: The nodes which make up the graph, nodes need to be unique and all be generic in the same
        state type.
      name: Optional name for the graph, if not provided the name will be inferred from the calling frame
        on the first call to a graph method.
      state_type: The type of the state for the graph, this can generally be inferred from `nodes`.
      run_end_type: The type of the result of running the graph, this can generally be inferred from `nodes`.
      auto_instrument: Whether to create a span for the graph run and the execution of each node's run method.
    """
    self.name = name
    self._state_type = state_type
    self._run_end_type = run_end_type
    self.auto_instrument = auto_instrument
    parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
    self.node_defs: dict[str, NodeDef[StateT, DepsT, RunEndT]] = {}
    for node in nodes:
      self._register_node(node, parent_namespace)
    self._validate_edges()
  async defrun(
    self,
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
    infer_name: bool = True,
    span: LogfireSpan | None = None,
  ) -> GraphRunResult[StateT, RunEndT]:
"""Run the graph from a starting node until it ends.
    Args:
      start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
        you need to provide the starting node.
      state: The initial state of the graph.
      deps: The dependencies of the graph.
      persistence: State persistence interface, defaults to
        [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
      infer_name: Whether to infer the graph name from the calling frame.
      span: The span to use for the graph run. If not provided, a span will be created depending on the value of
        the `auto_instrument` field.
    Returns:
      A `GraphRunResult` containing information about the run, including its final result.
    Here's an example of running the graph from [above][pydantic_graph.graph.Graph]:
    ```py {title="run_never_42.py" noqa="I001" py="3.10"}
    from never_42 import Increment, MyState, never_42_graph
    async def main():
      state = MyState(1)
      await never_42_graph.run(Increment(), state=state)
      print(state)
      #> MyState(number=2)
      state = MyState(41)
      await never_42_graph.run(Increment(), state=state)
      print(state)
      #> MyState(number=43)
    ```
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    async with self.iter(
      start_node, state=state, deps=deps, persistence=persistence, span=span, infer_name=False
    ) as graph_run:
      async for _node in graph_run:
        pass
    final_result = graph_run.result
    assert final_result is not None, 'GraphRun should have a final result'
    return final_result
  defrun_sync(
    self,
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
    infer_name: bool = True,
  ) -> GraphRunResult[StateT, RunEndT]:
"""Synchronously run the graph.
    This is a convenience method that wraps [`self.run`][pydantic_graph.Graph.run] with `loop.run_until_complete(...)`.
    You therefore can't use this method inside async code or if there's an active event loop.
    Args:
      start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
        you need to provide the starting node.
      state: The initial state of the graph.
      deps: The dependencies of the graph.
      persistence: State persistence interface, defaults to
        [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
      infer_name: Whether to infer the graph name from the calling frame.
    Returns:
      The result type from ending the run and the history of the run.
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    return _utils.get_event_loop().run_until_complete(
      self.run(start_node, state=state, deps=deps, persistence=persistence, infer_name=False)
    )
  @asynccontextmanager
  async defiter(
    self,
    start_node: BaseNode[StateT, DepsT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
    span: AbstractContextManager[Any] | None = None,
    infer_name: bool = True,
  ) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
"""A contextmanager which can be used to iterate over the graph's nodes as they are executed.
    This method returns a `GraphRun` object which can be used to async-iterate over the nodes of this `Graph` as
    they are executed. This is the API to use if you want to record or interact with the nodes as the graph
    execution unfolds.
    The `GraphRun` can also be used to manually drive the graph execution by calling
    [`GraphRun.next`][pydantic_graph.graph.GraphRun.next].
    The `GraphRun` provides access to the full run history, state, deps, and the final result of the run once
    it has completed.
    For more details, see the API documentation of [`GraphRun`][pydantic_graph.graph.GraphRun].
    Args:
      start_node: the first node to run. Since the graph definition doesn't define the entry point in the graph,
        you need to provide the starting node.
      state: The initial state of the graph.
      deps: The dependencies of the graph.
      persistence: State persistence interface, defaults to
        [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
      span: The span to use for the graph run. If not provided, a new span will be created.
      infer_name: Whether to infer the graph name from the calling frame.
    Returns: A GraphRun that can be async iterated over to drive the graph to completion.
    """
    if infer_name and self.name is None:
      # f_back because `asynccontextmanager` adds one frame
      if frame := inspect.currentframe(): # pragma: no branch
        self._infer_name(frame.f_back)
    if persistence is None:
      persistence = SimpleStatePersistence()
    persistence.set_graph_types(self)
    if self.auto_instrument and span is None:
      span = logfire_api.span('run graph {graph.name}', graph=self)
    with ExitStack() as stack:
      if span is not None:
        stack.enter_context(span)
      yield GraphRun[StateT, DepsT, RunEndT](
        graph=self, start_node=start_node, persistence=persistence, state=state, deps=deps
      )
  @asynccontextmanager
  async defiter_from_persistence(
    self,
    persistence: BaseStatePersistence[StateT, RunEndT],
    *,
    deps: DepsT = None,
    span: AbstractContextManager[Any] | None = None,
    infer_name: bool = True,
  ) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
"""A contextmanager to iterate over the graph's nodes as they are executed, created from a persistence object.
    This method has similar functionality to [`iter`][pydantic_graph.graph.Graph.iter],
    but instead of passing the node to run, it will restore the node and state from state persistence.
    Args:
      persistence: The state persistence interface to use.
      deps: The dependencies of the graph.
      span: The span to use for the graph run. If not provided, a new span will be created.
      infer_name: Whether to infer the graph name from the calling frame.
    Returns: A GraphRun that can be async iterated over to drive the graph to completion.
    """
    if infer_name and self.name is None:
      # f_back because `asynccontextmanager` adds one frame
      if frame := inspect.currentframe(): # pragma: no branch
        self._infer_name(frame.f_back)
    persistence.set_graph_types(self)
    snapshot = await persistence.load_next()
    if snapshot is None:
      raise exceptions.GraphRuntimeError('Unable to restore snapshot from state persistence.')
    snapshot.node.set_snapshot_id(snapshot.id)
    if self.auto_instrument and span is None:
      span = logfire_api.span('run graph {graph.name}', graph=self)
    with ExitStack() as stack:
      if span is not None:
        stack.enter_context(span)
      yield GraphRun[StateT, DepsT, RunEndT](
        graph=self,
        start_node=snapshot.node,
        persistence=persistence,
        state=snapshot.state,
        deps=deps,
        snapshot_id=snapshot.id,
      )
  async definitialize(
    self,
    node: BaseNode[StateT, DepsT, RunEndT],
    persistence: BaseStatePersistence[StateT, RunEndT],
    *,
    state: StateT = None,
    infer_name: bool = True,
  ) -> None:
"""Initialize a new graph run in persistence without running it.
    This is useful if you want to set up a graph run to be run later, e.g. via
    [`iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence].
    Args:
      node: The node to run first.
      persistence: State persistence interface.
      state: The start state of the graph.
      infer_name: Whether to infer the graph name from the calling frame.
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    persistence.set_graph_types(self)
    await persistence.snapshot_node(state, node)
  @deprecated('`next` is deprecated, use `async with graph.iter(...) as run: run.next()` instead')
  async defnext(
    self,
    node: BaseNode[StateT, DepsT, RunEndT],
    persistence: BaseStatePersistence[StateT, RunEndT],
    *,
    state: StateT = None,
    deps: DepsT = None,
    infer_name: bool = True,
  ) -> BaseNode[StateT, DepsT, Any] | End[RunEndT]:
"""Run a node in the graph and return the next node to run.
    Args:
      node: The node to run.
      persistence: State persistence interface, defaults to
        [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
      state: The current state of the graph.
      deps: The dependencies of the graph.
      infer_name: Whether to infer the graph name from the calling frame.
    Returns:
      The next node to run or [`End`][pydantic_graph.nodes.End] if the graph has finished.
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    persistence.set_graph_types(self)
    run = GraphRun[StateT, DepsT, RunEndT](
      graph=self,
      start_node=node,
      persistence=persistence,
      state=state,
      deps=deps,
    )
    return await run.next(node)
  defmermaid_code(
    self,
    *,
    start_node: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
    title: str | None | typing_extensions.Literal[False] = None,
    edge_labels: bool = True,
    notes: bool = True,
    highlighted_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
    highlight_css: str = mermaid.DEFAULT_HIGHLIGHT_CSS,
    infer_name: bool = True,
    direction: mermaid.StateDiagramDirection | None = None,
  ) -> str:
"""Generate a diagram representing the graph as [mermaid](https://mermaid.js.org/) diagram.
    This method calls [`pydantic_graph.mermaid.generate_code`][pydantic_graph.mermaid.generate_code].
    Args:
      start_node: The node or nodes which can start the graph.
      title: The title of the diagram, use `False` to not include a title.
      edge_labels: Whether to include edge labels.
      notes: Whether to include notes on each node.
      highlighted_nodes: Optional node or nodes to highlight.
      highlight_css: The CSS to use for highlighting nodes.
      infer_name: Whether to infer the graph name from the calling frame.
      direction: The direction of flow.
    Returns:
      The mermaid code for the graph, which can then be rendered as a diagram.
    Here's an example of generating a diagram for the graph from [above][pydantic_graph.graph.Graph]:
    ```py {title="mermaid_never_42.py" py="3.10"}
    from never_42 import Increment, never_42_graph
    print(never_42_graph.mermaid_code(start_node=Increment))
    '''
    ---
    title: never_42_graph
    ---
    stateDiagram-v2
     [*] --> Increment
     Increment --> Check42
     Check42 --> Increment
     Check42 --> [*]
    '''
    ```
    The rendered diagram will look like this:
    ```mermaid
    ---
    title: never_42_graph
    ---
    stateDiagram-v2
     [*] --> Increment
     Increment --> Check42
     Check42 --> Increment
     Check42 --> [*]
    ```
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    if title is None and self.name:
      title = self.name
    return mermaid.generate_code(
      self,
      start_node=start_node,
      highlighted_nodes=highlighted_nodes,
      highlight_css=highlight_css,
      title=title or None,
      edge_labels=edge_labels,
      notes=notes,
      direction=direction,
    )
  defmermaid_image(
    self, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
  ) -> bytes:
"""Generate a diagram representing the graph as an image.
    The format and diagram can be customized using `kwargs`,
    see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].
    !!! note "Uses external service"
      This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
      is a free service not affiliated with Pydantic.
    Args:
      infer_name: Whether to infer the graph name from the calling frame.
      **kwargs: Additional arguments to pass to `mermaid.request_image`.
    Returns:
      The image bytes.
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    if 'title' not in kwargs and self.name:
      kwargs['title'] = self.name
    return mermaid.request_image(self, **kwargs)
  defmermaid_save(
    self, path: Path | str, /, *, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
  ) -> None:
"""Generate a diagram representing the graph and save it as an image.
    The format and diagram can be customized using `kwargs`,
    see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].
    !!! note "Uses external service"
      This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
      is a free service not affiliated with Pydantic.
    Args:
      path: The path to save the image to.
      infer_name: Whether to infer the graph name from the calling frame.
      **kwargs: Additional arguments to pass to `mermaid.save_image`.
    """
    if infer_name and self.name is None:
      self._infer_name(inspect.currentframe())
    if 'title' not in kwargs and self.name:
      kwargs['title'] = self.name
    mermaid.save_image(path, self, **kwargs)
  defget_nodes(self) -> Sequence[type[BaseNode[StateT, DepsT, RunEndT]]]:
"""Get the nodes in the graph."""
    return [node_def.node for node_def in self.node_defs.values()]
  @cached_property
  definferred_types(self) -> tuple[type[StateT], type[RunEndT]]:
    # Get the types of the state and run end from the graph.
    if _utils.is_set(self._state_type) and _utils.is_set(self._run_end_type):
      return self._state_type, self._run_end_type
    state_type = self._state_type
    run_end_type = self._run_end_type
    for node_def in self.node_defs.values():
      for base in typing_extensions.get_original_bases(node_def.node):
        if typing_extensions.get_origin(base) is BaseNode:
          args = typing_extensions.get_args(base)
          if not _utils.is_set(state_type) and args:
            state_type = args[0]
          if not _utils.is_set(run_end_type) and len(args) == 3:
            t = args[2]
            if not typing_objects.is_never(t):
              run_end_type = t
          if _utils.is_set(state_type) and _utils.is_set(run_end_type):
            return state_type, run_end_type # pyright: ignore[reportReturnType]
          # break the inner (bases) loop
          break
    if not _utils.is_set(state_type):
      # state defaults to None, so use that if we can't infer it
      state_type = None
    if not _utils.is_set(run_end_type):
      # this happens if a graph has no return nodes, use None so any downstream errors are clear
      run_end_type = None
    return state_type, run_end_type # pyright: ignore[reportReturnType]
  def_register_node(
    self,
    node: type[BaseNode[StateT, DepsT, RunEndT]],
    parent_namespace: dict[str, Any] | None,
  ) -> None:
    node_id = node.get_node_id()
    if existing_node := self.node_defs.get(node_id):
      raise exceptions.GraphSetupError(
        f'Node ID `{node_id}` is not unique — found on {existing_node.node} and {node}'
      )
    else:
      self.node_defs[node_id] = node.get_node_def(parent_namespace)
  def_validate_edges(self):
    known_node_ids = self.node_defs.keys()
    bad_edges: dict[str, list[str]] = {}
    for node_id, node_def in self.node_defs.items():
      for edge in node_def.next_node_edges.keys():
        if edge not in known_node_ids:
          bad_edges.setdefault(edge, []).append(f'`{node_id}`')
    if bad_edges:
      bad_edges_list = [f'`{k}` is referenced by {_utils.comma_and(v)}' for k, v in bad_edges.items()]
      if len(bad_edges_list) == 1:
        raise exceptions.GraphSetupError(f'{bad_edges_list[0]} but not included in the graph.')
      else:
        b = '\n'.join(f' {be}' for be in bad_edges_list)
        raise exceptions.GraphSetupError(
          f'Nodes are referenced in the graph but not included in the graph:\n{b}'
        )
  def_infer_name(self, function_frame: types.FrameType | None) -> None:
"""Infer the agent name from the call frame.
    Usage should be `self._infer_name(inspect.currentframe())`.
    Copied from `Agent`.
    """
    assert self.name is None, 'Name already set'
    if function_frame is not None and (parent_frame := function_frame.f_back): # pragma: no branch
      for name, item in parent_frame.f_locals.items():
        if item is self:
          self.name = name
          return
      if parent_frame.f_locals != parent_frame.f_globals:
        # if we couldn't find the agent in locals and globals are a different dict, try globals
        for name, item in parent_frame.f_globals.items():
          if item is self:
            self.name = name
            return

```
  
---|---  
####  __init__
```
__init__(
  *,
  nodes: [[[, , ]]],
  name:  | None = None,
  state_type: [] |  = ,
  run_end_type: [] |  = ,
  auto_instrument:  = True
)

```

Create a graph from a sequence of nodes.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`nodes` |  `Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[type[](https://docs.python.org/3/library/functions.html#type)[BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]]]` |  The nodes which make up the graph, nodes need to be unique and all be generic in the same state type. |  _required_  
`name` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  Optional name for the graph, if not provided the name will be inferred from the calling frame on the first call to a graph method. |  `None`  
`state_type` |  `type[](https://docs.python.org/3/library/functions.html#type)[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT")] | Unset` |  The type of the state for the graph, this can generally be inferred from `nodes`. |  `UNSET`  
`run_end_type` |  `type[](https://docs.python.org/3/library/functions.html#type)[RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")] | Unset` |  The type of the result of running the graph, this can generally be inferred from `nodes`. |  `UNSET`  
`auto_instrument` |  Whether to create a span for the graph run and the execution of each node's run method. |  `True`  
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
def__init__(
  self,
  *,
  nodes: Sequence[type[BaseNode[StateT, DepsT, RunEndT]]],
  name: str | None = None,
  state_type: type[StateT] | _utils.Unset = _utils.UNSET,
  run_end_type: type[RunEndT] | _utils.Unset = _utils.UNSET,
  auto_instrument: bool = True,
):
"""Create a graph from a sequence of nodes.
  Args:
    nodes: The nodes which make up the graph, nodes need to be unique and all be generic in the same
      state type.
    name: Optional name for the graph, if not provided the name will be inferred from the calling frame
      on the first call to a graph method.
    state_type: The type of the state for the graph, this can generally be inferred from `nodes`.
    run_end_type: The type of the result of running the graph, this can generally be inferred from `nodes`.
    auto_instrument: Whether to create a span for the graph run and the execution of each node's run method.
  """
  self.name = name
  self._state_type = state_type
  self._run_end_type = run_end_type
  self.auto_instrument = auto_instrument
  parent_namespace = _utils.get_parent_namespace(inspect.currentframe())
  self.node_defs: dict[str, NodeDef[StateT, DepsT, RunEndT]] = {}
  for node in nodes:
    self._register_node(node, parent_namespace)
  self._validate_edges()

```
  
---|---  
####  run `async`
```
run(
  start_node: [, , ],
  *,
  state:  = None,
  deps:  = None,
  persistence: (
    [, ] | None
  ) = None,
  infer_name:  = True,
  span: LogfireSpan | None = None
) -> [, ]

```

Run the graph from a starting node until it ends.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`start_node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  the first node to run, since the graph definition doesn't define the entry point in the graph, you need to provide the starting node. |  _required_  
`state` |  |  The initial state of the graph. |  `None`  
`deps` |  |  The dependencies of the graph. |  `None`  
`persistence` |  `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")] | None` |  State persistence interface, defaults to [`SimpleStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.in_mem.SimpleStatePersistence) if `None`. |  `None`  
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
`span` |  `LogfireSpan | None` |  The span to use for the graph run. If not provided, a span will be created depending on the value of the `auto_instrument` field. |  `None`  
Returns:
Type | Description  
---|---  
`GraphRunResult[](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.GraphRunResult "pydantic_graph.graph.GraphRunResult")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  A `GraphRunResult` containing information about the run, including its final result.  
Here's an example of running the graph from [above](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph):
run_never_42.py```
fromnever_42import Increment, MyState, never_42_graph
async defmain():
  state = MyState(1)
  await never_42_graph.run(Increment(), state=state)
  print(state)
  #> MyState(number=2)
  state = MyState(41)
  await never_42_graph.run(Increment(), state=state)
  print(state)
  #> MyState(number=43)

```

Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
async defrun(
  self,
  start_node: BaseNode[StateT, DepsT, RunEndT],
  *,
  state: StateT = None,
  deps: DepsT = None,
  persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
  infer_name: bool = True,
  span: LogfireSpan | None = None,
) -> GraphRunResult[StateT, RunEndT]:
"""Run the graph from a starting node until it ends.
  Args:
    start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
      you need to provide the starting node.
    state: The initial state of the graph.
    deps: The dependencies of the graph.
    persistence: State persistence interface, defaults to
      [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
    infer_name: Whether to infer the graph name from the calling frame.
    span: The span to use for the graph run. If not provided, a span will be created depending on the value of
      the `auto_instrument` field.
  Returns:
    A `GraphRunResult` containing information about the run, including its final result.
  Here's an example of running the graph from [above][pydantic_graph.graph.Graph]:
  ```py {title="run_never_42.py" noqa="I001" py="3.10"}
  from never_42 import Increment, MyState, never_42_graph
  async def main():
    state = MyState(1)
    await never_42_graph.run(Increment(), state=state)
    print(state)
    #> MyState(number=2)
    state = MyState(41)
    await never_42_graph.run(Increment(), state=state)
    print(state)
    #> MyState(number=43)
  ```
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  async with self.iter(
    start_node, state=state, deps=deps, persistence=persistence, span=span, infer_name=False
  ) as graph_run:
    async for _node in graph_run:
      pass
  final_result = graph_run.result
  assert final_result is not None, 'GraphRun should have a final result'
  return final_result

```
  
---|---  
####  run_sync
```
run_sync(
  start_node: [, , ],
  *,
  state:  = None,
  deps:  = None,
  persistence: (
    [, ] | None
  ) = None,
  infer_name:  = True
) -> [, ]

```

Synchronously run the graph.
This is a convenience method that wraps [`self.run`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.run) with `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`start_node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  the first node to run, since the graph definition doesn't define the entry point in the graph, you need to provide the starting node. |  _required_  
`state` |  |  The initial state of the graph. |  `None`  
`deps` |  |  The dependencies of the graph. |  `None`  
`persistence` |  `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")] | None` |  State persistence interface, defaults to [`SimpleStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.in_mem.SimpleStatePersistence) if `None`. |  `None`  
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
Returns:
Type | Description  
---|---  
`GraphRunResult[](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.GraphRunResult "pydantic_graph.graph.GraphRunResult")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  The result type from ending the run and the history of the run.  
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
defrun_sync(
  self,
  start_node: BaseNode[StateT, DepsT, RunEndT],
  *,
  state: StateT = None,
  deps: DepsT = None,
  persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
  infer_name: bool = True,
) -> GraphRunResult[StateT, RunEndT]:
"""Synchronously run the graph.
  This is a convenience method that wraps [`self.run`][pydantic_graph.Graph.run] with `loop.run_until_complete(...)`.
  You therefore can't use this method inside async code or if there's an active event loop.
  Args:
    start_node: the first node to run, since the graph definition doesn't define the entry point in the graph,
      you need to provide the starting node.
    state: The initial state of the graph.
    deps: The dependencies of the graph.
    persistence: State persistence interface, defaults to
      [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
    infer_name: Whether to infer the graph name from the calling frame.
  Returns:
    The result type from ending the run and the history of the run.
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  return _utils.get_event_loop().run_until_complete(
    self.run(start_node, state=state, deps=deps, persistence=persistence, infer_name=False)
  )

```
  
---|---  
####  iter `async`
```
iter(
  start_node: [, , ],
  *,
  state:  = None,
  deps:  = None,
  persistence: (
    [, ] | None
  ) = None,
  span: [] | None = None,
  infer_name:  = True
) -> [[, , ]]

```

A contextmanager which can be used to iterate over the graph's nodes as they are executed.
This method returns a `GraphRun` object which can be used to async-iterate over the nodes of this `Graph` as they are executed. This is the API to use if you want to record or interact with the nodes as the graph execution unfolds.
The `GraphRun` can also be used to manually drive the graph execution by calling [`GraphRun.next`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.GraphRun.next).
The `GraphRun` provides access to the full run history, state, deps, and the final result of the run once it has completed.
For more details, see the API documentation of [`GraphRun`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.GraphRun).
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`start_node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  the first node to run. Since the graph definition doesn't define the entry point in the graph, you need to provide the starting node. |  _required_  
`state` |  |  The initial state of the graph. |  `None`  
`deps` |  |  The dependencies of the graph. |  `None`  
`persistence` |  `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")] | None` |  State persistence interface, defaults to [`SimpleStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.in_mem.SimpleStatePersistence) if `None`. |  `None`  
`span` |  `AbstractContextManager[](https://docs.python.org/3/library/contextlib.html#contextlib.AbstractContextManager "contextlib.AbstractContextManager")[Any[](https://docs.python.org/3/library/typing.html#typing.Any "typing.Any")] | None` |  The span to use for the graph run. If not provided, a new span will be created. |  `None`  
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
Returns: A GraphRun that can be async iterated over to drive the graph to completion.
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
@asynccontextmanager
async defiter(
  self,
  start_node: BaseNode[StateT, DepsT, RunEndT],
  *,
  state: StateT = None,
  deps: DepsT = None,
  persistence: BaseStatePersistence[StateT, RunEndT] | None = None,
  span: AbstractContextManager[Any] | None = None,
  infer_name: bool = True,
) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
"""A contextmanager which can be used to iterate over the graph's nodes as they are executed.
  This method returns a `GraphRun` object which can be used to async-iterate over the nodes of this `Graph` as
  they are executed. This is the API to use if you want to record or interact with the nodes as the graph
  execution unfolds.
  The `GraphRun` can also be used to manually drive the graph execution by calling
  [`GraphRun.next`][pydantic_graph.graph.GraphRun.next].
  The `GraphRun` provides access to the full run history, state, deps, and the final result of the run once
  it has completed.
  For more details, see the API documentation of [`GraphRun`][pydantic_graph.graph.GraphRun].
  Args:
    start_node: the first node to run. Since the graph definition doesn't define the entry point in the graph,
      you need to provide the starting node.
    state: The initial state of the graph.
    deps: The dependencies of the graph.
    persistence: State persistence interface, defaults to
      [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
    span: The span to use for the graph run. If not provided, a new span will be created.
    infer_name: Whether to infer the graph name from the calling frame.
  Returns: A GraphRun that can be async iterated over to drive the graph to completion.
  """
  if infer_name and self.name is None:
    # f_back because `asynccontextmanager` adds one frame
    if frame := inspect.currentframe(): # pragma: no branch
      self._infer_name(frame.f_back)
  if persistence is None:
    persistence = SimpleStatePersistence()
  persistence.set_graph_types(self)
  if self.auto_instrument and span is None:
    span = logfire_api.span('run graph {graph.name}', graph=self)
  with ExitStack() as stack:
    if span is not None:
      stack.enter_context(span)
    yield GraphRun[StateT, DepsT, RunEndT](
      graph=self, start_node=start_node, persistence=persistence, state=state, deps=deps
    )

```
  
---|---  
####  iter_from_persistence `async`
```
iter_from_persistence(
  persistence: [, ],
  *,
  deps:  = None,
  span: [] | None = None,
  infer_name:  = True
) -> [[, , ]]

```

A contextmanager to iterate over the graph's nodes as they are executed, created from a persistence object.
This method has similar functionality to [`iter`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.iter), but instead of passing the node to run, it will restore the node and state from state persistence.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`persistence` |  `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  The state persistence interface to use. |  _required_  
`deps` |  |  The dependencies of the graph. |  `None`  
`span` |  `AbstractContextManager[](https://docs.python.org/3/library/contextlib.html#contextlib.AbstractContextManager "contextlib.AbstractContextManager")[Any[](https://docs.python.org/3/library/typing.html#typing.Any "typing.Any")] | None` |  The span to use for the graph run. If not provided, a new span will be created. |  `None`  
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
Returns: A GraphRun that can be async iterated over to drive the graph to completion.
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
@asynccontextmanager
async defiter_from_persistence(
  self,
  persistence: BaseStatePersistence[StateT, RunEndT],
  *,
  deps: DepsT = None,
  span: AbstractContextManager[Any] | None = None,
  infer_name: bool = True,
) -> AsyncIterator[GraphRun[StateT, DepsT, RunEndT]]:
"""A contextmanager to iterate over the graph's nodes as they are executed, created from a persistence object.
  This method has similar functionality to [`iter`][pydantic_graph.graph.Graph.iter],
  but instead of passing the node to run, it will restore the node and state from state persistence.
  Args:
    persistence: The state persistence interface to use.
    deps: The dependencies of the graph.
    span: The span to use for the graph run. If not provided, a new span will be created.
    infer_name: Whether to infer the graph name from the calling frame.
  Returns: A GraphRun that can be async iterated over to drive the graph to completion.
  """
  if infer_name and self.name is None:
    # f_back because `asynccontextmanager` adds one frame
    if frame := inspect.currentframe(): # pragma: no branch
      self._infer_name(frame.f_back)
  persistence.set_graph_types(self)
  snapshot = await persistence.load_next()
  if snapshot is None:
    raise exceptions.GraphRuntimeError('Unable to restore snapshot from state persistence.')
  snapshot.node.set_snapshot_id(snapshot.id)
  if self.auto_instrument and span is None:
    span = logfire_api.span('run graph {graph.name}', graph=self)
  with ExitStack() as stack:
    if span is not None:
      stack.enter_context(span)
    yield GraphRun[StateT, DepsT, RunEndT](
      graph=self,
      start_node=snapshot.node,
      persistence=persistence,
      state=snapshot.state,
      deps=deps,
      snapshot_id=snapshot.id,
    )

```
  
---|---  
####  initialize `async`
```
initialize(
  node: [, , ],
  persistence: [, ],
  *,
  state:  = None,
  infer_name:  = True
) -> None

```

Initialize a new graph run in persistence without running it.
This is useful if you want to set up a graph run to be run later, e.g. via [`iter_from_persistence`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.iter_from_persistence).
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  The node to run first. |  _required_  
`persistence` |  `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  State persistence interface. |  _required_  
`state` |  |  The start state of the graph. |  `None`  
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
async definitialize(
  self,
  node: BaseNode[StateT, DepsT, RunEndT],
  persistence: BaseStatePersistence[StateT, RunEndT],
  *,
  state: StateT = None,
  infer_name: bool = True,
) -> None:
"""Initialize a new graph run in persistence without running it.
  This is useful if you want to set up a graph run to be run later, e.g. via
  [`iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence].
  Args:
    node: The node to run first.
    persistence: State persistence interface.
    state: The start state of the graph.
    infer_name: Whether to infer the graph name from the calling frame.
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  persistence.set_graph_types(self)
  await persistence.snapshot_node(state, node)

```
  
---|---  
####  next `async`
```
next(
  node: [, , ],
  persistence: [, ],
  *,
  state:  = None,
  deps:  = None,
  infer_name:  = True
) -> [, , ] | []

```

Run a node in the graph and return the next node to run.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  The node to run. |  _required_  
`persistence` |  `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  State persistence interface, defaults to [`SimpleStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.in_mem.SimpleStatePersistence) if `None`. |  _required_  
`state` |  |  The current state of the graph. |  `None`  
`deps` |  |  The dependencies of the graph. |  `None`  
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
Returns:
Type | Description  
---|---  
`BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), Any[](https://docs.python.org/3/library/typing.html#typing.Any "typing.Any")] | End[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End "pydantic_graph.nodes.End")[RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  The next node to run or [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) if the graph has finished.  
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
@deprecated('`next` is deprecated, use `async with graph.iter(...) as run: run.next()` instead')
async defnext(
  self,
  node: BaseNode[StateT, DepsT, RunEndT],
  persistence: BaseStatePersistence[StateT, RunEndT],
  *,
  state: StateT = None,
  deps: DepsT = None,
  infer_name: bool = True,
) -> BaseNode[StateT, DepsT, Any] | End[RunEndT]:
"""Run a node in the graph and return the next node to run.
  Args:
    node: The node to run.
    persistence: State persistence interface, defaults to
      [`SimpleStatePersistence`][pydantic_graph.SimpleStatePersistence] if `None`.
    state: The current state of the graph.
    deps: The dependencies of the graph.
    infer_name: Whether to infer the graph name from the calling frame.
  Returns:
    The next node to run or [`End`][pydantic_graph.nodes.End] if the graph has finished.
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  persistence.set_graph_types(self)
  run = GraphRun[StateT, DepsT, RunEndT](
    graph=self,
    start_node=node,
    persistence=persistence,
    state=state,
    deps=deps,
  )
  return await run.next(node)

```
  
---|---  
####  mermaid_code
```
mermaid_code(
  *,
  start_node: (
    [] |  | None
  ) = None,
  title:  | None | [False] = None,
  edge_labels:  = True,
  notes:  = True,
  highlighted_nodes: (
    [] |  | None
  ) = None,
  highlight_css:  = ,
  infer_name:  = True,
  direction:  | None = None
) -> 

```

Generate a diagram representing the graph as [mermaid](https://mermaid.js.org/) diagram.
This method calls [`pydantic_graph.mermaid.generate_code`](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.generate_code).
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`start_node` |  `Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[NodeIdent[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.NodeIdent "pydantic_graph.mermaid.NodeIdent")] | NodeIdent[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.NodeIdent "pydantic_graph.mermaid.NodeIdent") | None` |  The node or nodes which can start the graph. |  `None`  
`title` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None | Literal[](https://typing-extensions.readthedocs.io/en/latest/index.html#typing_extensions.Literal "typing_extensions.Literal")[False]` |  The title of the diagram, use `False` to not include a title. |  `None`  
`edge_labels` |  Whether to include edge labels. |  `True`  
`notes` |  Whether to include notes on each node. |  `True`  
`highlighted_nodes` |  `Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[NodeIdent[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.NodeIdent "pydantic_graph.mermaid.NodeIdent")] | NodeIdent[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.NodeIdent "pydantic_graph.mermaid.NodeIdent") | None` |  Optional node or nodes to highlight. |  `None`  
`highlight_css` |  The CSS to use for highlighting nodes. |   
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
`direction` |  `StateDiagramDirection[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.StateDiagramDirection "pydantic_graph.mermaid.StateDiagramDirection") | None` |  The direction of flow. |  `None`  
Returns:
Type | Description  
---|---  
The mermaid code for the graph, which can then be rendered as a diagram.  
Here's an example of generating a diagram for the graph from [above](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph):
mermaid_never_42.py```
fromnever_42import Increment, never_42_graph
print(never_42_graph.mermaid_code(start_node=Increment))
'''
---
title: never_42_graph
---
stateDiagram-v2
 [*] --> Increment
 Increment --> Check42
 Check42 --> Increment
 Check42 --> [*]
'''

```

The rendered diagram will look like this:
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
defmermaid_code(
  self,
  *,
  start_node: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
  title: str | None | typing_extensions.Literal[False] = None,
  edge_labels: bool = True,
  notes: bool = True,
  highlighted_nodes: Sequence[mermaid.NodeIdent] | mermaid.NodeIdent | None = None,
  highlight_css: str = mermaid.DEFAULT_HIGHLIGHT_CSS,
  infer_name: bool = True,
  direction: mermaid.StateDiagramDirection | None = None,
) -> str:
"""Generate a diagram representing the graph as [mermaid](https://mermaid.js.org/) diagram.
  This method calls [`pydantic_graph.mermaid.generate_code`][pydantic_graph.mermaid.generate_code].
  Args:
    start_node: The node or nodes which can start the graph.
    title: The title of the diagram, use `False` to not include a title.
    edge_labels: Whether to include edge labels.
    notes: Whether to include notes on each node.
    highlighted_nodes: Optional node or nodes to highlight.
    highlight_css: The CSS to use for highlighting nodes.
    infer_name: Whether to infer the graph name from the calling frame.
    direction: The direction of flow.
  Returns:
    The mermaid code for the graph, which can then be rendered as a diagram.
  Here's an example of generating a diagram for the graph from [above][pydantic_graph.graph.Graph]:
  ```py {title="mermaid_never_42.py" py="3.10"}
  from never_42 import Increment, never_42_graph
  print(never_42_graph.mermaid_code(start_node=Increment))
  '''
  ---
  title: never_42_graph
  ---
  stateDiagram-v2
   [*] --> Increment
   Increment --> Check42
   Check42 --> Increment
   Check42 --> [*]
  '''
  ```
  The rendered diagram will look like this:
  ```mermaid
  ---
  title: never_42_graph
  ---
  stateDiagram-v2
   [*] --> Increment
   Increment --> Check42
   Check42 --> Increment
   Check42 --> [*]
  ```
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  if title is None and self.name:
    title = self.name
  return mermaid.generate_code(
    self,
    start_node=start_node,
    highlighted_nodes=highlighted_nodes,
    highlight_css=highlight_css,
    title=title or None,
    edge_labels=edge_labels,
    notes=notes,
    direction=direction,
  )

```
  
---|---  
####  mermaid_image
```
mermaid_image(
  infer_name:  = True, **kwargs: []
) -> 

```

Generate a diagram representing the graph as an image.
The format and diagram can be customized using `kwargs`, see [`pydantic_graph.mermaid.MermaidConfig`](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.MermaidConfig).
Uses external service
This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink` is a free service not affiliated with Pydantic.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
`**kwargs` |  |  Additional arguments to pass to `mermaid.request_image`.  
Returns:
Type | Description  
---|---  
|  The image bytes.  
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
defmermaid_image(
  self, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
) -> bytes:
"""Generate a diagram representing the graph as an image.
  The format and diagram can be customized using `kwargs`,
  see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].
  !!! note "Uses external service"
    This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
    is a free service not affiliated with Pydantic.
  Args:
    infer_name: Whether to infer the graph name from the calling frame.
    **kwargs: Additional arguments to pass to `mermaid.request_image`.
  Returns:
    The image bytes.
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  if 'title' not in kwargs and self.name:
    kwargs['title'] = self.name
  return mermaid.request_image(self, **kwargs)

```
  
---|---  
####  mermaid_save
```
mermaid_save(
  path:  | ,
  /,
  *,
  infer_name:  = True,
  **kwargs: [],
) -> None

```

Generate a diagram representing the graph and save it as an image.
The format and diagram can be customized using `kwargs`, see [`pydantic_graph.mermaid.MermaidConfig`](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.MermaidConfig).
Uses external service
This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink` is a free service not affiliated with Pydantic.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`path` |  |  The path to save the image to. |  _required_  
`infer_name` |  Whether to infer the graph name from the calling frame. |  `True`  
`**kwargs` |  |  Additional arguments to pass to `mermaid.save_image`.  
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
defmermaid_save(
  self, path: Path | str, /, *, infer_name: bool = True, **kwargs: typing_extensions.Unpack[mermaid.MermaidConfig]
) -> None:
"""Generate a diagram representing the graph and save it as an image.
  The format and diagram can be customized using `kwargs`,
  see [`pydantic_graph.mermaid.MermaidConfig`][pydantic_graph.mermaid.MermaidConfig].
  !!! note "Uses external service"
    This method makes a request to [mermaid.ink](https://mermaid.ink) to render the image, `mermaid.ink`
    is a free service not affiliated with Pydantic.
  Args:
    path: The path to save the image to.
    infer_name: Whether to infer the graph name from the calling frame.
    **kwargs: Additional arguments to pass to `mermaid.save_image`.
  """
  if infer_name and self.name is None:
    self._infer_name(inspect.currentframe())
  if 'title' not in kwargs and self.name:
    kwargs['title'] = self.name
  mermaid.save_image(path, self, **kwargs)

```
  
---|---  
####  get_nodes
```
get_nodes() -> (
  [[[, , ]]]
)

```

Get the nodes in the graph.
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
defget_nodes(self) -> Sequence[type[BaseNode[StateT, DepsT, RunEndT]]]:
"""Get the nodes in the graph."""
  return [node_def.node for node_def in self.node_defs.values()]

```
  
---|---  
###  GraphRun
Bases: `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]`
A stateful, async-iterable run of a [`Graph`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph).
You typically get a `GraphRun` instance from calling `async with [my_graph.iter(...)][pydantic_graph.graph.Graph.iter] as graph_run:`. That gives you the ability to iterate through nodes as they run, either by `async for` iteration or by repeatedly calling `.next(...)`.
Here's an example of iterating over the graph from [above](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph): 
iter_never_42.py```
fromcopyimport deepcopy
fromnever_42import Increment, MyState, never_42_graph
async defmain():
  state = MyState(1)
  async with never_42_graph.iter(Increment(), state=state) as graph_run:
    node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
    async for node in graph_run:
      node_states.append((node, deepcopy(graph_run.state)))
    print(node_states)
'''

      (Increment(), MyState(number=1)),
      (Check42(), MyState(number=2)),
      (End(data=2), MyState(number=2)),

    '''
  state = MyState(41)
  async with never_42_graph.iter(Increment(), state=state) as graph_run:
    node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
    async for node in graph_run:
      node_states.append((node, deepcopy(graph_run.state)))
    print(node_states)
'''

      (Increment(), MyState(number=41)),
      (Check42(), MyState(number=42)),
      (Increment(), MyState(number=42)),
      (Check42(), MyState(number=43)),
      (End(data=43), MyState(number=43)),

    '''

```

See the [`GraphRun.next` documentation](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.GraphRun.next) for an example of how to manually drive the graph run.
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
classGraphRun(Generic[StateT, DepsT, RunEndT]):
"""A stateful, async-iterable run of a [`Graph`][pydantic_graph.graph.Graph].
  You typically get a `GraphRun` instance from calling
  `async with [my_graph.iter(...)][pydantic_graph.graph.Graph.iter] as graph_run:`. That gives you the ability to iterate
  through nodes as they run, either by `async for` iteration or by repeatedly calling `.next(...)`.
  Here's an example of iterating over the graph from [above][pydantic_graph.graph.Graph]:
  ```py {title="iter_never_42.py" noqa="I001" py="3.10"}
  from copy import deepcopy
  from never_42 import Increment, MyState, never_42_graph
  async def main():
    state = MyState(1)
    async with never_42_graph.iter(Increment(), state=state) as graph_run:
      node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
      async for node in graph_run:
        node_states.append((node, deepcopy(graph_run.state)))
      print(node_states)
      '''

        (Increment(), MyState(number=1)),
        (Check42(), MyState(number=2)),
        (End(data=2), MyState(number=2)),

      '''
    state = MyState(41)
    async with never_42_graph.iter(Increment(), state=state) as graph_run:
      node_states = [(graph_run.next_node, deepcopy(graph_run.state))]
      async for node in graph_run:
        node_states.append((node, deepcopy(graph_run.state)))
      print(node_states)
      '''

        (Increment(), MyState(number=41)),
        (Check42(), MyState(number=42)),
        (Increment(), MyState(number=42)),
        (Check42(), MyState(number=43)),
        (End(data=43), MyState(number=43)),

      '''
  ```
  See the [`GraphRun.next` documentation][pydantic_graph.graph.GraphRun.next] for an example of how to manually
  drive the graph run.
  """
  def__init__(
    self,
    *,
    graph: Graph[StateT, DepsT, RunEndT],
    start_node: BaseNode[StateT, DepsT, RunEndT],
    persistence: BaseStatePersistence[StateT, RunEndT],
    state: StateT,
    deps: DepsT,
    snapshot_id: str | None = None,
  ):
"""Create a new run for a given graph, starting at the specified node.
    Typically, you'll use [`Graph.iter`][pydantic_graph.graph.Graph.iter] rather than calling this directly.
    Args:
      graph: The [`Graph`][pydantic_graph.graph.Graph] to run.
      start_node: The node where execution will begin.
      persistence: State persistence interface.
      state: A shared state object or primitive (like a counter, dataclass, etc.) that is available
        to all nodes via `ctx.state`.
      deps: Optional dependencies that each node can access via `ctx.deps`, e.g. database connections,
        configuration, or logging clients.
      snapshot_id: The ID of the snapshot the node came from.
    """
    self.graph = graph
    self.persistence = persistence
    self._snapshot_id: str | None = snapshot_id
    self.state = state
    self.deps = deps
    self._next_node: BaseNode[StateT, DepsT, RunEndT] | End[RunEndT] = start_node
  @property
  defnext_node(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
"""The next node that will be run in the graph.
    This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
    """
    return self._next_node
  @property
  defresult(self) -> GraphRunResult[StateT, RunEndT] | None:
"""The final result of the graph run if the run is completed, otherwise `None`."""
    if not isinstance(self._next_node, End):
      return None # The GraphRun has not finished running
    return GraphRunResult(
      self._next_node.data,
      state=self.state,
      persistence=self.persistence,
    )
  async defnext(
    self, node: BaseNode[StateT, DepsT, RunEndT] | None = None
  ) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
"""Manually drive the graph run by passing in the node you want to run next.
    This lets you inspect or mutate the node before continuing execution, or skip certain nodes
    under dynamic conditions. The graph run should stop when you return an [`End`][pydantic_graph.nodes.End] node.
    Here's an example of using `next` to drive the graph from [above][pydantic_graph.graph.Graph]:
    ```py {title="next_never_42.py" noqa="I001" py="3.10"}
    from copy import deepcopy
    from pydantic_graph import End
    from never_42 import Increment, MyState, never_42_graph
    async def main():
      state = MyState(48)
      async with never_42_graph.iter(Increment(), state=state) as graph_run:
        next_node = graph_run.next_node # start with the first node
        node_states = [(next_node, deepcopy(graph_run.state))]
        while not isinstance(next_node, End):
          if graph_run.state.number == 50:
            graph_run.state.number = 42
          next_node = await graph_run.next(next_node)
          node_states.append((next_node, deepcopy(graph_run.state)))
        print(node_states)
        '''

          (Increment(), MyState(number=48)),
          (Check42(), MyState(number=49)),
          (End(data=49), MyState(number=49)),

        '''
    ```
    Args:
      node: The node to run next in the graph. If not specified, uses `self.next_node`, which is initialized to
        the `start_node` of the run and updated each time a new node is returned.
    Returns:
      The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
      the run has completed.
    """
    if node is None:
      # This cast is necessary because self._next_node could be an `End`. You'll get a runtime error if that's
      # the case, but if it is, the only way to get there would be to have tried calling next manually after
      # the run finished. Either way, maybe it would be better to not do this cast...
      node = cast(BaseNode[StateT, DepsT, RunEndT], self._next_node)
      node_snapshot_id = node.get_snapshot_id()
    else:
      node_snapshot_id = node.get_snapshot_id()
    if node_snapshot_id != self._snapshot_id:
      await self.persistence.snapshot_node_if_new(node_snapshot_id, self.state, node)
      self._snapshot_id = node_snapshot_id
    if not isinstance(node, BaseNode):
      # While technically this is not compatible with the documented method signature, it's an easy mistake to
      # make, and we should eagerly provide a more helpful error message than you'd get otherwise.
      raise TypeError(f'`next` must be called with a `BaseNode` instance, got {node!r}.')
    node_id = node.get_node_id()
    if node_id not in self.graph.node_defs:
      raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')
    with ExitStack() as stack:
      if self.graph.auto_instrument:
        stack.enter_context(_logfire.span('run node {node_id}', node_id=node_id, node=node))
      async with self.persistence.record_run(node_snapshot_id):
        ctx = GraphRunContext(self.state, self.deps)
        self._next_node = await node.run(ctx)
    if isinstance(self._next_node, End):
      self._snapshot_id = self._next_node.get_snapshot_id()
      await self.persistence.snapshot_end(self.state, self._next_node)
    elif isinstance(self._next_node, BaseNode):
      self._snapshot_id = self._next_node.get_snapshot_id()
      await self.persistence.snapshot_node(self.state, self._next_node)
    else:
      raise exceptions.GraphRuntimeError(
        f'Invalid node return type: `{type(self._next_node).__name__}`. Expected `BaseNode` or `End`.'
      )
    return self._next_node
  def__aiter__(self) -> AsyncIterator[BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]]:
    return self
  async def__anext__(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
"""Use the last returned node as the input to `Graph.next`."""
    if isinstance(self._next_node, End):
      raise StopAsyncIteration
    return await self.next(self._next_node)
  def__repr__(self) -> str:
    return f'<GraphRun graph={self.graph.nameor"[unnamed]"}>'

```
  
---|---  
####  __init__
```
__init__(
  *,
  graph: [, , ],
  start_node: [, , ],
  persistence: [, ],
  state: ,
  deps: ,
  snapshot_id:  | None = None
)

```

Create a new run for a given graph, starting at the specified node.
Typically, you'll use [`Graph.iter`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.iter) rather than calling this directly.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`graph` |  `Graph[](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph "pydantic_graph.graph.Graph")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  The [`Graph`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph) to run. |  _required_  
`start_node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  The node where execution will begin. |  _required_  
`persistence` |  `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  State persistence interface. |  _required_  
`state` |  |  A shared state object or primitive (like a counter, dataclass, etc.) that is available to all nodes via `ctx.state`. |  _required_  
`deps` |  |  Optional dependencies that each node can access via `ctx.deps`, e.g. database connections, configuration, or logging clients. |  _required_  
`snapshot_id` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The ID of the snapshot the node came from. |  `None`  
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
def__init__(
  self,
  *,
  graph: Graph[StateT, DepsT, RunEndT],
  start_node: BaseNode[StateT, DepsT, RunEndT],
  persistence: BaseStatePersistence[StateT, RunEndT],
  state: StateT,
  deps: DepsT,
  snapshot_id: str | None = None,
):
"""Create a new run for a given graph, starting at the specified node.
  Typically, you'll use [`Graph.iter`][pydantic_graph.graph.Graph.iter] rather than calling this directly.
  Args:
    graph: The [`Graph`][pydantic_graph.graph.Graph] to run.
    start_node: The node where execution will begin.
    persistence: State persistence interface.
    state: A shared state object or primitive (like a counter, dataclass, etc.) that is available
      to all nodes via `ctx.state`.
    deps: Optional dependencies that each node can access via `ctx.deps`, e.g. database connections,
      configuration, or logging clients.
    snapshot_id: The ID of the snapshot the node came from.
  """
  self.graph = graph
  self.persistence = persistence
  self._snapshot_id: str | None = snapshot_id
  self.state = state
  self.deps = deps
  self._next_node: BaseNode[StateT, DepsT, RunEndT] | End[RunEndT] = start_node

```
  
---|---  
####  next_node `property`
```
next_node: [, , ] | []

```

The next node that will be run in the graph.
This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
####  result `property`
```
result: [, ] | None

```

The final result of the graph run if the run is completed, otherwise `None`.
####  next `async`
```
next(
  node: [, , ] | None = None,
) -> [, , ] | []

```

Manually drive the graph run by passing in the node you want to run next.
This lets you inspect or mutate the node before continuing execution, or skip certain nodes under dynamic conditions. The graph run should stop when you return an [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) node.
Here's an example of using `next` to drive the graph from [above](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph): 
next_never_42.py```
fromcopyimport deepcopy
frompydantic_graphimport End
fromnever_42import Increment, MyState, never_42_graph
async defmain():
  state = MyState(48)
  async with never_42_graph.iter(Increment(), state=state) as graph_run:
    next_node = graph_run.next_node # start with the first node
    node_states = [(next_node, deepcopy(graph_run.state))]
    while not isinstance(next_node, End):
      if graph_run.state.number == 50:
        graph_run.state.number = 42
      next_node = await graph_run.next(next_node)
      node_states.append((next_node, deepcopy(graph_run.state)))
    print(node_states)
'''

      (Increment(), MyState(number=48)),
      (Check42(), MyState(number=49)),
      (End(data=49), MyState(number=49)),

    '''

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")] | None` |  The node to run next in the graph. If not specified, uses `self.next_node`, which is initialized to the `start_node` of the run and updated each time a new node is returned. |  `None`  
Returns:
Type | Description  
---|---  
`BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")] | End[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End "pydantic_graph.nodes.End")[RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  The next node returned by the graph logic, or an [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) node if  
`BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")] | End[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End "pydantic_graph.nodes.End")[RunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT "pydantic_graph.nodes.RunEndT")]` |  the run has completed.  
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
async defnext(
  self, node: BaseNode[StateT, DepsT, RunEndT] | None = None
) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
"""Manually drive the graph run by passing in the node you want to run next.
  This lets you inspect or mutate the node before continuing execution, or skip certain nodes
  under dynamic conditions. The graph run should stop when you return an [`End`][pydantic_graph.nodes.End] node.
  Here's an example of using `next` to drive the graph from [above][pydantic_graph.graph.Graph]:
  ```py {title="next_never_42.py" noqa="I001" py="3.10"}
  from copy import deepcopy
  from pydantic_graph import End
  from never_42 import Increment, MyState, never_42_graph
  async def main():
    state = MyState(48)
    async with never_42_graph.iter(Increment(), state=state) as graph_run:
      next_node = graph_run.next_node # start with the first node
      node_states = [(next_node, deepcopy(graph_run.state))]
      while not isinstance(next_node, End):
        if graph_run.state.number == 50:
          graph_run.state.number = 42
        next_node = await graph_run.next(next_node)
        node_states.append((next_node, deepcopy(graph_run.state)))
      print(node_states)
      '''

        (Increment(), MyState(number=48)),
        (Check42(), MyState(number=49)),
        (End(data=49), MyState(number=49)),

      '''
  ```
  Args:
    node: The node to run next in the graph. If not specified, uses `self.next_node`, which is initialized to
      the `start_node` of the run and updated each time a new node is returned.
  Returns:
    The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
    the run has completed.
  """
  if node is None:
    # This cast is necessary because self._next_node could be an `End`. You'll get a runtime error if that's
    # the case, but if it is, the only way to get there would be to have tried calling next manually after
    # the run finished. Either way, maybe it would be better to not do this cast...
    node = cast(BaseNode[StateT, DepsT, RunEndT], self._next_node)
    node_snapshot_id = node.get_snapshot_id()
  else:
    node_snapshot_id = node.get_snapshot_id()
  if node_snapshot_id != self._snapshot_id:
    await self.persistence.snapshot_node_if_new(node_snapshot_id, self.state, node)
    self._snapshot_id = node_snapshot_id
  if not isinstance(node, BaseNode):
    # While technically this is not compatible with the documented method signature, it's an easy mistake to
    # make, and we should eagerly provide a more helpful error message than you'd get otherwise.
    raise TypeError(f'`next` must be called with a `BaseNode` instance, got {node!r}.')
  node_id = node.get_node_id()
  if node_id not in self.graph.node_defs:
    raise exceptions.GraphRuntimeError(f'Node `{node}` is not in the graph.')
  with ExitStack() as stack:
    if self.graph.auto_instrument:
      stack.enter_context(_logfire.span('run node {node_id}', node_id=node_id, node=node))
    async with self.persistence.record_run(node_snapshot_id):
      ctx = GraphRunContext(self.state, self.deps)
      self._next_node = await node.run(ctx)
  if isinstance(self._next_node, End):
    self._snapshot_id = self._next_node.get_snapshot_id()
    await self.persistence.snapshot_end(self.state, self._next_node)
  elif isinstance(self._next_node, BaseNode):
    self._snapshot_id = self._next_node.get_snapshot_id()
    await self.persistence.snapshot_node(self.state, self._next_node)
  else:
    raise exceptions.GraphRuntimeError(
      f'Invalid node return type: `{type(self._next_node).__name__}`. Expected `BaseNode` or `End`.'
    )
  return self._next_node

```
  
---|---  
####  __anext__ `async`
```
__anext__() -> (
  [, , ] | []
)

```

Use the last returned node as the input to `Graph.next`.
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
async def__anext__(self) -> BaseNode[StateT, DepsT, RunEndT] | End[RunEndT]:
"""Use the last returned node as the input to `Graph.next`."""
  if isinstance(self._next_node, End):
    raise StopAsyncIteration
  return await self.next(self._next_node)

```
  
---|---  
###  GraphRunResult `dataclass`
Bases: 
The final result of running a graph.
Source code in `pydantic_graph/pydantic_graph/graph.py`
```
```
| ```
@dataclass
classGraphRunResult(Generic[StateT, RunEndT]):
"""The final result of running a graph."""
  output: RunEndT
  state: StateT
  persistence: BaseStatePersistence[StateT, RunEndT] = field(repr=False)

```
  
---|---

# `pydantic_graph.mermaid`
###  DEFAULT_HIGHLIGHT_CSS `module-attribute`
```
DEFAULT_HIGHLIGHT_CSS = 'fill:#fdff32'

```

The default CSS to use for highlighting nodes.
###  StateDiagramDirection `module-attribute`
```
StateDiagramDirection = ['TB', 'LR', 'RL', 'BT']

```

Used to specify the direction of the state diagram generated by mermaid.
  * `'TB'`: Top to bottom, this is the default for mermaid charts.
  * `'LR'`: Left to right
  * `'RL'`: Right to left
  * `'BT'`: Bottom to top


###  generate_code
```
generate_code(
  graph: [, , ],
  /,
  *,
  start_node: (
    [] |  | None
  ) = None,
  highlighted_nodes: (
    [] |  | None
  ) = None,
  highlight_css:  = ,
  title:  | None = None,
  edge_labels:  = True,
  notes:  = True,
  direction:  | None,
) -> 

```

Generate [Mermaid state diagram](https://mermaid.js.org/syntax/stateDiagram.html) code for a graph.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`graph` |  |  The graph to generate the image for. |  _required_  
`start_node` |  `Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[NodeIdent[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.NodeIdent "pydantic_graph.mermaid.NodeIdent")] | NodeIdent[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.NodeIdent "pydantic_graph.mermaid.NodeIdent") | None` |  Identifiers of nodes that start the graph. |  `None`  
`highlighted_nodes` |  `Sequence[](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "collections.abc.Sequence")[NodeIdent[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.NodeIdent "pydantic_graph.mermaid.NodeIdent")] | NodeIdent[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.NodeIdent "pydantic_graph.mermaid.NodeIdent") | None` |  Identifiers of nodes to highlight. |  `None`  
`highlight_css` |  CSS to use for highlighting nodes. |   
`title` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The title of the diagram. |  `None`  
`edge_labels` |  Whether to include edge labels in the diagram. |  `True`  
`notes` |  Whether to include notes in the diagram. |  `True`  
`direction` |  `StateDiagramDirection[](https://ai.pydantic.dev/api/pydantic_graph/mermaid/#pydantic_graph.mermaid.StateDiagramDirection "pydantic_graph.mermaid.StateDiagramDirection") | None` |  The direction of flow. |  _required_  
Returns:
Type | Description  
---|---  
The Mermaid code for the graph.  
Source code in `pydantic_graph/pydantic_graph/mermaid.py`
```
```
| ```
defgenerate_code( # noqa: C901
  graph: Graph[Any, Any, Any],
  /,
  *,
  start_node: Sequence[NodeIdent] | NodeIdent | None = None,
  highlighted_nodes: Sequence[NodeIdent] | NodeIdent | None = None,
  highlight_css: str = DEFAULT_HIGHLIGHT_CSS,
  title: str | None = None,
  edge_labels: bool = True,
  notes: bool = True,
  direction: StateDiagramDirection | None,
) -> str:
"""Generate [Mermaid state diagram](https://mermaid.js.org/syntax/stateDiagram.html) code for a graph.
  Args:
    graph: The graph to generate the image for.
    start_node: Identifiers of nodes that start the graph.
    highlighted_nodes: Identifiers of nodes to highlight.
    highlight_css: CSS to use for highlighting nodes.
    title: The title of the diagram.
    edge_labels: Whether to include edge labels in the diagram.
    notes: Whether to include notes in the diagram.
    direction: The direction of flow.

  Returns:
    The Mermaid code for the graph.
  """
  start_node_ids = set(_node_ids(start_node or ()))
  for node_id in start_node_ids:
    if node_id not in graph.node_defs:
      raise LookupError(f'Start node "{node_id}" is not in the graph.')
  lines: list[str] = []
  if title:
    lines = ['---', f'title: {title}', '---']
  lines.append('stateDiagram-v2')
  if direction is not None:
    lines.append(f' direction {direction}')
  for node_id, node_def in graph.node_defs.items():
    # we use round brackets (rounded box) for nodes other than the start and end
    if node_id in start_node_ids:
      lines.append(f' [*] --> {node_id}')
    if node_def.returns_base_node:
      for next_node_id in graph.node_defs:
        lines.append(f' {node_id} --> {next_node_id}')
    else:
      for next_node_id, edge in node_def.next_node_edges.items():
        line = f' {node_id} --> {next_node_id}'
        if edge_labels and edge.label:
          line += f': {edge.label}'
        lines.append(line)
    if end_edge := node_def.end_edge:
      line = f' {node_id} --> [*]'
      if edge_labels and end_edge.label:
        line += f': {end_edge.label}'
      lines.append(line)
    if notes and node_def.note:
      lines.append(f' note right of {node_id}')
      # mermaid doesn't like multiple paragraphs in a note, and shows if so
      clean_docs = re.sub('\n{2,}', '\n', node_def.note)
      lines.append(indent(clean_docs, '  '))
      lines.append(' end note')
  if highlighted_nodes:
    lines.append('')
    lines.append(f'classDef highlighted {highlight_css}')
    for node_id in _node_ids(highlighted_nodes):
      if node_id not in graph.node_defs:
        raise LookupError(f'Highlighted node "{node_id}" is not in the graph.')
      lines.append(f'class {node_id} highlighted')
  return '\n'.join(lines)

```
  
---|---  
###  request_image
```
request_image(
  graph: [, , ],
  /,
  **kwargs: [],
) -> 

```

Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink).
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`graph` |  |  The graph to generate the image for. |  _required_  
`**kwargs` |  |  Additional parameters to configure mermaid chart generation.  
Returns:
Type | Description  
---|---  
|  The image data.  
Source code in `pydantic_graph/pydantic_graph/mermaid.py`
```
```
| ```
defrequest_image(
  graph: Graph[Any, Any, Any],
  /,
  **kwargs: Unpack[MermaidConfig],
) -> bytes:
"""Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink).
  Args:
    graph: The graph to generate the image for.
    **kwargs: Additional parameters to configure mermaid chart generation.
  Returns:
    The image data.
  """
  code = generate_code(
    graph,
    start_node=kwargs.get('start_node'),
    highlighted_nodes=kwargs.get('highlighted_nodes'),
    highlight_css=kwargs.get('highlight_css', DEFAULT_HIGHLIGHT_CSS),
    title=kwargs.get('title'),
    edge_labels=kwargs.get('edge_labels', True),
    notes=kwargs.get('notes', True),
    direction=kwargs.get('direction'),
  )
  code_base64 = base64.b64encode(code.encode()).decode()
  params: dict[str, str | float] = {}
  if kwargs.get('image_type') == 'pdf':
    url = f'https://mermaid.ink/pdf/{code_base64}'
    if kwargs.get('pdf_fit'):
      params['fit'] = ''
    if kwargs.get('pdf_landscape'):
      params['landscape'] = ''
    if pdf_paper := kwargs.get('pdf_paper'):
      params['paper'] = pdf_paper
  elif kwargs.get('image_type') == 'svg':
    url = f'https://mermaid.ink/svg/{code_base64}'
  else:
    url = f'https://mermaid.ink/img/{code_base64}'
    if image_type := kwargs.get('image_type'):
      params['type'] = image_type
  if background_color := kwargs.get('background_color'):
    params['bgColor'] = background_color
  if theme := kwargs.get('theme'):
    params['theme'] = theme
  if width := kwargs.get('width'):
    params['width'] = width
  if height := kwargs.get('height'):
    params['height'] = height
  if scale := kwargs.get('scale'):
    params['scale'] = scale
  httpx_client = kwargs.get('httpx_client') or httpx.Client()
  response = httpx_client.get(url, params=params)
  if not response.is_success:
    raise httpx.HTTPStatusError(
      f'{response.status_code} error generating image:\n{response.text}',
      request=response.request,
      response=response,
    )
  return response.content

```
  
---|---  
###  save_image
```
save_image(
  path:  | ,
  graph: [, , ],
  /,
  **kwargs: [],
) -> None

```

Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink) and save it to a local file.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`path` |  |  The path to save the image to. |  _required_  
`graph` |  |  The graph to generate the image for. |  _required_  
`**kwargs` |  |  Additional parameters to configure mermaid chart generation.  
Source code in `pydantic_graph/pydantic_graph/mermaid.py`
```
```
| ```
defsave_image(
  path: Path | str,
  graph: Graph[Any, Any, Any],
  /,
  **kwargs: Unpack[MermaidConfig],
) -> None:
"""Generate an image of a Mermaid diagram using [mermaid.ink](https://mermaid.ink) and save it to a local file.
  Args:
    path: The path to save the image to.
    graph: The graph to generate the image for.
    **kwargs: Additional parameters to configure mermaid chart generation.
  """
  if isinstance(path, str):
    path = Path(path)
  if 'image_type' not in kwargs:
    ext = path.suffix.lower()[1:]
    # no need to check for .jpeg/.jpg, as it is the default
    if ext in ('png', 'webp', 'svg', 'pdf'):
      kwargs['image_type'] = ext
  image_data = request_image(graph, **kwargs)
  path.write_bytes(image_data)

```
  
---|---  
###  MermaidConfig
Bases: 
Parameters to configure mermaid chart generation.
Source code in `pydantic_graph/pydantic_graph/mermaid.py`
```
```
| ```
classMermaidConfig(TypedDict, total=False):
"""Parameters to configure mermaid chart generation."""
  start_node: Sequence[NodeIdent] | NodeIdent
"""Identifiers of nodes that start the graph."""
  highlighted_nodes: Sequence[NodeIdent] | NodeIdent
"""Identifiers of nodes to highlight."""
  highlight_css: str
"""CSS to use for highlighting nodes."""
  title: str | None
"""The title of the diagram."""
  edge_labels: bool
"""Whether to include edge labels in the diagram."""
  notes: bool
"""Whether to include notes on nodes in the diagram, defaults to true."""
  image_type: Literal['jpeg', 'png', 'webp', 'svg', 'pdf']
"""The image type to generate. If unspecified, the default behavior is `'jpeg'`."""
  pdf_fit: bool
"""When using image_type='pdf', whether to fit the diagram to the PDF page."""
  pdf_landscape: bool
"""When using image_type='pdf', whether to use landscape orientation for the PDF.
  This has no effect if using `pdf_fit`.
  """
  pdf_paper: Literal['letter', 'legal', 'tabloid', 'ledger', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
"""When using image_type='pdf', the paper size of the PDF."""
  background_color: str
"""The background color of the diagram.
  If None, the default transparent background is used. The color value is interpreted as a hexadecimal color
  code by default (and should not have a leading '#'), but you can also use named colors by prefixing the
  value with `'!'`. For example, valid choices include `background_color='!white'` or `background_color='FF0000'`.
  """
  theme: Literal['default', 'neutral', 'dark', 'forest']
"""The theme of the diagram. Defaults to 'default'."""
  width: int
"""The width of the diagram."""
  height: int
"""The height of the diagram."""
  scale: Annotated[float, Ge(1), Le(3)]
"""The scale of the diagram.
  The scale must be a number between 1 and 3, and you can only set a scale if one or both of width and height are set.
  """
  httpx_client: httpx.Client
"""An HTTPX client to use for requests, mostly for testing purposes."""
  direction: StateDiagramDirection
"""The direction of the state diagram."""

```
  
---|---  
####  start_node `instance-attribute`
```
start_node: [] | 

```

Identifiers of nodes that start the graph.
####  highlighted_nodes `instance-attribute`
```
highlighted_nodes: [] | 

```

Identifiers of nodes to highlight.
####  highlight_css `instance-attribute`
```
highlight_css: 

```

CSS to use for highlighting nodes.
####  title `instance-attribute`
```
title:  | None

```

The title of the diagram.
####  edge_labels `instance-attribute`
```
edge_labels: 

```

Whether to include edge labels in the diagram.
####  notes `instance-attribute`
```
notes: 

```

Whether to include notes on nodes in the diagram, defaults to true.
####  image_type `instance-attribute`
```
image_type: ['jpeg', 'png', 'webp', 'svg', 'pdf']

```

The image type to generate. If unspecified, the default behavior is `'jpeg'`.
####  pdf_fit `instance-attribute`
```
pdf_fit: 

```

When using image_type='pdf', whether to fit the diagram to the PDF page.
####  pdf_landscape `instance-attribute`
```
pdf_landscape: 

```

When using image_type='pdf', whether to use landscape orientation for the PDF.
This has no effect if using `pdf_fit`.
####  pdf_paper `instance-attribute`
```
pdf_paper: [
  "letter",
  "legal",
  "tabloid",
  "ledger",
  "a0",
  "a1",
  "a2",
  "a3",
  "a4",
  "a5",
  "a6",
]

```

When using image_type='pdf', the paper size of the PDF.
####  background_color `instance-attribute`
```
background_color: 

```

The background color of the diagram.
If None, the default transparent background is used. The color value is interpreted as a hexadecimal color code by default (and should not have a leading '#'), but you can also use named colors by prefixing the value with `'!'`. For example, valid choices include `background_color='!white'` or `background_color='FF0000'`.
####  theme `instance-attribute`
```
theme: ['default', 'neutral', 'dark', 'forest']

```

The theme of the diagram. Defaults to 'default'.
####  width `instance-attribute`
```
width: 

```

The width of the diagram.
####  height `instance-attribute`
```
height: 

```

The height of the diagram.
####  scale `instance-attribute`
```
scale: [, (1), (3)]

```

The scale of the diagram.
The scale must be a number between 1 and 3, and you can only set a scale if one or both of width and height are set.
####  httpx_client `instance-attribute`
```
httpx_client: Client

```

An HTTPX client to use for requests, mostly for testing purposes.
####  direction `instance-attribute`
```
direction: 

```

The direction of the state diagram.
###  NodeIdent `module-attribute`
```
NodeIdent:  = (
  "type[BaseNode[Any, Any, Any]] | BaseNode[Any, Any, Any] | str"
)

```

A type alias for a node identifier.
This can be:
  * A node instance (instance of a subclass of [`BaseNode`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode)).
  * A node class (subclass of [`BaseNode`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode)).
  * A string representing the node ID.



# `pydantic_graph.nodes`
###  StateT `module-attribute`
```
StateT = TypeVar('StateT', default=None)

```

Type variable for the state in a graph.
###  GraphRunContext `dataclass`
Bases: 
Context for a graph.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
@dataclass
classGraphRunContext(Generic[StateT, DepsT]):
"""Context for a graph."""
  # TODO: Can we get rid of this struct and just pass both these things around..?
  state: StateT
"""The state of the graph."""
  deps: DepsT
"""Dependencies for the graph."""

```
  
---|---  
####  state `instance-attribute`
```
state: 

```

The state of the graph.
####  deps `instance-attribute`
```
deps: 

```

Dependencies for the graph.
###  BaseNode
Bases: , `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), NodeRunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.NodeRunEndT "pydantic_graph.nodes.NodeRunEndT")]`
Base class for a node.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
classBaseNode(ABC, Generic[StateT, DepsT, NodeRunEndT]):
"""Base class for a node."""
  docstring_notes: ClassVar[bool] = False
"""Set to `True` to generate mermaid diagram notes from the class's docstring.
  While this can add valuable information to the diagram, it can make diagrams harder to view, hence
  it is disabled by default. You can also customise notes overriding the
  [`get_note`][pydantic_graph.nodes.BaseNode.get_note] method.
  """
  @abstractmethod
  async defrun(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[NodeRunEndT]:
"""Run the node.
    This is an abstract method that must be implemented by subclasses.
    !!! note "Return types used at runtime"
      The return type of this method are read by `pydantic_graph` at runtime and used to define which
      nodes can be called next in the graph. This is displayed in [mermaid diagrams](mermaid.md)
      and enforced when running the graph.
    Args:
      ctx: The graph context.
    Returns:
      The next node to run or [`End`][pydantic_graph.nodes.End] to signal the end of the graph.
    """
    ...
  defget_snapshot_id(self) -> str:
    if snapshot_id := getattr(self, '__snapshot_id', None):
      return snapshot_id
    else:
      self.__dict__['__snapshot_id'] = snapshot_id = generate_snapshot_id(self.get_node_id())
      return snapshot_id
  defset_snapshot_id(self, snapshot_id: str) -> None:
    self.__dict__['__snapshot_id'] = snapshot_id
  @classmethod
  @cache
  defget_node_id(cls) -> str:
"""Get the ID of the node."""
    return cls.__name__
  @classmethod
  defget_note(cls) -> str | None:
"""Get a note about the node to render on mermaid charts.
    By default, this returns a note only if [`docstring_notes`][pydantic_graph.nodes.BaseNode.docstring_notes]
    is `True`. You can override this method to customise the node notes.
    """
    if not cls.docstring_notes:
      return None
    docstring = cls.__doc__
    # dataclasses get an automatic docstring which is just their signature, we don't want that
    if docstring and is_dataclass(cls) and docstring.startswith(f'{cls.__name__}('):
      docstring = None
    if docstring:
      # remove indentation from docstring
      importinspect
      docstring = inspect.cleandoc(docstring)
    return docstring
  @classmethod
  defget_node_def(cls, local_ns: dict[str, Any] | None) -> NodeDef[StateT, DepsT, NodeRunEndT]:
"""Get the node definition."""
    type_hints = get_type_hints(cls.run, localns=local_ns, include_extras=True)
    try:
      return_hint = type_hints['return']
    except KeyError as e:
      raise exceptions.GraphSetupError(f'Node {cls} is missing a return type hint on its `run` method') frome
    next_node_edges: dict[str, Edge] = {}
    end_edge: Edge | None = None
    returns_base_node: bool = False
    for return_type in _utils.get_union_args(return_hint):
      return_type, annotations = _utils.unpack_annotated(return_type)
      edge = next((a for a in annotations if isinstance(a, Edge)), Edge(None))
      return_type_origin = get_origin(return_type) or return_type
      if return_type_origin is End:
        end_edge = edge
      elif return_type_origin is BaseNode:
        # TODO: Should we disallow this?
        returns_base_node = True
      elif issubclass(return_type_origin, BaseNode):
        next_node_edges[return_type.get_node_id()] = edge
      else:
        raise exceptions.GraphSetupError(f'Invalid return type: {return_type}')
    return NodeDef(
      cls,
      cls.get_node_id(),
      cls.get_note(),
      next_node_edges,
      end_edge,
      returns_base_node,
    )
  defdeep_copy(self) -> Self:
"""Returns a deep copy of the node."""
    return copy.deepcopy(self)

```
  
---|---  
####  docstring_notes `class-attribute`
```
docstring_notes:  = False

```

Set to `True` to generate mermaid diagram notes from the class's docstring.
While this can add valuable information to the diagram, it can make diagrams harder to view, hence it is disabled by default. You can also customise notes overriding the [`get_note`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.get_note) method.
####  run `abstractmethod` `async`
```
run(
  ctx: [, ],
) -> [, , ] | []

```

Run the node.
This is an abstract method that must be implemented by subclasses.
Return types used at runtime
The return type of this method are read by `pydantic_graph` at runtime and used to define which nodes can be called next in the graph. This is displayed in [mermaid diagrams](https://ai.pydantic.dev/api/pydantic_graph/mermaid/) and enforced when running the graph.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`ctx` |  `GraphRunContext[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.GraphRunContext "pydantic_graph.nodes.GraphRunContext")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT")]` |  The graph context. |  _required_  
Returns:
Type | Description  
---|---  
`BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT "pydantic_graph.nodes.StateT"), DepsT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT "pydantic_graph.nodes.DepsT"), Any[](https://docs.python.org/3/library/typing.html#typing.Any "typing.Any")] | End[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End "pydantic_graph.nodes.End")[NodeRunEndT[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.NodeRunEndT "pydantic_graph.nodes.NodeRunEndT")]` |  The next node to run or [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) to signal the end of the graph.  
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
@abstractmethod
async defrun(self, ctx: GraphRunContext[StateT, DepsT]) -> BaseNode[StateT, DepsT, Any] | End[NodeRunEndT]:
"""Run the node.
  This is an abstract method that must be implemented by subclasses.
  !!! note "Return types used at runtime"
    The return type of this method are read by `pydantic_graph` at runtime and used to define which
    nodes can be called next in the graph. This is displayed in [mermaid diagrams](mermaid.md)
    and enforced when running the graph.
  Args:
    ctx: The graph context.
  Returns:
    The next node to run or [`End`][pydantic_graph.nodes.End] to signal the end of the graph.
  """
  ...

```
  
---|---  
####  get_node_id `cached` `classmethod`
```
get_node_id() -> 

```

Get the ID of the node.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
@classmethod
@cache
defget_node_id(cls) -> str:
"""Get the ID of the node."""
  return cls.__name__

```
  
---|---  
####  get_note `classmethod`
```
get_note() ->  | None

```

Get a note about the node to render on mermaid charts.
By default, this returns a note only if [`docstring_notes`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.docstring_notes) is `True`. You can override this method to customise the node notes.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
@classmethod
defget_note(cls) -> str | None:
"""Get a note about the node to render on mermaid charts.
  By default, this returns a note only if [`docstring_notes`][pydantic_graph.nodes.BaseNode.docstring_notes]
  is `True`. You can override this method to customise the node notes.
  """
  if not cls.docstring_notes:
    return None
  docstring = cls.__doc__
  # dataclasses get an automatic docstring which is just their signature, we don't want that
  if docstring and is_dataclass(cls) and docstring.startswith(f'{cls.__name__}('):
    docstring = None
  if docstring:
    # remove indentation from docstring
    importinspect
    docstring = inspect.cleandoc(docstring)
  return docstring

```
  
---|---  
####  get_node_def `classmethod`
```
get_node_def(
  local_ns: [, ] | None,
) -> NodeDef[, , ]

```

Get the node definition.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
@classmethod
defget_node_def(cls, local_ns: dict[str, Any] | None) -> NodeDef[StateT, DepsT, NodeRunEndT]:
"""Get the node definition."""
  type_hints = get_type_hints(cls.run, localns=local_ns, include_extras=True)
  try:
    return_hint = type_hints['return']
  except KeyError as e:
    raise exceptions.GraphSetupError(f'Node {cls} is missing a return type hint on its `run` method') frome
  next_node_edges: dict[str, Edge] = {}
  end_edge: Edge | None = None
  returns_base_node: bool = False
  for return_type in _utils.get_union_args(return_hint):
    return_type, annotations = _utils.unpack_annotated(return_type)
    edge = next((a for a in annotations if isinstance(a, Edge)), Edge(None))
    return_type_origin = get_origin(return_type) or return_type
    if return_type_origin is End:
      end_edge = edge
    elif return_type_origin is BaseNode:
      # TODO: Should we disallow this?
      returns_base_node = True
    elif issubclass(return_type_origin, BaseNode):
      next_node_edges[return_type.get_node_id()] = edge
    else:
      raise exceptions.GraphSetupError(f'Invalid return type: {return_type}')
  return NodeDef(
    cls,
    cls.get_node_id(),
    cls.get_note(),
    next_node_edges,
    end_edge,
    returns_base_node,
  )

```
  
---|---  
####  deep_copy
```
deep_copy() -> 

```

Returns a deep copy of the node.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
defdeep_copy(self) -> Self:
"""Returns a deep copy of the node."""
  return copy.deepcopy(self)

```
  
---|---  
###  End `dataclass`
Bases: 
Type to return from a node to signal the end of the graph.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
@dataclass
classEnd(Generic[RunEndT]):
"""Type to return from a node to signal the end of the graph."""
  data: RunEndT
"""Data to return from the graph."""
  defdeep_copy_data(self) -> End[RunEndT]:
"""Returns a deep copy of the end of the run."""
    if self.data is None:
      return self
    else:
      end = End(copy.deepcopy(self.data))
      end.set_snapshot_id(self.get_snapshot_id())
      return end
  defget_snapshot_id(self) -> str:
    if snapshot_id := getattr(self, '__snapshot_id', None):
      return snapshot_id
    else:
      self.__dict__['__snapshot_id'] = snapshot_id = generate_snapshot_id('end')
      return snapshot_id
  defset_snapshot_id(self, set_id: str) -> None:
    self.__dict__['__snapshot_id'] = set_id

```
  
---|---  
####  data `instance-attribute`
```
data: 

```

Data to return from the graph.
####  deep_copy_data
```
deep_copy_data() -> []

```

Returns a deep copy of the end of the run.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
defdeep_copy_data(self) -> End[RunEndT]:
"""Returns a deep copy of the end of the run."""
  if self.data is None:
    return self
  else:
    end = End(copy.deepcopy(self.data))
    end.set_snapshot_id(self.get_snapshot_id())
    return end

```
  
---|---  
###  Edge `dataclass`
Annotation to apply a label to an edge in a graph.
Source code in `pydantic_graph/pydantic_graph/nodes.py`
```
```
| ```
@dataclass
classEdge:
"""Annotation to apply a label to an edge in a graph."""
  label: str | None
"""Label for the edge."""

```
  
---|---  
####  label `instance-attribute`
```
label:  | None

```

Label for the edge.
###  DepsT `module-attribute`
```
DepsT = TypeVar('DepsT', default=None, contravariant=True)

```

Type variable for the dependencies of a graph and node.
###  RunEndT `module-attribute`
```
RunEndT = TypeVar('RunEndT', covariant=True, default=None)

```

Covariant type variable for the return type of a graph [`run`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.run).
###  NodeRunEndT `module-attribute`
```
NodeRunEndT = TypeVar(
  "NodeRunEndT", covariant=True, default=
)

```

Covariant type variable for the return type of a node [`run`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.run).

# `pydantic_graph.persistence`
###  SnapshotStatus `module-attribute`
```
SnapshotStatus = [
  "created", "pending", "running", "success", "error"
]

```

The status of a snapshot.
  * `'created'`: The snapshot has been created but not yet run.
  * `'pending'`: The snapshot has been retrieved with [`load_next`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence.load_next) but not yet run.
  * `'running'`: The snapshot is currently running.
  * `'success'`: The snapshot has been run successfully.
  * `'error'`: The snapshot has been run but an error occurred.


###  NodeSnapshot `dataclass`
Bases: `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[StateT, RunEndT]`
History step describing the execution of a node in a graph.
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
@dataclass
classNodeSnapshot(Generic[StateT, RunEndT]):
"""History step describing the execution of a node in a graph."""
  state: StateT
"""The state of the graph before the node is run."""
  node: Annotated[BaseNode[StateT, Any, RunEndT], _utils.CustomNodeSchema()]
"""The node to run next."""
  start_ts: datetime | None = None
"""The timestamp when the node started running, `None` until the run starts."""
  duration: float | None = None
"""The duration of the node run in seconds, if the node has been run."""
  status: SnapshotStatus = 'created'
"""The status of the snapshot."""
  kind: Literal['node'] = 'node'
"""The kind of history step, can be used as a discriminator when deserializing history."""
  id: str = UNSET_SNAPSHOT_ID
"""Unique ID of the snapshot."""
  def__post_init__(self) -> None:
    if self.id == UNSET_SNAPSHOT_ID:
      self.id = self.node.get_snapshot_id()

```
  
---|---  
####  state `instance-attribute`
```
state: StateT

```

The state of the graph before the node is run.
####  node `instance-attribute`
```
node: [
  [StateT, , RunEndT], CustomNodeSchema()
]

```

The node to run next.
####  start_ts `class-attribute` `instance-attribute`
```
start_ts:  | None = None

```

The timestamp when the node started running, `None` until the run starts.
####  duration `class-attribute` `instance-attribute`
```
duration:  | None = None

```

The duration of the node run in seconds, if the node has been run.
####  status `class-attribute` `instance-attribute`
```
status:  = 'created'

```

The status of the snapshot.
####  kind `class-attribute` `instance-attribute`
```
kind: ['node'] = 'node'

```

The kind of history step, can be used as a discriminator when deserializing history.
####  id `class-attribute` `instance-attribute`
```
id:  = UNSET_SNAPSHOT_ID

```

Unique ID of the snapshot.
###  EndSnapshot `dataclass`
Bases: `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[StateT, RunEndT]`
History step describing the end of a graph run.
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
@dataclass
classEndSnapshot(Generic[StateT, RunEndT]):
"""History step describing the end of a graph run."""
  state: StateT
"""The state of the graph at the end of the run."""
  result: End[RunEndT]
"""The result of the graph run."""
  ts: datetime = field(default_factory=_utils.now_utc)
"""The timestamp when the graph run ended."""
  kind: Literal['end'] = 'end'
"""The kind of history step, can be used as a discriminator when deserializing history."""
  id: str = UNSET_SNAPSHOT_ID
"""Unique ID of the snapshot."""
  def__post_init__(self) -> None:
    if self.id == UNSET_SNAPSHOT_ID:
      self.id = self.node.get_snapshot_id()
  @property
  defnode(self) -> End[RunEndT]:
"""Shim to get the [`result`][pydantic_graph.persistence.EndSnapshot.result].
    Useful to allow `[snapshot.node for snapshot in persistence.history]`.
    """
    return self.result

```
  
---|---  
####  state `instance-attribute`
```
state: StateT

```

The state of the graph at the end of the run.
####  result `instance-attribute`
```
result: [RunEndT]

```

The result of the graph run.
####  ts `class-attribute` `instance-attribute`
```
ts:  = (default_factory=now_utc)

```

The timestamp when the graph run ended.
####  kind `class-attribute` `instance-attribute`
```
kind: ['end'] = 'end'

```

The kind of history step, can be used as a discriminator when deserializing history.
####  id `class-attribute` `instance-attribute`
```
id:  = UNSET_SNAPSHOT_ID

```

Unique ID of the snapshot.
####  node `property`
```
node: [RunEndT]

```

Shim to get the [`result`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.EndSnapshot.result).
Useful to allow `[snapshot.node for snapshot in persistence.history]`.
###  Snapshot `module-attribute`
```
Snapshot = [
  [StateT, RunEndT],
  [StateT, RunEndT],
]

```

A step in the history of a graph run.
[`Graph.run`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.run) returns a list of these steps describing the execution of the graph, together with the run return value.
###  BaseStatePersistence
Bases: , `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[StateT, RunEndT]`
Abstract base class for storing the state of a graph run.
Each instance of a `BaseStatePersistence` subclass should be used for a single graph run.
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
classBaseStatePersistence(ABC, Generic[StateT, RunEndT]):
"""Abstract base class for storing the state of a graph run.
  Each instance of a `BaseStatePersistence` subclass should be used for a single graph run.
  """
  @abstractmethod
  async defsnapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
"""Snapshot the state of a graph, when the next step is to run a node.
    This method should add a [`NodeSnapshot`][pydantic_graph.persistence.NodeSnapshot] to persistence.
    Args:
      state: The state of the graph.
      next_node: The next node to run.
    """
    raise NotImplementedError
  @abstractmethod
  async defsnapshot_node_if_new(
    self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
  ) -> None:
"""Snapshot the state of a graph if the snapshot ID doesn't already exist in persistence.
    This method will generally call [`snapshot_node`][pydantic_graph.persistence.BaseStatePersistence.snapshot_node]
    but should do so in an atomic way.
    Args:
      snapshot_id: The ID of the snapshot to check.
      state: The state of the graph.
      next_node: The next node to run.
    """
    raise NotImplementedError
  @abstractmethod
  async defsnapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
"""Snapshot the state of a graph when the graph has ended.
    This method should add an [`EndSnapshot`][pydantic_graph.persistence.EndSnapshot] to persistence.
    Args:
      state: The state of the graph.
      end: data from the end of the run.
    """
    raise NotImplementedError
  @abstractmethod
  defrecord_run(self, snapshot_id: str) -> AbstractAsyncContextManager[None]:
"""Record the run of the node, or error if the node is already running.
    Args:
      snapshot_id: The ID of the snapshot to record.
    Raises:
      GraphNodeRunningError: if the node status it not `'created'` or `'pending'`.
      LookupError: if the snapshot ID is not found in persistence.
    Returns:
      An async context manager that records the run of the node.
    In particular this should set:
    - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'running'` and
     [`NodeSnapshot.start_ts`][pydantic_graph.persistence.NodeSnapshot.start_ts] when the run starts.
    - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'success'` or `'error'` and
     [`NodeSnapshot.duration`][pydantic_graph.persistence.NodeSnapshot.duration] when the run finishes.
    """
    raise NotImplementedError
  @abstractmethod
  async defload_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
"""Retrieve a node snapshot with status `'created`' and set its status to `'pending'`.
    This is used by [`Graph.iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence]
    to get the next node to run.
    Returns: The snapshot, or `None` if no snapshot with status `'created`' exists.
    """
    raise NotImplementedError
  @abstractmethod
  async defload_all(self) -> list[Snapshot[StateT, RunEndT]]:
"""Load the entire history of snapshots.
    `load_all` is not used by pydantic-graph itself, instead it's provided to make it convenient to
    get all [snapshots][pydantic_graph.persistence.Snapshot] from persistence.
    Returns: The list of snapshots.
    """
    raise NotImplementedError
  defset_graph_types(self, graph: Graph[StateT, Any, RunEndT]) -> None:
"""Set the types of the state and run end from a graph.
    You generally won't need to customise this method, instead implement
    [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types] and
    [`should_set_types`][pydantic_graph.persistence.BaseStatePersistence.should_set_types].
    """
    if self.should_set_types():
      with _utils.set_nodes_type_context(graph.get_nodes()):
        self.set_types(*graph.inferred_types)
  defshould_set_types(self) -> bool:
"""Whether types need to be set.
    Implementations should override this method to return `True` when types have not been set if they are needed.
    """
    return False
  defset_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
"""Set the types of the state and run end.
    This can be used to create [type adapters][pydantic.TypeAdapter] for serializing and deserializing snapshots,
    e.g. with [`build_snapshot_list_type_adapter`][pydantic_graph.persistence.build_snapshot_list_type_adapter].
    Args:
      state_type: The state type.
      run_end_type: The run end type.
    """
    pass

```
  
---|---  
####  snapshot_node `abstractmethod` `async`
```
snapshot_node(
  state: StateT, next_node: [StateT, , RunEndT]
) -> None

```

Snapshot the state of a graph, when the next step is to run a node.
This method should add a [`NodeSnapshot`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.NodeSnapshot) to persistence.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`state` |  `StateT` |  The state of the graph. |  _required_  
`next_node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT, Any[](https://docs.python.org/3/library/typing.html#typing.Any "typing.Any"), RunEndT]` |  The next node to run. |  _required_  
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
@abstractmethod
async defsnapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
"""Snapshot the state of a graph, when the next step is to run a node.
  This method should add a [`NodeSnapshot`][pydantic_graph.persistence.NodeSnapshot] to persistence.
  Args:
    state: The state of the graph.
    next_node: The next node to run.
  """
  raise NotImplementedError

```
  
---|---  
####  snapshot_node_if_new `abstractmethod` `async`
```
snapshot_node_if_new(
  snapshot_id: ,
  state: StateT,
  next_node: [StateT, , RunEndT],
) -> None

```

Snapshot the state of a graph if the snapshot ID doesn't already exist in persistence.
This method will generally call [`snapshot_node`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence.snapshot_node) but should do so in an atomic way.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`snapshot_id` |  The ID of the snapshot to check. |  _required_  
`state` |  `StateT` |  The state of the graph. |  _required_  
`next_node` |  `BaseNode[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode "pydantic_graph.nodes.BaseNode")[StateT, Any[](https://docs.python.org/3/library/typing.html#typing.Any "typing.Any"), RunEndT]` |  The next node to run. |  _required_  
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
@abstractmethod
async defsnapshot_node_if_new(
  self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
) -> None:
"""Snapshot the state of a graph if the snapshot ID doesn't already exist in persistence.
  This method will generally call [`snapshot_node`][pydantic_graph.persistence.BaseStatePersistence.snapshot_node]
  but should do so in an atomic way.
  Args:
    snapshot_id: The ID of the snapshot to check.
    state: The state of the graph.
    next_node: The next node to run.
  """
  raise NotImplementedError

```
  
---|---  
####  snapshot_end `abstractmethod` `async`
```
snapshot_end(state: StateT, end: [RunEndT]) -> None

```

Snapshot the state of a graph when the graph has ended.
This method should add an [`EndSnapshot`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.EndSnapshot) to persistence.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`state` |  `StateT` |  The state of the graph. |  _required_  
`end` |  `End[](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End "pydantic_graph.nodes.End")[RunEndT]` |  data from the end of the run. |  _required_  
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
@abstractmethod
async defsnapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
"""Snapshot the state of a graph when the graph has ended.
  This method should add an [`EndSnapshot`][pydantic_graph.persistence.EndSnapshot] to persistence.
  Args:
    state: The state of the graph.
    end: data from the end of the run.
  """
  raise NotImplementedError

```
  
---|---  
####  record_run `abstractmethod`
```
record_run(
  snapshot_id: ,
) -> [None]

```

Record the run of the node, or error if the node is already running.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`snapshot_id` |  The ID of the snapshot to record. |  _required_  
Raises:
Type | Description  
---|---  
`GraphNodeRunningError` |  if the node status it not `'created'` or `'pending'`.  
|  if the snapshot ID is not found in persistence.  
Returns:
Type | Description  
---|---  
`AbstractAsyncContextManager[](https://docs.python.org/3/library/contextlib.html#contextlib.AbstractAsyncContextManager "contextlib.AbstractAsyncContextManager")[None]` |  An async context manager that records the run of the node.  
In particular this should set:
  * [`NodeSnapshot.status`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.NodeSnapshot.status) to `'running'` and [`NodeSnapshot.start_ts`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.NodeSnapshot.start_ts) when the run starts.
  * [`NodeSnapshot.status`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.NodeSnapshot.status) to `'success'` or `'error'` and [`NodeSnapshot.duration`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.NodeSnapshot.duration) when the run finishes.

Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
@abstractmethod
defrecord_run(self, snapshot_id: str) -> AbstractAsyncContextManager[None]:
"""Record the run of the node, or error if the node is already running.
  Args:
    snapshot_id: The ID of the snapshot to record.
  Raises:
    GraphNodeRunningError: if the node status it not `'created'` or `'pending'`.
    LookupError: if the snapshot ID is not found in persistence.
  Returns:
    An async context manager that records the run of the node.
  In particular this should set:
  - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'running'` and
   [`NodeSnapshot.start_ts`][pydantic_graph.persistence.NodeSnapshot.start_ts] when the run starts.
  - [`NodeSnapshot.status`][pydantic_graph.persistence.NodeSnapshot.status] to `'success'` or `'error'` and
   [`NodeSnapshot.duration`][pydantic_graph.persistence.NodeSnapshot.duration] when the run finishes.
  """
  raise NotImplementedError

```
  
---|---  
####  load_next `abstractmethod` `async`
```
load_next() -> [StateT, RunEndT] | None

```

Retrieve a node snapshot with status `'created`' and set its status to `'pending'`.
This is used by [`Graph.iter_from_persistence`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.iter_from_persistence) to get the next node to run.
Returns: The snapshot, or `None` if no snapshot with status `'created`' exists.
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
@abstractmethod
async defload_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
"""Retrieve a node snapshot with status `'created`' and set its status to `'pending'`.
  This is used by [`Graph.iter_from_persistence`][pydantic_graph.graph.Graph.iter_from_persistence]
  to get the next node to run.
  Returns: The snapshot, or `None` if no snapshot with status `'created`' exists.
  """
  raise NotImplementedError

```
  
---|---  
####  load_all `abstractmethod` `async`
```
load_all() -> [[StateT, RunEndT]]

```

Load the entire history of snapshots.
`load_all` is not used by pydantic-graph itself, instead it's provided to make it convenient to get all [snapshots](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.Snapshot) from persistence.
Returns: The list of snapshots.
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
@abstractmethod
async defload_all(self) -> list[Snapshot[StateT, RunEndT]]:
"""Load the entire history of snapshots.
  `load_all` is not used by pydantic-graph itself, instead it's provided to make it convenient to
  get all [snapshots][pydantic_graph.persistence.Snapshot] from persistence.
  Returns: The list of snapshots.
  """
  raise NotImplementedError

```
  
---|---  
####  set_graph_types
```
set_graph_types(graph: [StateT, , RunEndT]) -> None

```

Set the types of the state and run end from a graph.
You generally won't need to customise this method, instead implement [`set_types`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence.set_types) and [`should_set_types`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence.should_set_types).
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
defset_graph_types(self, graph: Graph[StateT, Any, RunEndT]) -> None:
"""Set the types of the state and run end from a graph.
  You generally won't need to customise this method, instead implement
  [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types] and
  [`should_set_types`][pydantic_graph.persistence.BaseStatePersistence.should_set_types].
  """
  if self.should_set_types():
    with _utils.set_nodes_type_context(graph.get_nodes()):
      self.set_types(*graph.inferred_types)

```
  
---|---  
####  should_set_types
```
should_set_types() -> 

```

Whether types need to be set.
Implementations should override this method to return `True` when types have not been set if they are needed.
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
defshould_set_types(self) -> bool:
"""Whether types need to be set.
  Implementations should override this method to return `True` when types have not been set if they are needed.
  """
  return False

```
  
---|---  
####  set_types
```
set_types(
  state_type: [StateT], run_end_type: [RunEndT]
) -> None

```

Set the types of the state and run end.
This can be used to create [type adapters](https://docs.pydantic.dev/latest/api/type_adapter/#pydantic.type_adapter.TypeAdapter) for serializing and deserializing snapshots, e.g. with [`build_snapshot_list_type_adapter`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.build_snapshot_list_type_adapter).
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`state_type` |  `type[](https://docs.python.org/3/library/functions.html#type)[StateT]` |  The state type. |  _required_  
`run_end_type` |  `type[](https://docs.python.org/3/library/functions.html#type)[RunEndT]` |  The run end type. |  _required_  
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
defset_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
"""Set the types of the state and run end.
  This can be used to create [type adapters][pydantic.TypeAdapter] for serializing and deserializing snapshots,
  e.g. with [`build_snapshot_list_type_adapter`][pydantic_graph.persistence.build_snapshot_list_type_adapter].
  Args:
    state_type: The state type.
    run_end_type: The run end type.
  """
  pass

```
  
---|---  
###  build_snapshot_list_type_adapter
```
build_snapshot_list_type_adapter(
  state_t: [StateT], run_end_t: [RunEndT]
) -> [[[StateT, RunEndT]]]

```

Build a type adapter for a list of snapshots.
This method should be called from within [`set_types`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence.set_types) where context variables will be set such that Pydantic can create a schema for [`NodeSnapshot.node`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.NodeSnapshot.node).
Source code in `pydantic_graph/pydantic_graph/persistence/__init__.py`
```
```
| ```
defbuild_snapshot_list_type_adapter(
  state_t: type[StateT], run_end_t: type[RunEndT]
) -> pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]]:
"""Build a type adapter for a list of snapshots.
  This method should be called from within
  [`set_types`][pydantic_graph.persistence.BaseStatePersistence.set_types]
  where context variables will be set such that Pydantic can create a schema for
  [`NodeSnapshot.node`][pydantic_graph.persistence.NodeSnapshot.node].
  """
  return pydantic.TypeAdapter(list[Annotated[Snapshot[state_t, run_end_t], pydantic.Discriminator('kind')]])

```
  
---|---  
In memory state persistence.
This module provides simple in memory state persistence for graphs.
###  SimpleStatePersistence `dataclass`
Bases: `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT, RunEndT]`
Simple in memory state persistence that just hold the latest snapshot.
If no state persistence implementation is provided when running a graph, this is used by default.
Source code in `pydantic_graph/pydantic_graph/persistence/in_mem.py`
```
```
| ```
@dataclass
classSimpleStatePersistence(BaseStatePersistence[StateT, RunEndT]):
"""Simple in memory state persistence that just hold the latest snapshot.
  If no state persistence implementation is provided when running a graph, this is used by default.
  """
  last_snapshot: Snapshot[StateT, RunEndT] | None = None
"""The last snapshot."""
  async defsnapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
    self.last_snapshot = NodeSnapshot(state=state, node=next_node)
  async defsnapshot_node_if_new(
    self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
  ) -> None:
    if self.last_snapshot and self.last_snapshot.id == snapshot_id:
      return
    else:
      await self.snapshot_node(state, next_node)
  async defsnapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
    self.last_snapshot = EndSnapshot(state=state, result=end)
  @asynccontextmanager
  async defrecord_run(self, snapshot_id: str) -> AsyncIterator[None]:
    if self.last_snapshot is None or snapshot_id != self.last_snapshot.id:
      raise LookupError(f'No snapshot found with id={snapshot_id!r}')
    assert isinstance(self.last_snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
    exceptions.GraphNodeStatusError.check(self.last_snapshot.status)
    self.last_snapshot.status = 'running'
    self.last_snapshot.start_ts = _utils.now_utc()
    start = perf_counter()
    try:
      yield
    except Exception:
      self.last_snapshot.duration = perf_counter() - start
      self.last_snapshot.status = 'error'
      raise
    else:
      self.last_snapshot.duration = perf_counter() - start
      self.last_snapshot.status = 'success'
  async defload_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
    if isinstance(self.last_snapshot, NodeSnapshot) and self.last_snapshot.status == 'created':
      self.last_snapshot.status = 'pending'
      return self.last_snapshot
  async defload_all(self) -> list[Snapshot[StateT, RunEndT]]:
    raise NotImplementedError('load is not supported for SimpleStatePersistence')

```
  
---|---  
####  last_snapshot `class-attribute` `instance-attribute`
```
last_snapshot: [StateT, RunEndT] | None = None

```

The last snapshot.
###  FullStatePersistence `dataclass`
Bases: `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT, RunEndT]`
In memory state persistence that hold a list of snapshots.
Source code in `pydantic_graph/pydantic_graph/persistence/in_mem.py`
```
```
| ```
@dataclass
classFullStatePersistence(BaseStatePersistence[StateT, RunEndT]):
"""In memory state persistence that hold a list of snapshots."""
  deep_copy: bool = True
"""Whether to deep copy the state and nodes when storing them.
  Defaults to `True` so even if nodes or state are modified after the snapshot is taken,
  the persistence history will record the value at the time of the snapshot.
  """
  history: list[Snapshot[StateT, RunEndT]] = field(default_factory=list)
"""List of snapshots taken during the graph run."""
  _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
    default=None, init=False, repr=False
  )
  async defsnapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
    snapshot = NodeSnapshot(
      state=self._prep_state(state),
      node=next_node.deep_copy() if self.deep_copy else next_node,
    )
    self.history.append(snapshot)
  async defsnapshot_node_if_new(
    self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
  ) -> None:
    if not any(s.id == snapshot_id for s in self.history):
      await self.snapshot_node(state, next_node)
  async defsnapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
    snapshot = EndSnapshot(
      state=self._prep_state(state),
      result=end.deep_copy_data() if self.deep_copy else end,
    )
    self.history.append(snapshot)
  @asynccontextmanager
  async defrecord_run(self, snapshot_id: str) -> AsyncIterator[None]:
    try:
      snapshot = next(s for s in self.history if s.id == snapshot_id)
    except StopIteration as e:
      raise LookupError(f'No snapshot found with id={snapshot_id!r}') frome
    assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
    exceptions.GraphNodeStatusError.check(snapshot.status)
    snapshot.status = 'running'
    snapshot.start_ts = _utils.now_utc()
    start = perf_counter()
    try:
      yield
    except Exception:
      snapshot.duration = perf_counter() - start
      snapshot.status = 'error'
      raise
    else:
      snapshot.duration = perf_counter() - start
      snapshot.status = 'success'
  async defload_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
    if snapshot := next((s for s in self.history if isinstance(s, NodeSnapshot) and s.status == 'created'), None):
      snapshot.status = 'pending'
      return snapshot
  async defload_all(self) -> list[Snapshot[StateT, RunEndT]]:
    return self.history
  defshould_set_types(self) -> bool:
    return self._snapshots_type_adapter is None
  defset_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
    self._snapshots_type_adapter = build_snapshot_list_type_adapter(state_type, run_end_type)
  defdump_json(self, *, indent: int | None = None) -> bytes:
"""Dump the history to JSON bytes."""
    assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `dump_json`'
    return self._snapshots_type_adapter.dump_json(self.history, indent=indent)
  defload_json(self, json_data: str | bytes | bytearray) -> None:
"""Load the history from JSON."""
    assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `load_json`'
    self.history = self._snapshots_type_adapter.validate_json(json_data)
  def_prep_state(self, state: StateT) -> StateT:
"""Prepare state for snapshot, uses [`copy.deepcopy`][copy.deepcopy] by default."""
    if not self.deep_copy or state is None:
      return state
    else:
      return copy.deepcopy(state)

```
  
---|---  
####  deep_copy `class-attribute` `instance-attribute`
```
deep_copy:  = True

```

Whether to deep copy the state and nodes when storing them.
Defaults to `True` so even if nodes or state are modified after the snapshot is taken, the persistence history will record the value at the time of the snapshot.
####  history `class-attribute` `instance-attribute`
```
history: [[StateT, RunEndT]] = (
  default_factory=
)

```

List of snapshots taken during the graph run.
####  dump_json
```
dump_json(*, indent:  | None = None) -> 

```

Dump the history to JSON bytes.
Source code in `pydantic_graph/pydantic_graph/persistence/in_mem.py`
```
```
| ```
defdump_json(self, *, indent: int | None = None) -> bytes:
"""Dump the history to JSON bytes."""
  assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `dump_json`'
  return self._snapshots_type_adapter.dump_json(self.history, indent=indent)

```
  
---|---  
####  load_json
```
load_json(json_data:  |  | ) -> None

```

Load the history from JSON.
Source code in `pydantic_graph/pydantic_graph/persistence/in_mem.py`
```
```
| ```
defload_json(self, json_data: str | bytes | bytearray) -> None:
"""Load the history from JSON."""
  assert self._snapshots_type_adapter is not None, 'type adapter must be set to use `load_json`'
  self.history = self._snapshots_type_adapter.validate_json(json_data)

```
  
---|---  
###  FileStatePersistence `dataclass`
Bases: `BaseStatePersistence[](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence "pydantic_graph.persistence.BaseStatePersistence")[StateT, RunEndT]`
File based state persistence that hold graph run state in a JSON file.
Source code in `pydantic_graph/pydantic_graph/persistence/file.py`
```
```
| ```
@dataclass
classFileStatePersistence(BaseStatePersistence[StateT, RunEndT]):
"""File based state persistence that hold graph run state in a JSON file."""
  json_file: Path
"""Path to the JSON file where the snapshots are stored.
  You should use a different file for each graph run, but a single file should be reused for multiple
  steps of the same run.
  For example if you have a run ID of the form `run_123abc`, you might create a `FileStatePersistence` thus:
  ```py
  from pathlib import Path
  from pydantic_graph import FullStatePersistence
  run_id = 'run_123abc'
  persistence = FullStatePersistence(Path('runs') / f'{run_id}.json')
  ```
  """
  _snapshots_type_adapter: pydantic.TypeAdapter[list[Snapshot[StateT, RunEndT]]] | None = field(
    default=None, init=False, repr=False
  )
  async defsnapshot_node(self, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]) -> None:
    await self._append_save(NodeSnapshot(state=state, node=next_node))
  async defsnapshot_node_if_new(
    self, snapshot_id: str, state: StateT, next_node: BaseNode[StateT, Any, RunEndT]
  ) -> None:
    async with self._lock():
      snapshots = await self.load_all()
      if not any(s.id == snapshot_id for s in snapshots):
        await self._append_save(NodeSnapshot(state=state, node=next_node), lock=False)
  async defsnapshot_end(self, state: StateT, end: End[RunEndT]) -> None:
    await self._append_save(EndSnapshot(state=state, result=end))
  @asynccontextmanager
  async defrecord_run(self, snapshot_id: str) -> AsyncIterator[None]:
    async with self._lock():
      snapshots = await self.load_all()
      try:
        snapshot = next(s for s in snapshots if s.id == snapshot_id)
      except StopIteration as e:
        raise LookupError(f'No snapshot found with id={snapshot_id!r}') frome
      assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
      exceptions.GraphNodeStatusError.check(snapshot.status)
      snapshot.status = 'running'
      snapshot.start_ts = _utils.now_utc()
      await self._save(snapshots)
    start = perf_counter()
    try:
      yield
    except Exception:
      duration = perf_counter() - start
      async with self._lock():
        await _graph_utils.run_in_executor(self._after_run_sync, snapshot_id, duration, 'error')
      raise
    else:
      snapshot.duration = perf_counter() - start
      async with self._lock():
        await _graph_utils.run_in_executor(self._after_run_sync, snapshot_id, snapshot.duration, 'success')
  async defload_next(self) -> NodeSnapshot[StateT, RunEndT] | None:
    async with self._lock():
      snapshots = await self.load_all()
      if snapshot := next((s for s in snapshots if isinstance(s, NodeSnapshot) and s.status == 'created'), None):
        snapshot.status = 'pending'
        await self._save(snapshots)
        return snapshot
  defshould_set_types(self) -> bool:
"""Whether types need to be set."""
    return self._snapshots_type_adapter is None
  defset_types(self, state_type: type[StateT], run_end_type: type[RunEndT]) -> None:
    self._snapshots_type_adapter = build_snapshot_list_type_adapter(state_type, run_end_type)
  async defload_all(self) -> list[Snapshot[StateT, RunEndT]]:
    return await _graph_utils.run_in_executor(self._load_sync)
  def_load_sync(self) -> list[Snapshot[StateT, RunEndT]]:
    assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
    try:
      content = self.json_file.read_bytes()
    except FileNotFoundError:
      return []
    else:
      return self._snapshots_type_adapter.validate_json(content)
  def_after_run_sync(self, snapshot_id: str, duration: float, status: SnapshotStatus) -> None:
    snapshots = self._load_sync()
    snapshot = next(s for s in snapshots if s.id == snapshot_id)
    assert isinstance(snapshot, NodeSnapshot), 'Only NodeSnapshot can be recorded'
    snapshot.duration = duration
    snapshot.status = status
    self._save_sync(snapshots)
  async def_save(self, snapshots: list[Snapshot[StateT, RunEndT]]) -> None:
    await _graph_utils.run_in_executor(self._save_sync, snapshots)
  def_save_sync(self, snapshots: list[Snapshot[StateT, RunEndT]]) -> None:
    assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
    self.json_file.write_bytes(self._snapshots_type_adapter.dump_json(snapshots, indent=2))
  async def_append_save(self, snapshot: Snapshot[StateT, RunEndT], *, lock: bool = True) -> None:
    assert self._snapshots_type_adapter is not None, 'snapshots type adapter must be set'
    async with AsyncExitStack() as stack:
      if lock:
        await stack.enter_async_context(self._lock())
      snapshots = await self.load_all()
      snapshots.append(snapshot)
      await self._save(snapshots)
  @asynccontextmanager
  async def_lock(self, *, timeout: float = 1.0) -> AsyncIterator[None]:
"""Lock a file by checking and writing a `.pydantic-graph-persistence-lock` to it.
    Args:
      timeout: how long to wait for the lock
    Returns: an async context manager that holds the lock
    """
    lock_file = self.json_file.parent / f'{self.json_file.name}.pydantic-graph-persistence-lock'
    lock_id = secrets.token_urlsafe().encode()
    await asyncio.wait_for(_get_lock(lock_file, lock_id), timeout=timeout)
    try:
      yield
    finally:
      await _graph_utils.run_in_executor(lock_file.unlink, missing_ok=True)

```
  
---|---  
####  json_file `instance-attribute`
```
json_file: 

```

Path to the JSON file where the snapshots are stored.
You should use a different file for each graph run, but a single file should be reused for multiple steps of the same run.
For example if you have a run ID of the form `run_123abc`, you might create a `FileStatePersistence` thus:
```
frompathlibimport Path
frompydantic_graphimport FullStatePersistence
run_id = 'run_123abc'
persistence = FullStatePersistence(Path('runs') / f'{run_id}.json')

```

####  should_set_types
```
should_set_types() -> 

```

Whether types need to be set.
Source code in `pydantic_graph/pydantic_graph/persistence/file.py`
```
```
| ```
defshould_set_types(self) -> bool:
"""Whether types need to be set."""
  return self._snapshots_type_adapter is None

```
  
---|---

# `pydantic_ai.result`
###  ResultDataT `module-attribute`
```
ResultDataT = TypeVar(
  "ResultDataT", default=, covariant=True
)

```

Covariant type variable for the result data type of a run.
###  StreamedRunResult `dataclass`
Bases: `Generic[](https://docs.python.org/3/library/typing.html#typing.Generic "typing.Generic")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT"), ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")]`
Result of a streamed run that returns structured data via a tool call.
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
@dataclass
classStreamedRunResult(Generic[AgentDepsT, ResultDataT]):
"""Result of a streamed run that returns structured data via a tool call."""
  _all_messages: list[_messages.ModelMessage]
  _new_message_index: int
  _usage_limits: UsageLimits | None
  _stream_response: models.StreamedResponse
  _result_schema: _result.ResultSchema[ResultDataT] | None
  _run_ctx: RunContext[AgentDepsT]
  _result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]]
  _result_tool_name: str | None
  _on_complete: Callable[[], Awaitable[None]]
  _initial_run_ctx_usage: Usage = field(init=False)
  is_complete: bool = field(default=False, init=False)
"""Whether the stream has all been received.
  This is set to `True` when one of
  [`stream`][pydantic_ai.result.StreamedRunResult.stream],
  [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text],
  [`stream_structured`][pydantic_ai.result.StreamedRunResult.stream_structured] or
  [`get_data`][pydantic_ai.result.StreamedRunResult.get_data] completes.
  """
  def__post_init__(self):
    self._initial_run_ctx_usage = copy(self._run_ctx.usage)
  defall_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
"""Return the history of _messages.
    Args:
      result_tool_return_content: The return content of the tool call to set in the last message.
        This provides a convenient way to modify the content of the result tool call if you want to continue
        the conversation and want to set the response to the result tool call. If `None`, the last message will
        not be modified.
    Returns:
      List of messages.
    """
    # this is a method to be consistent with the other methods
    if result_tool_return_content is not None:
      raise NotImplementedError('Setting result tool return content is not supported for this result type.')
    return self._all_messages
  defall_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
"""Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResult.all_messages] as JSON bytes.
    Args:
      result_tool_return_content: The return content of the tool call to set in the last message.
        This provides a convenient way to modify the content of the result tool call if you want to continue
        the conversation and want to set the response to the result tool call. If `None`, the last message will
        not be modified.
    Returns:
      JSON bytes representing the messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(
      self.all_messages(result_tool_return_content=result_tool_return_content)
    )
  defnew_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
"""Return new messages associated with this run.
    Messages from older runs are excluded.
    Args:
      result_tool_return_content: The return content of the tool call to set in the last message.
        This provides a convenient way to modify the content of the result tool call if you want to continue
        the conversation and want to set the response to the result tool call. If `None`, the last message will
        not be modified.
    Returns:
      List of new messages.
    """
    return self.all_messages(result_tool_return_content=result_tool_return_content)[self._new_message_index :]
  defnew_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
"""Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResult.new_messages] as JSON bytes.
    Args:
      result_tool_return_content: The return content of the tool call to set in the last message.
        This provides a convenient way to modify the content of the result tool call if you want to continue
        the conversation and want to set the response to the result tool call. If `None`, the last message will
        not be modified.
    Returns:
      JSON bytes representing the new messages.
    """
    return _messages.ModelMessagesTypeAdapter.dump_json(
      self.new_messages(result_tool_return_content=result_tool_return_content)
    )
  async defstream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[ResultDataT]:
"""Stream the response as an async iterable.
    The pydantic validator for structured data will be called in
    [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
    on each iteration.
    Args:
      debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
        Debouncing is particularly important for long structured responses to reduce the overhead of
        performing validation as each token is received.
    Returns:
      An async iterable of the response data.
    """
    async for structured_message, is_last in self.stream_structured(debounce_by=debounce_by):
      result = await self.validate_structured_result(structured_message, allow_partial=not is_last)
      yield result
  async defstream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
"""Stream the text result as an async iterable.
    !!! note
      Result validators will NOT be called on the text result if `delta=True`.
    Args:
      delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
        up to the current point.
      debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
        Debouncing is particularly important for long structured responses to reduce the overhead of
        performing validation as each token is received.
    """
    if self._result_schema and not self._result_schema.allow_text_result:
      raise exceptions.UserError('stream_text() can only be used with text responses')
    if delta:
      async for text in self._stream_response_text(delta=delta, debounce_by=debounce_by):
        yield text
    else:
      async for text in self._stream_response_text(delta=delta, debounce_by=debounce_by):
        combined_validated_text = await self._validate_text_result(text)
        yield combined_validated_text
    await self._marked_completed(self._stream_response.get())
  async defstream_structured(
    self, *, debounce_by: float | None = 0.1
  ) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
"""Stream the response as an async iterable of Structured LLM Messages.
    Args:
      debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
        Debouncing is particularly important for long structured responses to reduce the overhead of
        performing validation as each token is received.
    Returns:
      An async iterable of the structured response message and whether that is the last message.
    """
    # if the message currently has any parts with content, yield before streaming
    msg = self._stream_response.get()
    for part in msg.parts:
      if part.has_content():
        yield msg, False
        break
    async for msg in self._stream_response_structured(debounce_by=debounce_by):
      yield msg, False
    msg = self._stream_response.get()
    yield msg, True
    await self._marked_completed(msg)
  async defget_data(self) -> ResultDataT:
"""Stream the whole response, validate and return it."""
    usage_checking_stream = _get_usage_checking_stream_response(
      self._stream_response, self._usage_limits, self.usage
    )
    async for _ in usage_checking_stream:
      pass
    message = self._stream_response.get()
    await self._marked_completed(message)
    return await self.validate_structured_result(message)
  defusage(self) -> Usage:
"""Return the usage of the whole run.
    !!! note
      This won't return the full usage until the stream is finished.
    """
    return self._initial_run_ctx_usage + self._stream_response.usage()
  deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
    return self._stream_response.timestamp
  async defvalidate_structured_result(
    self, message: _messages.ModelResponse, *, allow_partial: bool = False
  ) -> ResultDataT:
"""Validate a structured result message."""
    if self._result_schema is not None and self._result_tool_name is not None:
      match = self._result_schema.find_named_tool(message.parts, self._result_tool_name)
      if match is None:
        raise exceptions.UnexpectedModelBehavior(
          f'Invalid response, unable to find tool: {self._result_schema.tool_names()}'
        )
      call, result_tool = match
      result_data = result_tool.validate(call, allow_partial=allow_partial, wrap_validation_errors=False)
      for validator in self._result_validators:
        result_data = await validator.validate(result_data, call, self._run_ctx)
      return result_data
    else:
      text = '\n\n'.join(x.content for x in message.parts if isinstance(x, _messages.TextPart))
      for validator in self._result_validators:
        text = await validator.validate(
          text,
          None,
          self._run_ctx,
        )
      # Since there is no result tool, we can assume that str is compatible with ResultDataT
      return cast(ResultDataT, text)
  async def_validate_text_result(self, text: str) -> str:
    for validator in self._result_validators:
      text = await validator.validate(
        text,
        None,
        self._run_ctx,
      )
    return text
  async def_marked_completed(self, message: _messages.ModelResponse) -> None:
    self.is_complete = True
    self._all_messages.append(message)
    await self._on_complete()
  async def_stream_response_structured(
    self, *, debounce_by: float | None = 0.1
  ) -> AsyncIterator[_messages.ModelResponse]:
    async with _utils.group_by_temporal(self._stream_response, debounce_by) as group_iter:
      async for _items in group_iter:
        yield self._stream_response.get()
  async def_stream_response_text(
    self, *, delta: bool = False, debounce_by: float | None = 0.1
  ) -> AsyncIterator[str]:
"""Stream the response as an async iterable of text."""
    # Define a "merged" version of the iterator that will yield items that have already been retrieved
    # and items that we receive while streaming. We define a dedicated async iterator for this so we can
    # pass the combined stream to the group_by_temporal function within `_stream_text_deltas` below.
    async def_stream_text_deltas_ungrouped() -> AsyncIterator[tuple[str, int]]:
      # yields tuples of (text_content, part_index)
      # we don't currently make use of the part_index, but in principle this may be useful
      # so we retain it here for now to make possible future refactors simpler
      msg = self._stream_response.get()
      for i, part in enumerate(msg.parts):
        if isinstance(part, _messages.TextPart) and part.content:
          yield part.content, i
      async for event in self._stream_response:
        if (
          isinstance(event, _messages.PartStartEvent)
          and isinstance(event.part, _messages.TextPart)
          and event.part.content
        ):
          yield event.part.content, event.index
        elif (
          isinstance(event, _messages.PartDeltaEvent)
          and isinstance(event.delta, _messages.TextPartDelta)
          and event.delta.content_delta
        ):
          yield event.delta.content_delta, event.index
    async def_stream_text_deltas() -> AsyncIterator[str]:
      async with _utils.group_by_temporal(_stream_text_deltas_ungrouped(), debounce_by) as group_iter:
        async for items in group_iter:
          # Note: we are currently just dropping the part index on the group here
          yield ''.join([content for content, _ in items])
    if delta:
      async for text in _stream_text_deltas():
        yield text
    else:
      # a quick benchmark shows it's faster to build up a string with concat when we're
      # yielding at each step
      deltas: list[str] = []
      async for text in _stream_text_deltas():
        deltas.append(text)
        yield ''.join(deltas)

```
  
---|---  
####  is_complete `class-attribute` `instance-attribute`
```
is_complete:  = (default=False, =False)

```

Whether the stream has all been received.
This is set to `True` when one of [`stream`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.stream), [`stream_text`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.stream_text), [`stream_structured`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.stream_structured) or [`get_data`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.get_data) completes.
####  all_messages
```
all_messages(
  *, result_tool_return_content:  | None = None
) -> []

```

Return the history of _messages.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`result_tool_return_content` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the result tool call if you want to continue the conversation and want to set the response to the result tool call. If `None`, the last message will not be modified. |  `None`  
Returns:
Type | Description  
---|---  
|  List of messages.  
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
defall_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
"""Return the history of _messages.
  Args:
    result_tool_return_content: The return content of the tool call to set in the last message.
      This provides a convenient way to modify the content of the result tool call if you want to continue
      the conversation and want to set the response to the result tool call. If `None`, the last message will
      not be modified.
  Returns:
    List of messages.
  """
  # this is a method to be consistent with the other methods
  if result_tool_return_content is not None:
    raise NotImplementedError('Setting result tool return content is not supported for this result type.')
  return self._all_messages

```
  
---|---  
####  all_messages_json
```
all_messages_json(
  *, result_tool_return_content:  | None = None
) -> 

```

Return all messages from [`all_messages`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.all_messages) as JSON bytes.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`result_tool_return_content` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the result tool call if you want to continue the conversation and want to set the response to the result tool call. If `None`, the last message will not be modified. |  `None`  
Returns:
Type | Description  
---|---  
|  JSON bytes representing the messages.  
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
defall_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
"""Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResult.all_messages] as JSON bytes.
  Args:
    result_tool_return_content: The return content of the tool call to set in the last message.
      This provides a convenient way to modify the content of the result tool call if you want to continue
      the conversation and want to set the response to the result tool call. If `None`, the last message will
      not be modified.
  Returns:
    JSON bytes representing the messages.
  """
  return _messages.ModelMessagesTypeAdapter.dump_json(
    self.all_messages(result_tool_return_content=result_tool_return_content)
  )

```
  
---|---  
####  new_messages
```
new_messages(
  *, result_tool_return_content:  | None = None
) -> []

```

Return new messages associated with this run.
Messages from older runs are excluded.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`result_tool_return_content` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the result tool call if you want to continue the conversation and want to set the response to the result tool call. If `None`, the last message will not be modified. |  `None`  
Returns:
Type | Description  
---|---  
|  List of new messages.  
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
defnew_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
"""Return new messages associated with this run.
  Messages from older runs are excluded.
  Args:
    result_tool_return_content: The return content of the tool call to set in the last message.
      This provides a convenient way to modify the content of the result tool call if you want to continue
      the conversation and want to set the response to the result tool call. If `None`, the last message will
      not be modified.
  Returns:
    List of new messages.
  """
  return self.all_messages(result_tool_return_content=result_tool_return_content)[self._new_message_index :]

```
  
---|---  
####  new_messages_json
```
new_messages_json(
  *, result_tool_return_content:  | None = None
) -> 

```

Return new messages from [`new_messages`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.new_messages) as JSON bytes.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`result_tool_return_content` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  The return content of the tool call to set in the last message. This provides a convenient way to modify the content of the result tool call if you want to continue the conversation and want to set the response to the result tool call. If `None`, the last message will not be modified. |  `None`  
Returns:
Type | Description  
---|---  
|  JSON bytes representing the new messages.  
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
defnew_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
"""Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResult.new_messages] as JSON bytes.
  Args:
    result_tool_return_content: The return content of the tool call to set in the last message.
      This provides a convenient way to modify the content of the result tool call if you want to continue
      the conversation and want to set the response to the result tool call. If `None`, the last message will
      not be modified.
  Returns:
    JSON bytes representing the new messages.
  """
  return _messages.ModelMessagesTypeAdapter.dump_json(
    self.new_messages(result_tool_return_content=result_tool_return_content)
  )

```
  
---|---  
####  stream `async`
```
stream(
  *, debounce_by:  | None = 0.1
) -> []

```

Stream the response as an async iterable.
The pydantic validator for structured data will be called in [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation) on each iteration.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`debounce_by` |  `float[](https://docs.python.org/3/library/functions.html#float) | None` |  by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing. Debouncing is particularly important for long structured responses to reduce the overhead of performing validation as each token is received. |  `0.1`  
Returns:
Type | Description  
---|---  
`AsyncIterator[](https://docs.python.org/3/library/collections.abc.html#collections.abc.AsyncIterator "collections.abc.AsyncIterator")[ResultDataT[](https://ai.pydantic.dev/api/result/#pydantic_ai.result.ResultDataT "pydantic_ai.result.ResultDataT")]` |  An async iterable of the response data.  
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
async defstream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[ResultDataT]:
"""Stream the response as an async iterable.
  The pydantic validator for structured data will be called in
  [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
  on each iteration.
  Args:
    debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
      Debouncing is particularly important for long structured responses to reduce the overhead of
      performing validation as each token is received.
  Returns:
    An async iterable of the response data.
  """
  async for structured_message, is_last in self.stream_structured(debounce_by=debounce_by):
    result = await self.validate_structured_result(structured_message, allow_partial=not is_last)
    yield result

```
  
---|---  
####  stream_text `async`
```
stream_text(
  *, delta:  = False, debounce_by:  | None = 0.1
) -> []

```

Stream the text result as an async iterable.
Note
Result validators will NOT be called on the text result if `delta=True`.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`delta` |  if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text up to the current point. |  `False`  
`debounce_by` |  `float[](https://docs.python.org/3/library/functions.html#float) | None` |  by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing. Debouncing is particularly important for long structured responses to reduce the overhead of performing validation as each token is received. |  `0.1`  
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
async defstream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
"""Stream the text result as an async iterable.
  !!! note
    Result validators will NOT be called on the text result if `delta=True`.
  Args:
    delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
      up to the current point.
    debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
      Debouncing is particularly important for long structured responses to reduce the overhead of
      performing validation as each token is received.
  """
  if self._result_schema and not self._result_schema.allow_text_result:
    raise exceptions.UserError('stream_text() can only be used with text responses')
  if delta:
    async for text in self._stream_response_text(delta=delta, debounce_by=debounce_by):
      yield text
  else:
    async for text in self._stream_response_text(delta=delta, debounce_by=debounce_by):
      combined_validated_text = await self._validate_text_result(text)
      yield combined_validated_text
  await self._marked_completed(self._stream_response.get())

```
  
---|---  
####  stream_structured `async`
```
stream_structured(
  *, debounce_by:  | None = 0.1
) -> [[, ]]

```

Stream the response as an async iterable of Structured LLM Messages.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`debounce_by` |  `float[](https://docs.python.org/3/library/functions.html#float) | None` |  by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing. Debouncing is particularly important for long structured responses to reduce the overhead of performing validation as each token is received. |  `0.1`  
Returns:
Type | Description  
---|---  
`AsyncIterator[](https://docs.python.org/3/library/collections.abc.html#collections.abc.AsyncIterator "collections.abc.AsyncIterator")[tuple[](https://docs.python.org/3/library/stdtypes.html#tuple)[ModelResponse[](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelResponse "pydantic_ai.messages.ModelResponse"), bool[](https://docs.python.org/3/library/functions.html#bool)]]` |  An async iterable of the structured response message and whether that is the last message.  
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
async defstream_structured(
  self, *, debounce_by: float | None = 0.1
) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
"""Stream the response as an async iterable of Structured LLM Messages.
  Args:
    debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
      Debouncing is particularly important for long structured responses to reduce the overhead of
      performing validation as each token is received.
  Returns:
    An async iterable of the structured response message and whether that is the last message.
  """
  # if the message currently has any parts with content, yield before streaming
  msg = self._stream_response.get()
  for part in msg.parts:
    if part.has_content():
      yield msg, False
      break
  async for msg in self._stream_response_structured(debounce_by=debounce_by):
    yield msg, False
  msg = self._stream_response.get()
  yield msg, True
  await self._marked_completed(msg)

```
  
---|---  
####  get_data `async`
```
get_data() -> 

```

Stream the whole response, validate and return it.
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
async defget_data(self) -> ResultDataT:
"""Stream the whole response, validate and return it."""
  usage_checking_stream = _get_usage_checking_stream_response(
    self._stream_response, self._usage_limits, self.usage
  )
  async for _ in usage_checking_stream:
    pass
  message = self._stream_response.get()
  await self._marked_completed(message)
  return await self.validate_structured_result(message)

```
  
---|---  
####  usage
```
usage() -> 

```

Return the usage of the whole run.
Note
This won't return the full usage until the stream is finished.
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
defusage(self) -> Usage:
"""Return the usage of the whole run.
  !!! note
    This won't return the full usage until the stream is finished.
  """
  return self._initial_run_ctx_usage + self._stream_response.usage()

```
  
---|---  
####  timestamp
```
timestamp() -> 

```

Get the timestamp of the response.
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
deftimestamp(self) -> datetime:
"""Get the timestamp of the response."""
  return self._stream_response.timestamp

```
  
---|---  
####  validate_structured_result `async`
```
validate_structured_result(
  message: , *, allow_partial:  = False
) -> 

```

Validate a structured result message.
Source code in `pydantic_ai_slim/pydantic_ai/result.py`
```
```
| ```
async defvalidate_structured_result(
  self, message: _messages.ModelResponse, *, allow_partial: bool = False
) -> ResultDataT:
"""Validate a structured result message."""
  if self._result_schema is not None and self._result_tool_name is not None:
    match = self._result_schema.find_named_tool(message.parts, self._result_tool_name)
    if match is None:
      raise exceptions.UnexpectedModelBehavior(
        f'Invalid response, unable to find tool: {self._result_schema.tool_names()}'
      )
    call, result_tool = match
    result_data = result_tool.validate(call, allow_partial=allow_partial, wrap_validation_errors=False)
    for validator in self._result_validators:
      result_data = await validator.validate(result_data, call, self._run_ctx)
    return result_data
  else:
    text = '\n\n'.join(x.content for x in message.parts if isinstance(x, _messages.TextPart))
    for validator in self._result_validators:
      text = await validator.validate(
        text,
        None,
        self._run_ctx,
      )
    # Since there is no result tool, we can assume that str is compatible with ResultDataT
    return cast(ResultDataT, text)

```
  
---|---

# `pydantic_ai.settings`
###  ModelSettings
Bases: 
Settings to configure an LLM.
Here we include only settings which apply to multiple models / model providers, though not all of these settings are supported by all models.
Source code in `pydantic_ai_slim/pydantic_ai/settings.py`
```
```
| ```
classModelSettings(TypedDict, total=False):
"""Settings to configure an LLM.
  Here we include only settings which apply to multiple models / model providers,
  though not all of these settings are supported by all models.
  """
  max_tokens: int
"""The maximum number of tokens to generate before stopping.
  Supported by:
  * Gemini
  * Anthropic
  * OpenAI
  * Groq
  * Cohere
  * Mistral
  * Bedrock
  """
  temperature: float
"""Amount of randomness injected into the response.
  Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to a model's
  maximum `temperature` for creative and generative tasks.
  Note that even with `temperature` of `0.0`, the results will not be fully deterministic.
  Supported by:
  * Gemini
  * Anthropic
  * OpenAI
  * Groq
  * Cohere
  * Mistral
  * Bedrock
  """
  top_p: float
"""An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
  So 0.1 means only the tokens comprising the top 10% probability mass are considered.
  You should either alter `temperature` or `top_p`, but not both.
  Supported by:
  * Gemini
  * Anthropic
  * OpenAI
  * Groq
  * Cohere
  * Mistral
  * Bedrock
  """
  timeout: float | Timeout
"""Override the client-level default timeout for a request, in seconds.
  Supported by:
  * Gemini
  * Anthropic
  * OpenAI
  * Groq
  * Mistral
  """
  parallel_tool_calls: bool
"""Whether to allow parallel tool calls.
  Supported by:
  * OpenAI (some models, not o1)
  * Groq
  * Anthropic
  """
  seed: int
"""The random seed to use for the model, theoretically allowing for deterministic results.
  Supported by:
  * OpenAI
  * Groq
  * Cohere
  * Mistral
  """
  presence_penalty: float
"""Penalize new tokens based on whether they have appeared in the text so far.
  Supported by:
  * OpenAI
  * Groq
  * Cohere
  * Gemini
  * Mistral
  """
  frequency_penalty: float
"""Penalize new tokens based on their existing frequency in the text so far.
  Supported by:
  * OpenAI
  * Groq
  * Cohere
  * Gemini
  * Mistral
  """
  logit_bias: dict[str, int]
"""Modify the likelihood of specified tokens appearing in the completion.
  Supported by:
  * OpenAI
  * Groq
  """

```
  
---|---  
####  max_tokens `instance-attribute`
```
max_tokens: 

```

The maximum number of tokens to generate before stopping.
Supported by:
  * Gemini
  * Anthropic
  * OpenAI
  * Groq
  * Cohere
  * Mistral
  * Bedrock


####  temperature `instance-attribute`
```
temperature: 

```

Amount of randomness injected into the response.
Use `temperature` closer to `0.0` for analytical / multiple choice, and closer to a model's maximum `temperature` for creative and generative tasks.
Note that even with `temperature` of `0.0`, the results will not be fully deterministic.
Supported by:
  * Gemini
  * Anthropic
  * OpenAI
  * Groq
  * Cohere
  * Mistral
  * Bedrock


####  top_p `instance-attribute`
```
top_p: 

```

An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
So 0.1 means only the tokens comprising the top 10% probability mass are considered.
You should either alter `temperature` or `top_p`, but not both.
Supported by:
  * Gemini
  * Anthropic
  * OpenAI
  * Groq
  * Cohere
  * Mistral
  * Bedrock


####  timeout `instance-attribute`
```
timeout:  | Timeout

```

Override the client-level default timeout for a request, in seconds.
Supported by:
  * Gemini
  * Anthropic
  * OpenAI
  * Groq
  * Mistral


####  parallel_tool_calls `instance-attribute`
```
parallel_tool_calls: 

```

Whether to allow parallel tool calls.
Supported by:
  * OpenAI (some models, not o1)
  * Groq
  * Anthropic


####  seed `instance-attribute`
```
seed: 

```

The random seed to use for the model, theoretically allowing for deterministic results.
Supported by:
  * OpenAI
  * Groq
  * Cohere
  * Mistral


####  presence_penalty `instance-attribute`
```
presence_penalty: 

```

Penalize new tokens based on whether they have appeared in the text so far.
Supported by:
  * OpenAI
  * Groq
  * Cohere
  * Gemini
  * Mistral


####  frequency_penalty `instance-attribute`
```
frequency_penalty: 

```

Penalize new tokens based on their existing frequency in the text so far.
Supported by:
  * OpenAI
  * Groq
  * Cohere
  * Gemini
  * Mistral


####  logit_bias `instance-attribute`
```
logit_bias: [, ]

```

Modify the likelihood of specified tokens appearing in the completion.
Supported by:
  * OpenAI
  * Groq



# `pydantic_ai.tools`
###  AgentDepsT `module-attribute`
```
AgentDepsT = TypeVar(
  "AgentDepsT", default=None, contravariant=True
)

```

Type variable for agent dependencies.
###  RunContext `dataclass`
Bases: 
Information about the current call.
Source code in `pydantic_ai_slim/pydantic_ai/tools.py`
```
```
| ```
@dataclasses.dataclass
classRunContext(Generic[AgentDepsT]):
"""Information about the current call."""
  deps: AgentDepsT
"""Dependencies for the agent."""
  model: models.Model
"""The model used in this run."""
  usage: Usage
"""LLM usage associated with the run."""
  prompt: str | Sequence[_messages.UserContent]
"""The original user prompt passed to the run."""
  messages: list[_messages.ModelMessage] = field(default_factory=list)
"""Messages exchanged in the conversation so far."""
  tool_call_id: str | None = None
"""The ID of the tool call."""
  tool_name: str | None = None
"""Name of the tool being called."""
  retry: int = 0
"""Number of retries so far."""
  run_step: int = 0
"""The current step in the run."""
  defreplace_with(
    self, retry: int | None = None, tool_name: str | None | _utils.Unset = _utils.UNSET
  ) -> RunContext[AgentDepsT]:
    # Create a new `RunContext` a new `retry` value and `tool_name`.
    kwargs = {}
    if retry is not None:
      kwargs['retry'] = retry
    if tool_name is not _utils.UNSET:
      kwargs['tool_name'] = tool_name
    return dataclasses.replace(self, **kwargs)

```
  
---|---  
####  deps `instance-attribute`
```
deps: 

```

Dependencies for the agent.
####  model `instance-attribute`
```
model: 

```

The model used in this run.
####  usage `instance-attribute`
```
usage: 

```

LLM usage associated with the run.
####  prompt `instance-attribute`
```
prompt:  | [UserContent]

```

The original user prompt passed to the run.
####  messages `class-attribute` `instance-attribute`
```
messages: [] = (default_factory=)

```

Messages exchanged in the conversation so far.
####  tool_call_id `class-attribute` `instance-attribute`
```
tool_call_id:  | None = None

```

The ID of the tool call.
####  tool_name `class-attribute` `instance-attribute`
```
tool_name:  | None = None

```

Name of the tool being called.
####  retry `class-attribute` `instance-attribute`
```
retry:  = 0

```

Number of retries so far.
####  run_step `class-attribute` `instance-attribute`
```
run_step:  = 0

```

The current step in the run.
###  ToolParams `module-attribute`
```
ToolParams = ('ToolParams', default=...)

```

Retrieval function param spec.
###  SystemPromptFunc `module-attribute`
```
SystemPromptFunc = [
  [[[]], ],
  [[[]], []],
  [[], ],
  [[], []],
]

```

A function that may or maybe not take `RunContext` as an argument, and may or may not be async.
Usage `SystemPromptFunc[AgentDepsT]`.
###  ToolFuncContext `module-attribute`
```
ToolFuncContext = [
  [[], ], 
]

```

A tool function that takes `RunContext` as the first argument.
Usage `ToolContextFunc[AgentDepsT, ToolParams]`.
###  ToolFuncPlain `module-attribute`
```
ToolFuncPlain = [, ]

```

A tool function that does not take `RunContext` as the first argument.
Usage `ToolPlainFunc[ToolParams]`.
###  ToolFuncEither `module-attribute`
```
ToolFuncEither = [
  [, ],
  [],
]

```

Either kind of tool function.
This is just a union of [`ToolFuncContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolFuncContext) and [`ToolFuncPlain`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolFuncPlain).
Usage `ToolFuncEither[AgentDepsT, ToolParams]`.
###  ToolPrepareFunc `module-attribute`
```
ToolPrepareFunc:  = (
  "Callable[[RunContext[AgentDepsT], ToolDefinition], Awaitable[ToolDefinition | None]]"
)

```

Definition of a function that can prepare a tool definition at call time.
See [tool docs](https://ai.pydantic.dev/tools/#tool-prepare) for more information.
Example — here `only_if_42` is valid as a `ToolPrepareFunc`:
```
fromtypingimport Union
frompydantic_aiimport RunContext, Tool
frompydantic_ai.toolsimport ToolDefinition
async defonly_if_42(
  ctx: RunContext[int], tool_def: ToolDefinition
) -> Union[ToolDefinition, None]:
  if ctx.deps == 42:
    return tool_def
defhitchhiker(ctx: RunContext[int], answer: str) -> str:
  return f'{ctx.deps}{answer}'
hitchhiker = Tool(hitchhiker, prepare=only_if_42)

```

Usage `ToolPrepareFunc[AgentDepsT]`.
###  DocstringFormat `module-attribute`
```
DocstringFormat = [
  "google", "numpy", "sphinx", "auto"
]

```

Supported docstring formats.
  * `'google'` — [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings.
  * `'numpy'` — [Numpy-style](https://numpydoc.readthedocs.io/en/latest/format.html) docstrings.
  * `'sphinx'` — [Sphinx-style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format) docstrings.
  * `'auto'` — Automatically infer the format based on the structure of the docstring.


###  Tool `dataclass`
Bases: 
A tool function for an agent.
Source code in `pydantic_ai_slim/pydantic_ai/tools.py`
```
```
| ```
@dataclass(init=False)
classTool(Generic[AgentDepsT]):
"""A tool function for an agent."""
  function: ToolFuncEither[AgentDepsT]
  takes_ctx: bool
  max_retries: int | None
  name: str
  description: str
  prepare: ToolPrepareFunc[AgentDepsT] | None
  docstring_format: DocstringFormat
  require_parameter_descriptions: bool
  _is_async: bool = field(init=False)
  _single_arg_name: str | None = field(init=False)
  _positional_fields: list[str] = field(init=False)
  _var_positional_field: str | None = field(init=False)
  _validator: SchemaValidator = field(init=False, repr=False)
  _parameters_json_schema: ObjectJsonSchema = field(init=False)
  # TODO: Move this state off the Tool class, which is otherwise stateless.
  #  This should be tracked inside a specific agent run, not the tool.
  current_retry: int = field(default=0, init=False)
  def__init__(
    self,
    function: ToolFuncEither[AgentDepsT],
    *,
    takes_ctx: bool | None = None,
    max_retries: int | None = None,
    name: str | None = None,
    description: str | None = None,
    prepare: ToolPrepareFunc[AgentDepsT] | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
    schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
  ):
"""Create a new tool instance.
    Example usage:
    ```python {noqa="I001"}
    from pydantic_ai import Agent, RunContext, Tool
    async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
      return f'{ctx.deps} {x} {y}'
    agent = Agent('test', tools=[Tool(my_tool)])
    ```
    or with a custom prepare method:
    ```python {noqa="I001"}
    from typing import Union
    from pydantic_ai import Agent, RunContext, Tool
    from pydantic_ai.tools import ToolDefinition
    async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
      return f'{ctx.deps} {x} {y}'
    async def prep_my_tool(
      ctx: RunContext[int], tool_def: ToolDefinition
    ) -> Union[ToolDefinition, None]:
      # only register the tool if `deps == 42`
      if ctx.deps == 42:
        return tool_def
    agent = Agent('test', tools=[Tool(my_tool, prepare=prep_my_tool)])
    ```

    Args:
      function: The Python function to call as the tool.
      takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] first argument,
        this is inferred if unset.
      max_retries: Maximum number of retries allowed for this tool, set to the agent default if `None`.
      name: Name of the tool, inferred from the function if `None`.
      description: Description of the tool, inferred from the function if `None`.
      prepare: custom method to prepare the tool definition for each step, return `None` to omit this
        tool from a given step. This is useful if you want to customise a tool at call time,
        or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
      docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
        Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
      require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
      schema_generator: The JSON schema generator class to use. Defaults to `GenerateToolJsonSchema`.
    """
    if takes_ctx is None:
      takes_ctx = _pydantic.takes_ctx(function)
    f = _pydantic.function_schema(
      function, takes_ctx, docstring_format, require_parameter_descriptions, schema_generator
    )
    self.function = function
    self.takes_ctx = takes_ctx
    self.max_retries = max_retries
    self.name = name or function.__name__
    self.description = description or f['description']
    self.prepare = prepare
    self.docstring_format = docstring_format
    self.require_parameter_descriptions = require_parameter_descriptions
    self._is_async = inspect.iscoroutinefunction(self.function)
    self._single_arg_name = f['single_arg_name']
    self._positional_fields = f['positional_fields']
    self._var_positional_field = f['var_positional_field']
    self._validator = f['validator']
    self._parameters_json_schema = f['json_schema']
  async defprepare_tool_def(self, ctx: RunContext[AgentDepsT]) -> ToolDefinition | None:
"""Get the tool definition.
    By default, this method creates a tool definition, then either returns it, or calls `self.prepare`
    if it's set.
    Returns:
      return a `ToolDefinition` or `None` if the tools should not be registered for this run.
    """
    tool_def = ToolDefinition(
      name=self.name,
      description=self.description,
      parameters_json_schema=self._parameters_json_schema,
    )
    if self.prepare is not None:
      return await self.prepare(ctx, tool_def)
    else:
      return tool_def
  async defrun(
    self, message: _messages.ToolCallPart, run_context: RunContext[AgentDepsT]
  ) -> _messages.ToolReturnPart | _messages.RetryPromptPart:
"""Run the tool function asynchronously."""
    try:
      if isinstance(message.args, str):
        args_dict = self._validator.validate_json(message.args)
      else:
        args_dict = self._validator.validate_python(message.args)
    except ValidationError as e:
      return self._on_error(e, message)
    args, kwargs = self._call_args(args_dict, message, run_context)
    try:
      if self._is_async:
        function = cast(Callable[[Any], Awaitable[str]], self.function)
        response_content = await function(*args, **kwargs)
      else:
        function = cast(Callable[[Any], str], self.function)
        response_content = await _utils.run_in_executor(function, *args, **kwargs)
    except ModelRetry as e:
      return self._on_error(e, message)
    self.current_retry = 0
    return _messages.ToolReturnPart(
      tool_name=message.tool_name,
      content=response_content,
      tool_call_id=message.tool_call_id,
    )
  def_call_args(
    self,
    args_dict: dict[str, Any],
    message: _messages.ToolCallPart,
    run_context: RunContext[AgentDepsT],
  ) -> tuple[list[Any], dict[str, Any]]:
    if self._single_arg_name:
      args_dict = {self._single_arg_name: args_dict}
    ctx = dataclasses.replace(
      run_context,
      retry=self.current_retry,
      tool_name=message.tool_name,
      tool_call_id=message.tool_call_id,
    )
    args = [ctx] if self.takes_ctx else []
    for positional_field in self._positional_fields:
      args.append(args_dict.pop(positional_field))
    if self._var_positional_field:
      args.extend(args_dict.pop(self._var_positional_field))
    return args, args_dict
  def_on_error(
    self, exc: ValidationError | ModelRetry, call_message: _messages.ToolCallPart
  ) -> _messages.RetryPromptPart:
    self.current_retry += 1
    if self.max_retries is None or self.current_retry  self.max_retries:
      raise UnexpectedModelBehavior(f'Tool exceeded max retries count of {self.max_retries}') fromexc
    else:
      if isinstance(exc, ValidationError):
        content = exc.errors(include_url=False)
      else:
        content = exc.message
      return _messages.RetryPromptPart(
        tool_name=call_message.tool_name,
        content=content,
        tool_call_id=call_message.tool_call_id,
      )

```
  
---|---  
####  __init__
```
__init__(
  function: [],
  *,
  takes_ctx:  | None = None,
  max_retries:  | None = None,
  name:  | None = None,
  description:  | None = None,
  prepare: [] | None = None,
  docstring_format:  = "auto",
  require_parameter_descriptions:  = False,
  schema_generator: [
    
  ] = GenerateToolJsonSchema
)

```

Create a new tool instance.
Example usage:
```
frompydantic_aiimport Agent, RunContext, Tool
async defmy_tool(ctx: RunContext[int], x: int, y: int) -> str:
  return f'{ctx.deps}{x}{y}'
agent = Agent('test', tools=[Tool(my_tool)])

```

or with a custom prepare method:
```
fromtypingimport Union
frompydantic_aiimport Agent, RunContext, Tool
frompydantic_ai.toolsimport ToolDefinition
async defmy_tool(ctx: RunContext[int], x: int, y: int) -> str:
  return f'{ctx.deps}{x}{y}'
async defprep_my_tool(
  ctx: RunContext[int], tool_def: ToolDefinition
) -> Union[ToolDefinition, None]:
  # only register the tool if `deps == 42`
  if ctx.deps == 42:
    return tool_def
agent = Agent('test', tools=[Tool(my_tool, prepare=prep_my_tool)])

```

Parameters:
Name | Type | Description | Default  
---|---|---|---  
`function` |  `ToolFuncEither[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolFuncEither "pydantic_ai.tools.ToolFuncEither")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT")]` |  The Python function to call as the tool. |  _required_  
`takes_ctx` |  `bool[](https://docs.python.org/3/library/functions.html#bool) | None` |  Whether the function takes a [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) first argument, this is inferred if unset. |  `None`  
`max_retries` |  `int[](https://docs.python.org/3/library/functions.html#int) | None` |  Maximum number of retries allowed for this tool, set to the agent default if `None`. |  `None`  
`name` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  Name of the tool, inferred from the function if `None`. |  `None`  
`description` |  `str[](https://docs.python.org/3/library/stdtypes.html#str) | None` |  Description of the tool, inferred from the function if `None`. |  `None`  
`prepare` |  `ToolPrepareFunc[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolPrepareFunc "pydantic_ai.tools.ToolPrepareFunc")[AgentDepsT[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.AgentDepsT "pydantic_ai.tools.AgentDepsT")] | None` |  custom method to prepare the tool definition for each step, return `None` to omit this tool from a given step. This is useful if you want to customise a tool at call time, or omit it completely from a step. See [`ToolPrepareFunc`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolPrepareFunc). |  `None`  
`docstring_format` |  |  The format of the docstring, see [`DocstringFormat`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.DocstringFormat). Defaults to `'auto'`, such that the format is inferred from the structure of the docstring. |  `'auto'`  
`require_parameter_descriptions` |  If True, raise an error if a parameter description is missing. Defaults to False. |  `False`  
`schema_generator` |  |  The JSON schema generator class to use. Defaults to `GenerateToolJsonSchema`. |  `GenerateToolJsonSchema`  
Source code in `pydantic_ai_slim/pydantic_ai/tools.py`
```
```
| ```
def__init__(
  self,
  function: ToolFuncEither[AgentDepsT],
  *,
  takes_ctx: bool | None = None,
  max_retries: int | None = None,
  name: str | None = None,
  description: str | None = None,
  prepare: ToolPrepareFunc[AgentDepsT] | None = None,
  docstring_format: DocstringFormat = 'auto',
  require_parameter_descriptions: bool = False,
  schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
):
"""Create a new tool instance.
  Example usage:
  ```python {noqa="I001"}
  from pydantic_ai import Agent, RunContext, Tool
  async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
    return f'{ctx.deps} {x} {y}'
  agent = Agent('test', tools=[Tool(my_tool)])
  ```
  or with a custom prepare method:
  ```python {noqa="I001"}
  from typing import Union
  from pydantic_ai import Agent, RunContext, Tool
  from pydantic_ai.tools import ToolDefinition
  async def my_tool(ctx: RunContext[int], x: int, y: int) -> str:
    return f'{ctx.deps} {x} {y}'
  async def prep_my_tool(
    ctx: RunContext[int], tool_def: ToolDefinition
  ) -> Union[ToolDefinition, None]:
    # only register the tool if `deps == 42`
    if ctx.deps == 42:
      return tool_def
  agent = Agent('test', tools=[Tool(my_tool, prepare=prep_my_tool)])
  ```

  Args:
    function: The Python function to call as the tool.
    takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] first argument,
      this is inferred if unset.
    max_retries: Maximum number of retries allowed for this tool, set to the agent default if `None`.
    name: Name of the tool, inferred from the function if `None`.
    description: Description of the tool, inferred from the function if `None`.
    prepare: custom method to prepare the tool definition for each step, return `None` to omit this
      tool from a given step. This is useful if you want to customise a tool at call time,
      or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
    docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
      Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
    require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
    schema_generator: The JSON schema generator class to use. Defaults to `GenerateToolJsonSchema`.
  """
  if takes_ctx is None:
    takes_ctx = _pydantic.takes_ctx(function)
  f = _pydantic.function_schema(
    function, takes_ctx, docstring_format, require_parameter_descriptions, schema_generator
  )
  self.function = function
  self.takes_ctx = takes_ctx
  self.max_retries = max_retries
  self.name = name or function.__name__
  self.description = description or f['description']
  self.prepare = prepare
  self.docstring_format = docstring_format
  self.require_parameter_descriptions = require_parameter_descriptions
  self._is_async = inspect.iscoroutinefunction(self.function)
  self._single_arg_name = f['single_arg_name']
  self._positional_fields = f['positional_fields']
  self._var_positional_field = f['var_positional_field']
  self._validator = f['validator']
  self._parameters_json_schema = f['json_schema']

```
  
---|---  
####  prepare_tool_def `async`
```
prepare_tool_def(
  ctx: [],
) ->  | None

```

Get the tool definition.
By default, this method creates a tool definition, then either returns it, or calls `self.prepare` if it's set.
Returns:
Type | Description  
---|---  
`ToolDefinition[](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolDefinition "pydantic_ai.tools.ToolDefinition") | None` |  return a `ToolDefinition` or `None` if the tools should not be registered for this run.  
Source code in `pydantic_ai_slim/pydantic_ai/tools.py`
```
```
| ```
async defprepare_tool_def(self, ctx: RunContext[AgentDepsT]) -> ToolDefinition | None:
"""Get the tool definition.
  By default, this method creates a tool definition, then either returns it, or calls `self.prepare`
  if it's set.
  Returns:
    return a `ToolDefinition` or `None` if the tools should not be registered for this run.
  """
  tool_def = ToolDefinition(
    name=self.name,
    description=self.description,
    parameters_json_schema=self._parameters_json_schema,
  )
  if self.prepare is not None:
    return await self.prepare(ctx, tool_def)
  else:
    return tool_def

```
  
---|---  
####  run `async`
```
run(
  message: ,
  run_context: [],
) ->  | 

```

Run the tool function asynchronously.
Source code in `pydantic_ai_slim/pydantic_ai/tools.py`
```
```
| ```
async defrun(
  self, message: _messages.ToolCallPart, run_context: RunContext[AgentDepsT]
) -> _messages.ToolReturnPart | _messages.RetryPromptPart:
"""Run the tool function asynchronously."""
  try:
    if isinstance(message.args, str):
      args_dict = self._validator.validate_json(message.args)
    else:
      args_dict = self._validator.validate_python(message.args)
  except ValidationError as e:
    return self._on_error(e, message)
  args, kwargs = self._call_args(args_dict, message, run_context)
  try:
    if self._is_async:
      function = cast(Callable[[Any], Awaitable[str]], self.function)
      response_content = await function(*args, **kwargs)
    else:
      function = cast(Callable[[Any], str], self.function)
      response_content = await _utils.run_in_executor(function, *args, **kwargs)
  except ModelRetry as e:
    return self._on_error(e, message)
  self.current_retry = 0
  return _messages.ToolReturnPart(
    tool_name=message.tool_name,
    content=response_content,
    tool_call_id=message.tool_call_id,
  )

```
  
---|---  
###  ObjectJsonSchema `module-attribute`
```
ObjectJsonSchema:  = [, ]

```

Type representing JSON schema of an object, e.g. where `"type": "object"`.
This type is used to define tools parameters (aka arguments) in [ToolDefinition](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolDefinition).
With PEP-728 this should be a TypedDict with `type: Literal['object']`, and `extra_parts=Any`
###  ToolDefinition `dataclass`
Definition of a tool passed to a model.
This is used for both function tools result tools.
Source code in `pydantic_ai_slim/pydantic_ai/tools.py`
```
```
| ```
@dataclass
classToolDefinition:
"""Definition of a tool passed to a model.
  This is used for both function tools result tools.
  """
  name: str
"""The name of the tool."""
  description: str
"""The description of the tool."""
  parameters_json_schema: ObjectJsonSchema
"""The JSON schema for the tool's parameters."""
  outer_typed_dict_key: str | None = None
"""The key in the outer [TypedDict] that wraps a result tool.
  This will only be set for result tools which don't have an `object` JSON schema.
  """

```
  
---|---  
####  name `instance-attribute`
```
name: 

```

The name of the tool.
####  description `instance-attribute`
```
description: 

```

The description of the tool.
####  parameters_json_schema `instance-attribute`
```
parameters_json_schema: 

```

The JSON schema for the tool's parameters.
####  outer_typed_dict_key `class-attribute` `instance-attribute`
```
outer_typed_dict_key:  | None = None

```

The key in the outer [TypedDict] that wraps a result tool.
This will only be set for result tools which don't have an `object` JSON schema.

# `pydantic_ai.usage`
###  Usage `dataclass`
LLM usage associated with a request or run.
Responsibility for calculating usage is on the model; PydanticAI simply sums the usage information across requests.
You'll need to look up the documentation of the model you're using to convert usage to monetary costs.
Source code in `pydantic_ai_slim/pydantic_ai/usage.py`
```
```
| ```
@dataclass
classUsage:
"""LLM usage associated with a request or run.
  Responsibility for calculating usage is on the model; PydanticAI simply sums the usage information across requests.
  You'll need to look up the documentation of the model you're using to convert usage to monetary costs.
  """
  requests: int = 0
"""Number of requests made to the LLM API."""
  request_tokens: int | None = None
"""Tokens used in processing requests."""
  response_tokens: int | None = None
"""Tokens used in generating responses."""
  total_tokens: int | None = None
"""Total tokens used in the whole run, should generally be equal to `request_tokens + response_tokens`."""
  details: dict[str, int] | None = None
"""Any extra details returned by the model."""
  defincr(self, incr_usage: Usage, *, requests: int = 0) -> None:
"""Increment the usage in place.
    Args:
      incr_usage: The usage to increment by.
      requests: The number of requests to increment by in addition to `incr_usage.requests`.
    """
    self.requests += requests
    for f in 'requests', 'request_tokens', 'response_tokens', 'total_tokens':
      self_value = getattr(self, f)
      other_value = getattr(incr_usage, f)
      if self_value is not None or other_value is not None:
        setattr(self, f, (self_value or 0) + (other_value or 0))
    if incr_usage.details:
      self.details = self.details or {}
      for key, value in incr_usage.details.items():
        self.details[key] = self.details.get(key, 0) + value
  def__add__(self, other: Usage) -> Usage:
"""Add two Usages together.
    This is provided so it's trivial to sum usage information from multiple requests and runs.
    """
    new_usage = copy(self)
    new_usage.incr(other)
    return new_usage
  defopentelemetry_attributes(self) -> dict[str, int]:
"""Get the token limits as OpenTelemetry attributes."""
    result = {
      'gen_ai.usage.input_tokens': self.request_tokens,
      'gen_ai.usage.output_tokens': self.response_tokens,
    }
    for key, value in (self.details or {}).items():
      result[f'gen_ai.usage.details.{key}'] = value
    return {k: v for k, v in result.items() if v}

```
  
---|---  
####  requests `class-attribute` `instance-attribute`
```
requests:  = 0

```

Number of requests made to the LLM API.
####  request_tokens `class-attribute` `instance-attribute`
```
request_tokens:  | None = None

```

Tokens used in processing requests.
####  response_tokens `class-attribute` `instance-attribute`
```
response_tokens:  | None = None

```

Tokens used in generating responses.
####  total_tokens `class-attribute` `instance-attribute`
```
total_tokens:  | None = None

```

Total tokens used in the whole run, should generally be equal to `request_tokens + response_tokens`.
####  details `class-attribute` `instance-attribute`
```
details: [, ] | None = None

```

Any extra details returned by the model.
####  incr
```
incr(incr_usage: , *, requests:  = 0) -> None

```

Increment the usage in place.
Parameters:
Name | Type | Description | Default  
---|---|---|---  
`incr_usage` |  |  The usage to increment by. |  _required_  
`requests` |  The number of requests to increment by in addition to `incr_usage.requests`.  
Source code in `pydantic_ai_slim/pydantic_ai/usage.py`
```
```
| ```
defincr(self, incr_usage: Usage, *, requests: int = 0) -> None:
"""Increment the usage in place.
  Args:
    incr_usage: The usage to increment by.
    requests: The number of requests to increment by in addition to `incr_usage.requests`.
  """
  self.requests += requests
  for f in 'requests', 'request_tokens', 'response_tokens', 'total_tokens':
    self_value = getattr(self, f)
    other_value = getattr(incr_usage, f)
    if self_value is not None or other_value is not None:
      setattr(self, f, (self_value or 0) + (other_value or 0))
  if incr_usage.details:
    self.details = self.details or {}
    for key, value in incr_usage.details.items():
      self.details[key] = self.details.get(key, 0) + value

```
  
---|---  
####  __add__
```
__add__(other: ) -> 

```

Add two Usages together.
This is provided so it's trivial to sum usage information from multiple requests and runs.
Source code in `pydantic_ai_slim/pydantic_ai/usage.py`
```
```
| ```
def__add__(self, other: Usage) -> Usage:
"""Add two Usages together.
  This is provided so it's trivial to sum usage information from multiple requests and runs.
  """
  new_usage = copy(self)
  new_usage.incr(other)
  return new_usage

```
  
---|---  
####  opentelemetry_attributes
```
opentelemetry_attributes() -> [, ]

```

Get the token limits as OpenTelemetry attributes.
Source code in `pydantic_ai_slim/pydantic_ai/usage.py`
```
```
| ```
defopentelemetry_attributes(self) -> dict[str, int]:
"""Get the token limits as OpenTelemetry attributes."""
  result = {
    'gen_ai.usage.input_tokens': self.request_tokens,
    'gen_ai.usage.output_tokens': self.response_tokens,
  }
  for key, value in (self.details or {}).items():
    result[f'gen_ai.usage.details.{key}'] = value
  return {k: v for k, v in result.items() if v}

```
  
---|---  
###  UsageLimits `dataclass`
Limits on model usage.
The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model. Token counts are provided in responses from the model, and the token limits are checked after each response.
Each of the limits can be set to `None` to disable that limit.
Source code in `pydantic_ai_slim/pydantic_ai/usage.py`
```
```
| ```
@dataclass
classUsageLimits:
"""Limits on model usage.
  The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model.
  Token counts are provided in responses from the model, and the token limits are checked after each response.
  Each of the limits can be set to `None` to disable that limit.
  """
  request_limit: int | None = 50
"""The maximum number of requests allowed to the model."""
  request_tokens_limit: int | None = None
"""The maximum number of tokens allowed in requests to the model."""
  response_tokens_limit: int | None = None
"""The maximum number of tokens allowed in responses from the model."""
  total_tokens_limit: int | None = None
"""The maximum number of tokens allowed in requests and responses combined."""
  defhas_token_limits(self) -> bool:
"""Returns `True` if this instance places any limits on token counts.
    If this returns `False`, the `check_tokens` method will never raise an error.
    This is useful because if we have token limits, we need to check them after receiving each streamed message.
    If there are no limits, we can skip that processing in the streaming response iterator.
    """
    return any(
      limit is not None
      for limit in (self.request_tokens_limit, self.response_tokens_limit, self.total_tokens_limit)
    )
  defcheck_before_request(self, usage: Usage) -> None:
"""Raises a `UsageLimitExceeded` exception if the next request would exceed the request_limit."""
    request_limit = self.request_limit
    if request_limit is not None and usage.requests >= request_limit:
      raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')
  defcheck_tokens(self, usage: Usage) -> None:
"""Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
    request_tokens = usage.request_tokens or 0
    if self.request_tokens_limit is not None and request_tokens  self.request_tokens_limit:
      raise UsageLimitExceeded(
        f'Exceeded the request_tokens_limit of {self.request_tokens_limit} ({request_tokens=})'
      )
    response_tokens = usage.response_tokens or 0
    if self.response_tokens_limit is not None and response_tokens  self.response_tokens_limit:
      raise UsageLimitExceeded(
        f'Exceeded the response_tokens_limit of {self.response_tokens_limit} ({response_tokens=})'
      )
    total_tokens = usage.total_tokens or 0
    if self.total_tokens_limit is not None and total_tokens  self.total_tokens_limit:
      raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

```
  
---|---  
####  request_limit `class-attribute` `instance-attribute`
```
request_limit:  | None = 50

```

The maximum number of requests allowed to the model.
####  request_tokens_limit `class-attribute` `instance-attribute`
```
request_tokens_limit:  | None = None

```

The maximum number of tokens allowed in requests to the model.
####  response_tokens_limit `class-attribute` `instance-attribute`
```
response_tokens_limit:  | None = None

```

The maximum number of tokens allowed in responses from the model.
####  total_tokens_limit `class-attribute` `instance-attribute`
```
total_tokens_limit:  | None = None

```

The maximum number of tokens allowed in requests and responses combined.
####  has_token_limits
```
has_token_limits() -> 

```

Returns `True` if this instance places any limits on token counts.
If this returns `False`, the `check_tokens` method will never raise an error.
This is useful because if we have token limits, we need to check them after receiving each streamed message. If there are no limits, we can skip that processing in the streaming response iterator.
Source code in `pydantic_ai_slim/pydantic_ai/usage.py`
```
```
| ```
defhas_token_limits(self) -> bool:
"""Returns `True` if this instance places any limits on token counts.
  If this returns `False`, the `check_tokens` method will never raise an error.
  This is useful because if we have token limits, we need to check them after receiving each streamed message.
  If there are no limits, we can skip that processing in the streaming response iterator.
  """
  return any(
    limit is not None
    for limit in (self.request_tokens_limit, self.response_tokens_limit, self.total_tokens_limit)
  )

```
  
---|---  
####  check_before_request
```
check_before_request(usage: ) -> None

```

Raises a `UsageLimitExceeded` exception if the next request would exceed the request_limit.
Source code in `pydantic_ai_slim/pydantic_ai/usage.py`
```
```
| ```
defcheck_before_request(self, usage: Usage) -> None:
"""Raises a `UsageLimitExceeded` exception if the next request would exceed the request_limit."""
  request_limit = self.request_limit
  if request_limit is not None and usage.requests >= request_limit:
    raise UsageLimitExceeded(f'The next request would exceed the request_limit of {request_limit}')

```
  
---|---  
####  check_tokens
```
check_tokens(usage: ) -> None

```

Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits.
Source code in `pydantic_ai_slim/pydantic_ai/usage.py`
```
```
| ```
defcheck_tokens(self, usage: Usage) -> None:
"""Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
  request_tokens = usage.request_tokens or 0
  if self.request_tokens_limit is not None and request_tokens  self.request_tokens_limit:
    raise UsageLimitExceeded(
      f'Exceeded the request_tokens_limit of {self.request_tokens_limit} ({request_tokens=})'
    )
  response_tokens = usage.response_tokens or 0
  if self.response_tokens_limit is not None and response_tokens  self.response_tokens_limit:
    raise UsageLimitExceeded(
      f'Exceeded the response_tokens_limit of {self.response_tokens_limit} ({response_tokens=})'
    )
  total_tokens = usage.total_tokens or 0
  if self.total_tokens_limit is not None and total_tokens  self.total_tokens_limit:
    raise UsageLimitExceeded(f'Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})')

```
  
---|---

# Command Line Interface (CLI)
**PydanticAI** comes with a simple reference CLI application which you can use to interact with various LLMs directly from the command line. It provides a convenient way to chat with language models and quickly get answers right in the terminal.
We originally developed this CLI for our own use, but found ourselves using it so frequently that we decided to share it as part of the PydanticAI package.
We plan to continue adding new features, such as interaction with MCP servers, access to tools, and more.
## Installation
To use the CLI, you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `cli` optional group:
```
pipinstall"pydantic-ai[cli]"

```

```
uvadd"pydantic-ai[cli]"

```

To enable command-line argument autocompletion, run:
```
register-python-argcompletepai>>~/.bashrc# for bash
register-python-argcompletepai>>~/.zshrc# for zsh

```

## Usage
You'll need to set an environment variable depending on the provider you intend to use.
If using OpenAI, set the `OPENAI_API_KEY` environment variable:
```
exportOPENAI_API_KEY='your-api-key-here'

```

Then simply run:
```
$pai

```

This will start an interactive session where you can chat with the AI model. Special commands available in interactive mode:
  * `/exit`: Exit the session
  * `/markdown`: Show the last response in markdown format
  * `/multiline`: Toggle multiline input mode (use Ctrl+D to submit)


### Choose a model
You can specify which model to use with the `--model` flag:
```
$pai--model=openai:gpt-4"What's the capital of France?"

```


[ Skip to content ](https://ai.pydantic.dev/common_tools/#common-tools)
# Common Tools
PydanticAI ships with native tools that can be used to enhance your agent's capabilities.
## DuckDuckGo Search Tool
The DuckDuckGo search tool allows you to search the web for information. It is built on top of the [DuckDuckGo API](https://github.com/deedy5/duckduckgo_search).
### Installation
To use [`duckduckgo_search_tool`](https://ai.pydantic.dev/api/common_tools/#pydantic_ai.common_tools.duckduckgo.duckduckgo_search_tool), you need to install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `duckduckgo` optional group:
```
pipinstall"pydantic-ai-slim[duckduckgo]"

```

```
uvadd"pydantic-ai-slim[duckduckgo]"

```

### Usage
Here's an example of how you can use the DuckDuckGo search tool with an agent:
main.py```
frompydantic_aiimport Agent
frompydantic_ai.common_tools.duckduckgoimport duckduckgo_search_tool
agent = Agent(
  'openai:o3-mini',
  tools=[duckduckgo_search_tool()],
  system_prompt='Search DuckDuckGo for the given query and return the results.',
)
result = agent.run_sync(
  'Can you list the top five highest-grossing animated films of 2025?'
)
print(result.data)
"""
I looked into several sources on animated box‐office performance in 2025, and while detailed
rankings can shift as more money is tallied, multiple independent reports have already
highlighted a couple of record‐breaking shows. For example:
• Ne Zha 2 – News outlets (Variety, Wikipedia's "List of animated feature films of 2025", and others)
  have reported that this Chinese title not only became the highest‑grossing animated film of 2025
  but also broke records as the highest‑grossing non‑English animated film ever. One article noted
  its run exceeded US$1.7 billion.
• Inside Out 2 – According to data shared on Statista and in industry news, this Pixar sequel has been
  on pace to set new records (with some sources even noting it as the highest‑grossing animated film
  ever, as of January 2025).
Beyond those two, some entertainment trade sites (for example, a Just Jared article titled
"Top 10 Highest-Earning Animated Films at the Box Office Revealed") have begun listing a broader
top‑10. Although full consolidated figures can sometimes differ by source and are updated daily during
a box‑office run, many of the industry trackers have begun to single out five films as the biggest
earners so far in 2025.
Unfortunately, although multiple articles discuss the "top animated films" of 2025, there isn't yet a
single, universally accepted list with final numbers that names the complete top five. (Box‑office
rankings, especially mid‑year, can be fluid as films continue to add to their totals.)
Based on what several sources note so far, the two undisputed leaders are:
1. Ne Zha 2
2. Inside Out 2
The remaining top spots (3–5) are reported by some outlets in their "Top‑10 Animated Films"
lists for 2025 but the titles and order can vary depending on the source and the exact cut‑off
date of the data. For the most up‑to‑date and detailed ranking (including the 3rd, 4th, and 5th
highest‑grossing films), I recommend checking resources like:
• Wikipedia's "List of animated feature films of 2025" page
• Box‑office tracking sites (such as Box Office Mojo or The Numbers)
• Trade articles like the one on Just Jared
To summarize with what is clear from the current reporting:
1. Ne Zha 2
2. Inside Out 2
3–5. Other animated films (yet to be definitively finalized across all reporting outlets)
If you're looking for a final, consensus list of the top five, it may be best to wait until
the 2025 year‑end box‑office tallies are in or to consult a regularly updated entertainment industry source.
Would you like help finding a current source or additional details on where to look for the complete updated list?
"""

```

## Tavily Search Tool
Info
Tavily is a paid service, but they have free credits to explore their product.
You need to [sign up for an account](https://app.tavily.com/home) and get an API key to use the Tavily search tool.
The Tavily search tool allows you to search the web for information. It is built on top of the [Tavily API](https://tavily.com/).
### Installation
To use [`tavily_search_tool`](https://ai.pydantic.dev/api/common_tools/#pydantic_ai.common_tools.tavily.tavily_search_tool), you need to install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `tavily` optional group:
```
pipinstall"pydantic-ai-slim[tavily]"

```

```
uvadd"pydantic-ai-slim[tavily]"

```

### Usage
Here's an example of how you can use the Tavily search tool with an agent:
main.py```
importos
frompydantic_ai.agentimport Agent
frompydantic_ai.common_tools.tavilyimport tavily_search_tool
api_key = os.getenv('TAVILY_API_KEY')
assert api_key is not None

agent = Agent(
  'openai:o3-mini',
  tools=[tavily_search_tool(api_key)],
  system_prompt='Search Tavily for the given query and return the results.',
)
result = agent.run_sync('Tell me the top news in the GenAI world, give me links.')
print(result.data)
"""
Here are some of the top recent news articles related to GenAI:
1. How CLEAR users can improve risk analysis with GenAI – Thomson Reuters
  Read more: https://legal.thomsonreuters.com/blog/how-clear-users-can-improve-risk-analysis-with-genai/
  (This article discusses how CLEAR's new GenAI-powered tool streamlines risk analysis by quickly summarizing key information from various public data sources.)
2. TELUS Digital Survey Reveals Enterprise Employees Are Entering Sensitive Data Into AI Assistants More Than You Think – FT.com
  Read more: https://markets.ft.com/data/announce/detail?dockey=600-202502260645BIZWIRE_USPRX____20250226_BW490609-1
  (This news piece highlights findings from a TELUS Digital survey showing that many enterprise employees use public GenAI tools and sometimes even enter sensitive data.)
3. The Essential Guide to Generative AI – Virtualization Review
  Read more: https://virtualizationreview.com/Whitepapers/2025/02/SNOWFLAKE-The-Essential-Guide-to-Generative-AI.aspx
  (This guide provides insights into how GenAI is revolutionizing enterprise strategies and productivity, with input from industry leaders.)
Feel free to click on the links to dive deeper into each story!
"""

```


# Contributing
We'd love you to contribute to PydanticAI!
## Installation and Setup
Clone your fork and cd into the repo directory
```
gitclonegit@github.com:<yourusername>/pydantic-ai.git
cdpydantic-ai

```

Install `uv` (version 0.4.30 or later) and `pre-commit`
We use pipx here, for other options see:
  * [`uv` install docs](https://docs.astral.sh/uv/getting-started/installation/)
  * [`pre-commit` install docs](https://pre-commit.com/#install)


To get `pipx` itself, see [these docs](https://pypa.github.io/pipx/)
```
pipxinstalluvpre-commit

```

Install `pydantic-ai`, all dependencies and pre-commit hooks
```
makeinstall

```

## Running Tests etc.
We use `make` to manage most commands you'll need to run.
For details on available commands, run:
```
makehelp

```

To run code formatting, linting, static type checks, and tests with coverage report generation, run:
```
make

```

## Documentation Changes
To run the documentation page locally, run:
```
uvrunmkdocsserve

```

## Rules for adding new models to PydanticAI
To avoid an excessive workload for the maintainers of PydanticAI, we can't accept all model contributions, so we're setting the following rules for when we'll accept new models and when we won't. This should hopefully reduce the chances of disappointment and wasted work.
  * To add a new model with an extra dependency, that dependency needs > 500k monthly downloads from PyPI consistently over 3 months or more
  * To add a new model which uses another models logic internally and has no extra dependencies, that model's GitHub org needs > 20k stars in total
  * For any other model that's just a custom URL and API key, we're happy to add a one-paragraph description with a link and instructions on the URL to use
  * For any other model that requires more logic, we recommend you release your own Python package `pydantic-ai-xxx`, which depends on [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) and implements a model that inherits from our [`Model`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model) ABC


If you're unsure about adding a model, please [create an issue](https://github.com/pydantic/pydantic-ai/issues).

[ Skip to content ](https://ai.pydantic.dev/dependencies/#dependencies)
# Dependencies
PydanticAI uses a dependency injection system to provide data and services to your agent's [system prompts](https://ai.pydantic.dev/agents/#system-prompts), [tools](https://ai.pydantic.dev/tools/) and [result validators](https://ai.pydantic.dev/results/#result-validators-functions).
Matching PydanticAI's design philosophy, our dependency system tries to use existing best practice in Python development rather than inventing esoteric "magic", this should make dependencies type-safe, understandable easier to test and ultimately easier to deploy in production.
## Defining Dependencies
Dependencies can be any python type. While in simple cases you might be able to pass a single object as a dependency (e.g. an HTTP connection), [dataclasses](https://docs.python.org/3/library/dataclasses.html#module-dataclasses) are generally a convenient container when your dependencies included multiple objects.
Here's an example of defining an agent that requires dependencies.
(**Note:** dependencies aren't actually used in this example, see [Accessing Dependencies](https://ai.pydantic.dev/dependencies/#accessing-dependencies) below)
unused_dependencies.py```
fromdataclassesimport dataclass
importhttpx
frompydantic_aiimport Agent

@dataclass
classMyDeps: 
  api_key: str
  http_client: httpx.AsyncClient

agent = Agent(
  'openai:gpt-4o',
  deps_type=MyDeps, 
)

async defmain():
  async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run(
      'Tell me a joke.',
      deps=deps, 
    )
    print(result.data)
    #> Did you hear about the toothpaste scandal? They called it Colgate.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
## Accessing Dependencies
Dependencies are accessed through the [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) type, this should be the first parameter of system prompt functions etc.
system_prompt_dependencies.py```
fromdataclassesimport dataclass
importhttpx
frompydantic_aiimport Agent, RunContext

@dataclass
classMyDeps:
  api_key: str
  http_client: httpx.AsyncClient

agent = Agent(
  'openai:gpt-4o',
  deps_type=MyDeps,
)

@agent.system_prompt 
async defget_system_prompt(ctx: RunContext[MyDeps]) -> str: 
  response = await ctx.deps.http_client.get( 
    'https://example.com',
    headers={'Authorization': f'Bearer {ctx.deps.api_key}'}, 
  response.raise_for_status()
  return f'Prompt: {response.text}'

async defmain():
  async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run('Tell me a joke.', deps=deps)
    print(result.data)
    #> Did you hear about the toothpaste scandal? They called it Colgate.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
### Asynchronous vs. Synchronous dependencies
[System prompt functions](https://ai.pydantic.dev/agents/#system-prompts), [function tools](https://ai.pydantic.dev/tools/) and [result validators](https://ai.pydantic.dev/results/#result-validators-functions) are all run in the async context of an agent run.
If these functions are not coroutines (e.g. `async def`) they are called with [`run_in_executor`](https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor) in a thread pool, it's therefore marginally preferable to use `async` methods where dependencies perform IO, although synchronous dependencies should work fine too.
`run` vs. `run_sync` and Asynchronous vs. Synchronous dependencies
Whether you use synchronous or asynchronous dependencies, is completely independent of whether you use `run` or `run_sync` — `run_sync` is just a wrapper around `run` and agents are always run in an async context.
Here's the same example as above, but with a synchronous dependency:
sync_dependencies.py```
fromdataclassesimport dataclass
importhttpx
frompydantic_aiimport Agent, RunContext

@dataclass
classMyDeps:
  api_key: str
  http_client: httpx.Client 

agent = Agent(
  'openai:gpt-4o',
  deps_type=MyDeps,
)

@agent.system_prompt
defget_system_prompt(ctx: RunContext[MyDeps]) -> str: 
  response = ctx.deps.http_client.get(
    'https://example.com', headers={'Authorization': f'Bearer {ctx.deps.api_key}'}
  )
  response.raise_for_status()
  return f'Prompt: {response.text}'

async defmain():
  deps = MyDeps('foobar', httpx.Client())
  result = await agent.run(
    'Tell me a joke.',
    deps=deps,
  )
  print(result.data)
  #> Did you hear about the toothpaste scandal? They called it Colgate.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
## Full Example
As well as system prompts, dependencies can be used in [tools](https://ai.pydantic.dev/tools/) and [result validators](https://ai.pydantic.dev/results/#result-validators-functions).
full_example.py```
fromdataclassesimport dataclass
importhttpx
frompydantic_aiimport Agent, ModelRetry, RunContext

@dataclass
classMyDeps:
  api_key: str
  http_client: httpx.AsyncClient

agent = Agent(
  'openai:gpt-4o',
  deps_type=MyDeps,
)

@agent.system_prompt
async defget_system_prompt(ctx: RunContext[MyDeps]) -> str:
  response = await ctx.deps.http_client.get('https://example.com')
  response.raise_for_status()
  return f'Prompt: {response.text}'

@agent.tool 
async defget_joke_material(ctx: RunContext[MyDeps], subject: str) -> str:
  response = await ctx.deps.http_client.get(
    'https://example.com#jokes',
    params={'subject': subject},
    headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
  response.raise_for_status()
  return response.text

@agent.result_validator 
async defvalidate_result(ctx: RunContext[MyDeps], final_response: str) -> str:
  response = await ctx.deps.http_client.post(
    'https://example.com#validate',
    headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    params={'query': final_response},
  if response.status_code == 400:
    raise ModelRetry(f'invalid response: {response.text}')
  response.raise_for_status()
  return final_response

async defmain():
  async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run('Tell me a joke.', deps=deps)
    print(result.data)
    #> Did you hear about the toothpaste scandal? They called it Colgate.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
## Overriding Dependencies
When testing agents, it's useful to be able to customise dependencies.
While this can sometimes be done by calling the agent directly within unit tests, we can also override dependencies while calling application code which in turn calls the agent.
This is done via the [`override`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.override) method on the agent.
joke_app.py```
fromdataclassesimport dataclass
importhttpx
frompydantic_aiimport Agent, RunContext

@dataclass
classMyDeps:
  api_key: str
  http_client: httpx.AsyncClient
  async defsystem_prompt_factory(self) -> str: 
    response = await self.http_client.get('https://example.com')
    response.raise_for_status()
    return f'Prompt: {response.text}'

joke_agent = Agent('openai:gpt-4o', deps_type=MyDeps)

@joke_agent.system_prompt
async defget_system_prompt(ctx: RunContext[MyDeps]) -> str:
  return await ctx.deps.system_prompt_factory() 

async defapplication_code(prompt: str) -> str: 
  ...
  ...
  # now deep within application code we call our agent
  async with httpx.AsyncClient() as client:
    app_deps = MyDeps('foobar', client)
    result = await joke_agent.run(prompt, deps=app_deps) 
  return result.data

```

_(This example is complete, it can be run "as is")_
test_joke_app.py```
fromjoke_appimport MyDeps, application_code, joke_agent

classTestMyDeps(MyDeps): 
  async defsystem_prompt_factory(self) -> str:
    return 'test prompt'

async deftest_application_code():
  test_deps = TestMyDeps('test_key', None) 
  with joke_agent.override(deps=test_deps): 
    joke = await application_code('Tell me a joke.') 
  assert joke.startswith('Did you hear about the toothpaste scandal?')

```

## Examples
The following examples demonstrate how to use dependencies in PydanticAI:
  * [SQL Generation](https://ai.pydantic.dev/examples/sql-gen/)



[ Skip to content ](https://ai.pydantic.dev/examples/#examples)
# Examples
Examples of how to use PydanticAI and what it can do.
## Usage
These examples are distributed with `pydantic-ai` so you can run them either by cloning the [pydantic-ai repo](https://github.com/pydantic/pydantic-ai) or by simply installing `pydantic-ai` from PyPI with `pip` or `uv`.
### Installing required dependencies
Either way you'll need to install extra dependencies to run some examples, you just need to install the `examples` optional dependency group.
If you've installed `pydantic-ai` via pip/uv, you can install the extra dependencies with:
```
pipinstall"pydantic-ai[examples]"

```

```
uvadd"pydantic-ai[examples]"

```

If you clone the repo, you should instead use `uv sync --extra examples` to install extra dependencies.
### Setting model environment variables
These examples will need you to set up authentication with one or more of the LLMs, see the [model configuration](https://ai.pydantic.dev/models/) docs for details on how to do this.
TL;DR: in most cases you'll need to set one of the following environment variables:
[Google Gemini](https://ai.pydantic.dev/examples/#__tabbed_2_2)
```
exportOPENAI_API_KEY=your-api-key

```

```
exportGEMINI_API_KEY=your-api-key

```

### Running Examples
To run the examples (this will work whether you installed `pydantic_ai`, or cloned the repo), run:
```
python-mpydantic_ai_examples.<example_module_name>

```

```
uvrun-mpydantic_ai_examples.<example_module_name>

```

For examples, to run the very simple [`pydantic_model`](https://ai.pydantic.dev/examples/pydantic-model/) example:
```
python-mpydantic_ai_examples.pydantic_model

```

```
uvrun-mpydantic_ai_examples.pydantic_model

```

If you like one-liners and you're using uv, you can run a pydantic-ai example with zero setup:
```
OPENAI_API_KEY='your-api-key'\
uvrun--with"pydantic-ai[examples]"\
-mpydantic_ai_examples.pydantic_model

```

You'll probably want to edit examples in addition to just running them. You can copy the examples to a new directory with:
```
python-mpydantic_ai_examples--copy-toexamples/

```

```
uvrun-mpydantic_ai_examples--copy-toexamples/

```


# Bank support
Small but complete example of using PydanticAI to build a support agent for a bank.
Demonstrates:
  * [dynamic system prompt](https://ai.pydantic.dev/agents/#system-prompts)
  * [structured `result_type`](https://ai.pydantic.dev/results/#structured-result-validation)


## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.bank_support

```

```
uvrun-mpydantic_ai_examples.bank_support

```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)
## Example Code
bank_support.py```
fromdataclassesimport dataclass
frompydanticimport BaseModel, Field
frompydantic_aiimport Agent, RunContext

classDatabaseConn:
"""This is a fake database for example purposes.
  In reality, you'd be connecting to an external database
  (e.g. PostgreSQL) to get information about customers.
  """
  @classmethod
  async defcustomer_name(cls, *, id: int) -> str | None:
    if id == 123:
      return 'John'
  @classmethod
  async defcustomer_balance(cls, *, id: int, include_pending: bool) -> float:
    if id == 123 and include_pending:
      return 123.45
    else:
      raise ValueError('Customer not found')

@dataclass
classSupportDependencies:
  customer_id: int
  db: DatabaseConn

classSupportResult(BaseModel):
  support_advice: str = Field(description='Advice returned to the customer')
  block_card: bool = Field(description='Whether to block their card or not')
  risk: int = Field(description='Risk level of query', ge=0, le=10)

support_agent = Agent(
  'openai:gpt-4o',
  deps_type=SupportDependencies,
  result_type=SupportResult,
  system_prompt=(
    'You are a support agent in our bank, give the '
    'customer support and judge the risk level of their query. '
    "Reply using the customer's name."
  ),
)

@support_agent.system_prompt
async defadd_customer_name(ctx: RunContext[SupportDependencies]) -> str:
  customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
  return f"The customer's name is {customer_name!r}"

@support_agent.tool
async defcustomer_balance(
  ctx: RunContext[SupportDependencies], include_pending: bool
) -> str:
"""Returns the customer's current account balance."""
  balance = await ctx.deps.db.customer_balance(
    id=ctx.deps.customer_id,
    include_pending=include_pending,
  )
  return f'${balance:.2f}'

if __name__ == '__main__':
  deps = SupportDependencies(customer_id=123, db=DatabaseConn())
  result = support_agent.run_sync('What is my balance?', deps=deps)
  print(result.data)
"""
  support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
  """
  result = support_agent.run_sync('I just lost my card!', deps=deps)
  print(result.data)
"""
  support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
  """

```


# Chat App with FastAPI
Simple chat app example build with FastAPI.
Demonstrates:
  * [reusing chat history](https://ai.pydantic.dev/message-history/)
  * [serializing messages](https://ai.pydantic.dev/message-history/#accessing-messages-from-results)
  * [streaming responses](https://ai.pydantic.dev/results/#streamed-results)


This demonstrates storing chat history between requests and using it to give the model context for new responses.
Most of the complex logic here is between `chat_app.py` which streams the response to the browser, and `chat_app.ts` which renders messages in the browser.
## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.chat_app

```

```
uvrun-mpydantic_ai_examples.chat_app

```

Then open the app at [localhost:8000](http://localhost:8000).
## Example Code
Python code that runs the chat app:
chat_app.py```
from__future__import annotations as _annotations
importasyncio
importjson
importsqlite3
fromcollections.abcimport AsyncIterator
fromconcurrent.futures.threadimport ThreadPoolExecutor
fromcontextlibimport asynccontextmanager
fromdataclassesimport dataclass
fromdatetimeimport datetime, timezone
fromfunctoolsimport partial
frompathlibimport Path
fromtypingimport Annotated, Any, Callable, Literal, TypeVar
importfastapi
importlogfire
fromfastapiimport Depends, Request
fromfastapi.responsesimport FileResponse, Response, StreamingResponse
fromtyping_extensionsimport LiteralString, ParamSpec, TypedDict
frompydantic_aiimport Agent
frompydantic_ai.exceptionsimport UnexpectedModelBehavior
frompydantic_ai.messagesimport (
  ModelMessage,
  ModelMessagesTypeAdapter,
  ModelRequest,
  ModelResponse,
  TextPart,
  UserPromptPart,
)
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
agent = Agent('openai:gpt-4o', instrument=True)
THIS_DIR = Path(__file__).parent

@asynccontextmanager
async deflifespan(_app: fastapi.FastAPI):
  async with Database.connect() as db:
    yield {'db': db}

app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)

@app.get('/')
async defindex() -> FileResponse:
  return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')

@app.get('/chat_app.ts')
async defmain_ts() -> FileResponse:
"""Get the raw typescript code, it's compiled in the browser, forgive me."""
  return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')

async defget_db(request: Request) -> Database:
  return request.state.db

@app.get('/chat/')
async defget_chat(database: Database = Depends(get_db)) -> Response:
  msgs = await database.get_messages()
  return Response(
    b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
    media_type='text/plain',
  )

classChatMessage(TypedDict):
"""Format of messages sent to the browser."""
  role: Literal['user', 'model']
  timestamp: str
  content: str

defto_chat_message(m: ModelMessage) -> ChatMessage:
  first_part = m.parts[0]
  if isinstance(m, ModelRequest):
    if isinstance(first_part, UserPromptPart):
      assert isinstance(first_part.content, str)
      return {
        'role': 'user',
        'timestamp': first_part.timestamp.isoformat(),
        'content': first_part.content,
      }
  elif isinstance(m, ModelResponse):
    if isinstance(first_part, TextPart):
      return {
        'role': 'model',
        'timestamp': m.timestamp.isoformat(),
        'content': first_part.content,
      }
  raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')

@app.post('/chat/')
async defpost_chat(
  prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
  async defstream_messages():
"""Streams new line delimited JSON `Message`s to the client."""
    # stream the user prompt so that can be displayed straight away
    yield (
      json.dumps(
        {
          'role': 'user',
          'timestamp': datetime.now(tz=timezone.utc).isoformat(),
          'content': prompt,
        }
      ).encode('utf-8')
      + b'\n'
    )
    # get the chat history so far to pass as context to the agent
    messages = await database.get_messages()
    # run the agent with the user prompt and the chat history
    async with agent.run_stream(prompt, message_history=messages) as result:
      async for text in result.stream(debounce_by=0.01):
        # text here is a `str` and the frontend wants
        # JSON encoded ModelResponse, so we create one
        m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
        yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'
    # add new messages (e.g. the user prompt and the agent response in this case) to the database
    await database.add_messages(result.new_messages_json())
  return StreamingResponse(stream_messages(), media_type='text/plain')

P = ParamSpec('P')
R = TypeVar('R')

@dataclass
classDatabase:
"""Rudimentary database to store chat messages in SQLite.
  The SQLite standard library package is synchronous, so we
  use a thread pool executor to run queries asynchronously.
  """
  con: sqlite3.Connection
  _loop: asyncio.AbstractEventLoop
  _executor: ThreadPoolExecutor
  @classmethod
  @asynccontextmanager
  async defconnect(
    cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
  ) -> AsyncIterator[Database]:
    with logfire.span('connect to DB'):
      loop = asyncio.get_event_loop()
      executor = ThreadPoolExecutor(max_workers=1)
      con = await loop.run_in_executor(executor, cls._connect, file)
      slf = cls(con, loop, executor)
    try:
      yield slf
    finally:
      await slf._asyncify(con.close)
  @staticmethod
  def_connect(file: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(file))
    con = logfire.instrument_sqlite3(con)
    cur = con.cursor()
    cur.execute(
      'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
    )
    con.commit()
    return con
  async defadd_messages(self, messages: bytes):
    await self._asyncify(
      self._execute,
      'INSERT INTO messages (message_list) VALUES (?);',
      messages,
      commit=True,
    )
    await self._asyncify(self.con.commit)
  async defget_messages(self) -> list[ModelMessage]:
    c = await self._asyncify(
      self._execute, 'SELECT message_list FROM messages order by id'
    )
    rows = await self._asyncify(c.fetchall)
    messages: list[ModelMessage] = []
    for row in rows:
      messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
    return messages
  def_execute(
    self, sql: LiteralString, *args: Any, commit: bool = False
  ) -> sqlite3.Cursor:
    cur = self.con.cursor()
    cur.execute(sql, args)
    if commit:
      self.con.commit()
    return cur
  async def_asyncify(
    self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
  ) -> R:
    return await self._loop.run_in_executor( # type: ignore
      self._executor,
      partial(func, **kwargs),
      *args, # type: ignore
    )

if __name__ == '__main__':
  importuvicorn
  uvicorn.run(
    'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
  )

```

Simple HTML page to render the app:
chat_app.html```
<!DOCTYPE html>
html lang="en"
head
 meta charset="UTF-8"
 meta name="viewport" content="width=device-width, initial-scale=1.0"
 titleChat App</title
 link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet"
 style
main{
max-width:700px;
}
#conversation.user::before{
content:'You asked: ';
font-weight:bold;
display:block;
}
#conversation.model::before{
content:'AI Response: ';
font-weight:bold;
display:block;
}
#spinner{
opacity:0;
transition:opacity500msease-in;
width:30px;
height:30px;
border:3pxsolid#222;
border-bottom-color:transparent;
border-radius:50%;
animation:rotation1slinearinfinite;
}
@keyframesrotation{
0%{transform:rotate(0deg);}
100%{transform:rotate(360deg);}
}
#spinner.active{
opacity:1;
}
</style
</head
body
 main class="border rounded mx-auto my-5 p-4"
  h1Chat App</h1
  pAsk me anything...</p
  div id="conversation" class="px-2"></div
  div class="d-flex justify-content-center mb-3"
   div id="spinner"></div
  </div
  form method="post"
   input id="prompt-input" name="prompt" class="form-control"/>
   div class="d-flex justify-content-end"
    button class="btn btn-primary mt-2"Send</button
   </div
  </form
  div id="error" class="d-none text-danger"
   Error occurred, check the browser developer console for more information.
  </div
 </main
</body
</html
script src="https://cdnjs.cloudflare.com/ajax/libs/typescript/5.6.3/typescript.min.js" crossorigin="anonymous" referrerpolicy="no-referrer"></script
script type="module"
// to let me write TypeScript, without adding the burden of npm we do a dirty, non-production-ready hack
// and transpile the TypeScript code in the browser
// this is (arguably) A neat demo trick, but not suitable for production!
asyncfunctionloadTs(){
constresponse=awaitfetch('/chat_app.ts');
consttsCode=awaitresponse.text();
constjsCode=window.ts.transpile(tsCode,{target:"es2015"});
letscript=document.createElement('script');
script.type='module';
script.text=jsCode;
document.body.appendChild(script);
}
loadTs().catch((e)=>{
console.error(e);
document.getElementById('error').classList.remove('d-none');
document.getElementById('spinner').classList.remove('active');
});
</script

```

TypeScript to handle rendering the messages, to keep this simple (and at the risk of offending frontend developers) the typescript code is passed to the browser as plain text and transpiled in the browser.
chat_app.ts```
// BIG FAT WARNING: to avoid the complexity of npm, this typescript is compiled in the browser
// there's currently no static type checking
import{marked}from'https://cdnjs.cloudflare.com/ajax/libs/marked/15.0.0/lib/marked.esm.js'
constconvElement=document.getElementById('conversation')
constpromptInput=document.getElementById('prompt-input')asHTMLInputElement
constspinner=document.getElementById('spinner')
// stream the response and render messages as each chunk is received
// data is sent as newline-delimited JSON
asyncfunctiononFetchResponse(response:Response):Promisevoid{
lettext=''
letdecoder=newTextDecoder()
if(response.ok){
constreader=response.body.getReader()
while(true){
const{done,value}=awaitreader.read()
if(done){
break
}
text+=decoder.decode(value)
addMessages(text)
spinner.classList.remove('active')
}
addMessages(text)
promptInput.disabled=false
promptInput.focus()
}else{
consttext=awaitresponse.text()
console.error(`Unexpected response: ${response.status}`,{response,text})
thrownewError(`Unexpected response: ${response.status}`)
}
}
// The format of messages, this matches pydantic-ai both for brevity and understanding
// in production, you might not want to keep this format all the way to the frontend
interfaceMessage{
role:string
content:string
timestamp:string
}
// take raw response text and render messages into the `#conversation` element
// Message timestamp is assumed to be a unique identifier of a message, and is used to deduplicate
// hence you can send data about the same message multiple times, and it will be updated
// instead of creating a new message elements
functionaddMessages(responseText:string){
constlines=responseText.split('\n')
constmessages:Message[]=lines.filter(line=>line.length1).map(j=>JSON.parse(j))
for(constmessageofmessages){
// we use the timestamp as a crude element id
const{timestamp,role,content}=message
constid=`msg-${timestamp}`
letmsgDiv=document.getElementById(id)
if(!msgDiv){
msgDiv=document.createElement('div')
msgDiv.id=id
msgDiv.title=`${role} at ${timestamp}`
msgDiv.classList.add('border-top','pt-2',role)
convElement.appendChild(msgDiv)
}
msgDiv.innerHTML=marked.parse(content)
}
window.scrollTo({top:document.body.scrollHeight,behavior:'smooth'})
}
functiononError(error:any){
console.error(error)
document.getElementById('error').classList.remove('d-none')
document.getElementById('spinner').classList.remove('active')
}
asyncfunctiononSubmit(e:SubmitEvent):Promisevoid{
e.preventDefault()
spinner.classList.add('active')
constbody=newFormData(e.targetasHTMLFormElement)
promptInput.value=''
promptInput.disabled=true
constresponse=awaitfetch('/chat/',{method:'POST',body})
awaitonFetchResponse(response)
}
// call onSubmit when the form is submitted (e.g. user clicks the send button or hits Enter)
document.querySelector('form').addEventListener('submit',(e)=>onSubmit(e).catch(onError))
// load messages on page load
fetch('/chat/').then(onFetchResponse).catch(onError)

```


# Flight booking
Example of a multi-agent flow where one agent delegates work to another, then hands off control to a third agent.
Demonstrates:
  * [programmatic agent hand-off](https://ai.pydantic.dev/multi-agent-applications/#programmatic-agent-hand-off)


In this scenario, a group of agents work together to find the best flight for a user.
The control flow for this example can be summarised as follows:
## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.flight_booking

```

```
uvrun-mpydantic_ai_examples.flight_booking

```

## Example Code
flight_booking.py```
importdatetime
fromdataclassesimport dataclass
fromtypingimport Literal
importlogfire
frompydanticimport BaseModel, Field
fromrich.promptimport Prompt
frompydantic_aiimport Agent, ModelRetry, RunContext
frompydantic_ai.messagesimport ModelMessage
frompydantic_ai.usageimport Usage, UsageLimits
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

classFlightDetails(BaseModel):
"""Details of the most suitable flight."""
  flight_number: str
  price: int
  origin: str = Field(description='Three-letter airport code')
  destination: str = Field(description='Three-letter airport code')
  date: datetime.date

classNoFlightFound(BaseModel):
"""When no valid flight is found."""

@dataclass
classDeps:
  web_page_text: str
  req_origin: str
  req_destination: str
  req_date: datetime.date

# This agent is responsible for controlling the flow of the conversation.
search_agent = Agent[Deps, FlightDetails | NoFlightFound](
  'openai:gpt-4o',
  result_type=FlightDetails | NoFlightFound, # type: ignore
  retries=4,
  system_prompt=(
    'Your job is to find the cheapest flight for the user on the given date. '
  ),
  instrument=True,
)

# This agent is responsible for extracting flight details from web page text.
extraction_agent = Agent(
  'openai:gpt-4o',
  result_type=list[FlightDetails],
  system_prompt='Extract all the flight details from the given text.',
)

@search_agent.tool
async defextract_flights(ctx: RunContext[Deps]) -> list[FlightDetails]:
"""Get details of all flights."""
  # we pass the usage to the search agent so requests within this agent are counted
  result = await extraction_agent.run(ctx.deps.web_page_text, usage=ctx.usage)
  logfire.info('found {flight_count} flights', flight_count=len(result.data))
  return result.data

@search_agent.result_validator
async defvalidate_result(
  ctx: RunContext[Deps], result: FlightDetails | NoFlightFound
) -> FlightDetails | NoFlightFound:
"""Procedural validation that the flight meets the constraints."""
  if isinstance(result, NoFlightFound):
    return result
  errors: list[str] = []
  if result.origin != ctx.deps.req_origin:
    errors.append(
      f'Flight should have origin {ctx.deps.req_origin}, not {result.origin}'
    )
  if result.destination != ctx.deps.req_destination:
    errors.append(
      f'Flight should have destination {ctx.deps.req_destination}, not {result.destination}'
    )
  if result.date != ctx.deps.req_date:
    errors.append(f'Flight should be on {ctx.deps.req_date}, not {result.date}')
  if errors:
    raise ModelRetry('\n'.join(errors))
  else:
    return result

classSeatPreference(BaseModel):
  row: int = Field(ge=1, le=30)
  seat: Literal['A', 'B', 'C', 'D', 'E', 'F']

classFailed(BaseModel):
"""Unable to extract a seat selection."""

# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[None, SeatPreference | Failed](
  'openai:gpt-4o',
  result_type=SeatPreference | Failed, # type: ignore
  system_prompt=(
    "Extract the user's seat preference. "
    'Seats A and F are window seats. '
    'Row 1 is the front row and has extra leg room. '
    'Rows 14, and 20 also have extra leg room. '
  ),
)

# in reality this would be downloaded from a booking site,
# potentially using another agent to navigate the site
flights_web_page = """
1. Flight SFO-AK123
- Price: $350
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025
2. Flight SFO-AK456
- Price: $370
- Origin: San Francisco International Airport (SFO)
- Destination: Fairbanks International Airport (FAI)
- Date: January 10, 2025
3. Flight SFO-AK789
- Price: $400
- Origin: San Francisco International Airport (SFO)
- Destination: Juneau International Airport (JNU)
- Date: January 20, 2025
4. Flight NYC-LA101
- Price: $250
- Origin: San Francisco International Airport (SFO)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 10, 2025
5. Flight CHI-MIA202
- Price: $200
- Origin: Chicago O'Hare International Airport (ORD)
- Destination: Miami International Airport (MIA)
- Date: January 12, 2025
6. Flight BOS-SEA303
- Price: $120
- Origin: Boston Logan International Airport (BOS)
- Destination: Ted Stevens Anchorage International Airport (ANC)
- Date: January 12, 2025
7. Flight DFW-DEN404
- Price: $150
- Origin: Dallas/Fort Worth International Airport (DFW)
- Destination: Denver International Airport (DEN)
- Date: January 10, 2025
8. Flight ATL-HOU505
- Price: $180
- Origin: Hartsfield-Jackson Atlanta International Airport (ATL)
- Destination: George Bush Intercontinental Airport (IAH)
- Date: January 10, 2025
"""
# restrict how many requests this app can make to the LLM
usage_limits = UsageLimits(request_limit=15)

async defmain():
  deps = Deps(
    web_page_text=flights_web_page,
    req_origin='SFO',
    req_destination='ANC',
    req_date=datetime.date(2025, 1, 10),
  )
  message_history: list[ModelMessage] | None = None
  usage: Usage = Usage()
  # run the agent until a satisfactory flight is found
  while True:
    result = await search_agent.run(
      f'Find me a flight from {deps.req_origin} to {deps.req_destination} on {deps.req_date}',
      deps=deps,
      usage=usage,
      message_history=message_history,
      usage_limits=usage_limits,
    )
    if isinstance(result.data, NoFlightFound):
      print('No flight found')
      break
    else:
      flight = result.data
      print(f'Flight found: {flight}')
      answer = Prompt.ask(
        'Do you want to buy this flight, or keep searching? (buy/*search)',
        choices=['buy', 'search', ''],
        show_choices=False,
      )
      if answer == 'buy':
        seat = await find_seat(usage)
        await buy_tickets(flight, seat)
        break
      else:
        message_history = result.all_messages(
          result_tool_return_content='Please suggest another flight'
        )

async deffind_seat(usage: Usage) -> SeatPreference:
  message_history: list[ModelMessage] | None = None
  while True:
    answer = Prompt.ask('What seat would you like?')
    result = await seat_preference_agent.run(
      answer,
      message_history=message_history,
      usage=usage,
      usage_limits=usage_limits,
    )
    if isinstance(result.data, SeatPreference):
      return result.data
    else:
      print('Could not understand seat preference. Please try again.')
      message_history = result.all_messages()

async defbuy_tickets(flight_details: FlightDetails, seat: SeatPreference):
  print(f'Purchasing flight {flight_details=!r}{seat=!r}...')

if __name__ == '__main__':
  importasyncio
  asyncio.run(main())

```


[ Skip to content ](https://ai.pydantic.dev/examples/pydantic-model/#pydantic-model)
# Pydantic Model
Simple example of using PydanticAI to construct a Pydantic model from a text input.
Demonstrates:
  * [structured `result_type`](https://ai.pydantic.dev/results/#structured-result-validation)


## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.pydantic_model

```

```
uvrun-mpydantic_ai_examples.pydantic_model

```

This examples uses `openai:gpt-4o` by default, but it works well with other models, e.g. you can run it with Gemini using:
```
PYDANTIC_AI_MODEL=gemini-1.5-propython-mpydantic_ai_examples.pydantic_model

```

```
PYDANTIC_AI_MODEL=gemini-1.5-prouvrun-mpydantic_ai_examples.pydantic_model

```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)
## Example Code
pydantic_model.py```
importos
fromtypingimport cast
importlogfire
frompydanticimport BaseModel
frompydantic_aiimport Agent
frompydantic_ai.modelsimport KnownModelName
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

classMyModel(BaseModel):
  city: str
  country: str

model = cast(KnownModelName, os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-4o'))
print(f'Using model: {model}')
agent = Agent(model, result_type=MyModel, instrument=True)
if __name__ == '__main__':
  result = agent.run_sync('The windy city in the US of A.')
  print(result.data)
  print(result.usage())

```


[ Skip to content ](https://ai.pydantic.dev/examples/question-graph/#question-graph)
# Question Graph
Example of a graph for asking and evaluating questions.
Demonstrates:


## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.question_graph

```

```
uvrun-mpydantic_ai_examples.question_graph

```

## Example Code
question_graph.py```
from__future__import annotations as _annotations
fromdataclassesimport dataclass, field
frompathlibimport Path
importlogfire
fromgroqimport BaseModel
frompydantic_graphimport (
  BaseNode,
  End,
  Graph,
  GraphRunContext,
)
frompydantic_graph.persistence.fileimport FileStatePersistence
frompydantic_aiimport Agent
frompydantic_ai.format_as_xmlimport format_as_xml
frompydantic_ai.messagesimport ModelMessage
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
ask_agent = Agent('openai:gpt-4o', result_type=str, instrument=True)

@dataclass
classQuestionState:
  question: str | None = None
  ask_agent_messages: list[ModelMessage] = field(default_factory=list)
  evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)

@dataclass
classAsk(BaseNode[QuestionState]):
  async defrun(self, ctx: GraphRunContext[QuestionState]) -> Answer:
    result = await ask_agent.run(
      'Ask a simple question with a single correct answer.',
      message_history=ctx.state.ask_agent_messages,
    )
    ctx.state.ask_agent_messages += result.all_messages()
    ctx.state.question = result.data
    return Answer(result.data)

@dataclass
classAnswer(BaseNode[QuestionState]):
  question: str
  async defrun(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
    answer = input(f'{self.question}: ')
    return Evaluate(answer)

classEvaluationResult(BaseModel, use_attribute_docstrings=True):
  correct: bool
"""Whether the answer is correct."""
  comment: str
"""Comment on the answer, reprimand the user if the answer is wrong."""

evaluate_agent = Agent(
  'openai:gpt-4o',
  result_type=EvaluationResult,
  system_prompt='Given a question and answer, evaluate if the answer is correct.',
)

@dataclass
classEvaluate(BaseNode[QuestionState, None, str]):
  answer: str
  async defrun(
    self,
    ctx: GraphRunContext[QuestionState],
  ) -> End[str] | Reprimand:
    assert ctx.state.question is not None
    result = await evaluate_agent.run(
      format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
      message_history=ctx.state.evaluate_agent_messages,
    )
    ctx.state.evaluate_agent_messages += result.all_messages()
    if result.data.correct:
      return End(result.data.comment)
    else:
      return Reprimand(result.data.comment)

@dataclass
classReprimand(BaseNode[QuestionState]):
  comment: str
  async defrun(self, ctx: GraphRunContext[QuestionState]) -> Ask:
    print(f'Comment: {self.comment}')
    ctx.state.question = None
    return Ask()

question_graph = Graph(
  nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)

async defrun_as_continuous():
  state = QuestionState()
  node = Ask()
  end = await question_graph.run(node, state=state)
  print('END:', end.output)

async defrun_as_cli(answer: str | None):
  persistence = FileStatePersistence(Path('question_graph.json'))
  persistence.set_graph_types(question_graph)
  if snapshot := await persistence.load_next():
    state = snapshot.state
    assert answer is not None, (
      'answer required, usage "uv run -m pydantic_ai_examples.question_graph cli <answer>"'
    )
    node = Evaluate(answer)
  else:
    state = QuestionState()
    node = Ask()
  # debug(state, node)
  async with question_graph.iter(node, state=state, persistence=persistence) as run:
    while True:
      node = await run.next()
      if isinstance(node, End):
        print('END:', node.data)
        history = await persistence.load_all()
        print('history:', '\n'.join(str(e.node) for e in history), sep='\n')
        print('Finished!')
        break
      elif isinstance(node, Answer):
        print(node.question)
        break
      # otherwise just continue

if __name__ == '__main__':
  importasyncio
  importsys
  try:
    sub_command = sys.argv[1]
    assert sub_command in ('continuous', 'cli', 'mermaid')
  except (IndexError, AssertionError):
    print(
      'Usage:\n'
      ' uv run -m pydantic_ai_examples.question_graph mermaid\n'
      'or:\n'
      ' uv run -m pydantic_ai_examples.question_graph continuous\n'
      'or:\n'
      ' uv run -m pydantic_ai_examples.question_graph cli [answer]',
      file=sys.stderr,
    )
    sys.exit(1)
  if sub_command == 'mermaid':
    print(question_graph.mermaid_code(start_node=Ask))
  elif sub_command == 'continuous':
    asyncio.run(run_as_continuous())
  else:
    a = sys.argv[2] if len(sys.argv)  2 else None
    asyncio.run(run_as_cli(a))

```

The mermaid diagram generated in this example looks like this:

[ Skip to content ](https://ai.pydantic.dev/examples/rag/#rag)
# RAG
RAG search example. This demo allows you to ask question of the [logfire](https://pydantic.dev/logfire) documentation.
Demonstrates:
  * [agent dependencies](https://ai.pydantic.dev/dependencies/)
  * RAG search


This is done by creating a database containing each section of the markdown documentation, then registering the search tool with the PydanticAI agent.
Logic for extracting sections from markdown files and a JSON file with that data is available in [this gist](https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992).
[PostgreSQL with pgvector](https://github.com/pgvector/pgvector) is used as the search database, the easiest way to download and run pgvector is using Docker:
```
mkdirpostgres-data
dockerrun--rm\
-ePOSTGRES_PASSWORD=postgres\
-p54320:5432\
-v`pwd`/postgres-data:/var/lib/postgresql/data\
pgvector/pgvector:pg17

```

As with the [SQL gen](https://ai.pydantic.dev/examples/sql-gen/) example, we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running. We also mount the PostgreSQL `data` directory locally to persist the data if you need to stop and restart the container.
With that running and [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), we can build the search database with (**WARNING** : this requires the `OPENAI_API_KEY` env variable and will calling the OpenAI embedding API around 300 times to generate embeddings for each section of the documentation):
```
python-mpydantic_ai_examples.ragbuild

```

```
uvrun-mpydantic_ai_examples.ragbuild

```

(Note building the database doesn't use PydanticAI right now, instead it uses the OpenAI SDK directly.)
You can then ask the agent a question with:
```
python-mpydantic_ai_examples.ragsearch"How do I configure logfire to work with FastAPI?"

```

```
uvrun-mpydantic_ai_examples.ragsearch"How do I configure logfire to work with FastAPI?"

```

## Example Code
rag.py```
from__future__import annotations as _annotations
importasyncio
importre
importsys
importunicodedata
fromcontextlibimport asynccontextmanager
fromdataclassesimport dataclass
importasyncpg
importhttpx
importlogfire
importpydantic_core
fromopenaiimport AsyncOpenAI
frompydanticimport TypeAdapter
fromtyping_extensionsimport AsyncGenerator
frompydantic_aiimport RunContext
frompydantic_ai.agentimport Agent
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()

@dataclass
classDeps:
  openai: AsyncOpenAI
  pool: asyncpg.Pool

agent = Agent('openai:gpt-4o', deps_type=Deps, instrument=True)

@agent.tool
async defretrieve(context: RunContext[Deps], search_query: str) -> str:
"""Retrieve documentation sections based on a search query.
  Args:
    context: The call context.
    search_query: The search query.
  """
  with logfire.span(
    'create embedding for {search_query=}', search_query=search_query
  ):
    embedding = await context.deps.openai.embeddings.create(
      input=search_query,
      model='text-embedding-3-small',
    )
  assert len(embedding.data) == 1, (
    f'Expected 1 embedding, got {len(embedding.data)}, doc query: {search_query!r}'
  )
  embedding = embedding.data[0].embedding
  embedding_json = pydantic_core.to_json(embedding).decode()
  rows = await context.deps.pool.fetch(
    'SELECT url, title, content FROM doc_sections ORDER BY embedding <-> $1 LIMIT 8',
    embedding_json,
  )
  return '\n\n'.join(
    f'# {row["title"]}\nDocumentation URL:{row["url"]}\n\n{row["content"]}\n'
    for row in rows
  )

async defrun_agent(question: str):
"""Entry point to run the agent and perform RAG based question answering."""
  openai = AsyncOpenAI()
  logfire.instrument_openai(openai)
  logfire.info('Asking "{question}"', question=question)
  async with database_connect(False) as pool:
    deps = Deps(openai=openai, pool=pool)
    answer = await agent.run(question, deps=deps)
  print(answer.data)

#######################################################
# The rest of this file is dedicated to preparing the #
# search database, and some utilities.        #
#######################################################
# JSON document from
# https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992
DOCS_JSON = (
  'https://gist.githubusercontent.com/'
  'samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992/raw/'
  '80c5925c42f1442c24963aaf5eb1a324d47afe95/logfire_docs.json'
)

async defbuild_search_db():
"""Build the search database."""
  async with httpx.AsyncClient() as client:
    response = await client.get(DOCS_JSON)
    response.raise_for_status()
  sections = sessions_ta.validate_json(response.content)
  openai = AsyncOpenAI()
  logfire.instrument_openai(openai)
  async with database_connect(True) as pool:
    with logfire.span('create schema'):
      async with pool.acquire() as conn:
        async with conn.transaction():
          await conn.execute(DB_SCHEMA)
    sem = asyncio.Semaphore(10)
    async with asyncio.TaskGroup() as tg:
      for section in sections:
        tg.create_task(insert_doc_section(sem, openai, pool, section))

async definsert_doc_section(
  sem: asyncio.Semaphore,
  openai: AsyncOpenAI,
  pool: asyncpg.Pool,
  section: DocsSection,
) -> None:
  async with sem:
    url = section.url()
    exists = await pool.fetchval('SELECT 1 FROM doc_sections WHERE url = $1', url)
    if exists:
      logfire.info('Skipping {url=}', url=url)
      return
    with logfire.span('create embedding for {url=}', url=url):
      embedding = await openai.embeddings.create(
        input=section.embedding_content(),
        model='text-embedding-3-small',
      )
    assert len(embedding.data) == 1, (
      f'Expected 1 embedding, got {len(embedding.data)}, doc section: {section}'
    )
    embedding = embedding.data[0].embedding
    embedding_json = pydantic_core.to_json(embedding).decode()
    await pool.execute(
      'INSERT INTO doc_sections (url, title, content, embedding) VALUES ($1, $2, $3, $4)',
      url,
      section.title,
      section.content,
      embedding_json,
    )

@dataclass
classDocsSection:
  id: int
  parent: int | None
  path: str
  level: int
  title: str
  content: str
  defurl(self) -> str:
    url_path = re.sub(r'\.md$', '', self.path)
    return (
      f'https://logfire.pydantic.dev/docs/{url_path}/#{slugify(self.title,"-")}'
    )
  defembedding_content(self) -> str:
    return '\n\n'.join((f'path: {self.path}', f'title: {self.title}', self.content))

sessions_ta = TypeAdapter(list[DocsSection])

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async defdatabase_connect(
  create_db: bool = False,
) -> AsyncGenerator[asyncpg.Pool, None]:
  server_dsn, database = (
    'postgresql://postgres:postgres@localhost:54320',
    'pydantic_ai_rag',
  )
  if create_db:
    with logfire.span('check and create DB'):
      conn = await asyncpg.connect(server_dsn)
      try:
        db_exists = await conn.fetchval(
          'SELECT 1 FROM pg_database WHERE datname = $1', database
        )
        if not db_exists:
          await conn.execute(f'CREATE DATABASE {database}')
      finally:
        await conn.close()
  pool = await asyncpg.create_pool(f'{server_dsn}/{database}')
  try:
    yield pool
  finally:
    await pool.close()

DB_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS doc_sections (
  id serial PRIMARY KEY,
  url text NOT NULL UNIQUE,
  title text NOT NULL,
  content text NOT NULL,
  -- text-embedding-3-small returns a vector of 1536 floats
  embedding vector(1536) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_embedding ON doc_sections USING hnsw (embedding vector_l2_ops);
"""

defslugify(value: str, separator: str, unicode: bool = False) -> str:
"""Slugify a string, to make it URL friendly."""
  # Taken unchanged from https://github.com/Python-Markdown/markdown/blob/3.7/markdown/extensions/toc.py#L38
  if not unicode:
    # Replace Extended Latin characters with ASCII, i.e. `žlutý` => `zluty`
    value = unicodedata.normalize('NFKD', value)
    value = value.encode('ascii', 'ignore').decode('ascii')
  value = re.sub(r'[^\w\s-]', '', value).strip().lower()
  return re.sub(rf'[{separator}\s]+', separator, value)

if __name__ == '__main__':
  action = sys.argv[1] if len(sys.argv)  1 else None
  if action == 'build':
    asyncio.run(build_search_db())
  elif action == 'search':
    if len(sys.argv) == 3:
      q = sys.argv[2]
    else:
      q = 'How do I configure logfire to work with FastAPI?'
    asyncio.run(run_agent(q))
  else:
    print(
      'uv run --extra examples -m pydantic_ai_examples.rag build|search',
      file=sys.stderr,
    )
    sys.exit(1)

```


[ Skip to content ](https://ai.pydantic.dev/examples/sql-gen/#sql-generation)
# SQL Generation
Example demonstrating how to use PydanticAI to generate SQL queries based on user input.
Demonstrates:
  * [dynamic system prompt](https://ai.pydantic.dev/agents/#system-prompts)
  * [structured `result_type`](https://ai.pydantic.dev/results/#structured-result-validation)
  * [result validation](https://ai.pydantic.dev/results/#result-validators-functions)
  * [agent dependencies](https://ai.pydantic.dev/dependencies/)


## Running the Example
The resulting SQL is validated by running it as an `EXPLAIN` query on PostgreSQL. To run the example, you first need to run PostgreSQL, e.g. via Docker:
```
dockerrun--rm-ePOSTGRES_PASSWORD=postgres-p54320:5432postgres

```

_(we run postgres on port`54320` to avoid conflicts with any other postgres instances you may have running)_
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.sql_gen

```

```
uvrun-mpydantic_ai_examples.sql_gen

```

or to use a custom prompt:
```
python-mpydantic_ai_examples.sql_gen"find me errors"

```

```
uvrun-mpydantic_ai_examples.sql_gen"find me errors"

```

This model uses `gemini-1.5-flash` by default since Gemini is good at single shot queries of this kind.
## Example Code
sql_gen.py```
importasyncio
importsys
fromcollections.abcimport AsyncGenerator
fromcontextlibimport asynccontextmanager
fromdataclassesimport dataclass
fromdatetimeimport date
fromtypingimport Annotated, Any, Union
importasyncpg
importlogfire
fromannotated_typesimport MinLen
fromdevtoolsimport debug
frompydanticimport BaseModel, Field
fromtyping_extensionsimport TypeAlias
frompydantic_aiimport Agent, ModelRetry, RunContext
frompydantic_ai.format_as_xmlimport format_as_xml
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_asyncpg()
DB_SCHEMA = """
CREATE TABLE records (
  created_at timestamptz,
  start_timestamp timestamptz,
  end_timestamp timestamptz,
  trace_id text,
  span_id text,
  parent_span_id text,
  level log_level,
  span_name text,
  message text,
  attributes_json_schema text,
  attributes jsonb,
  tags text[],
  is_exception boolean,
  otel_status_message text,
  service_name text
);
"""
SQL_EXAMPLES = [
  {
    'request': 'show me records where foobar is false',
    'response': "SELECT * FROM records WHERE attributes->>'foobar' = false",
  },
  {
    'request': 'show me records where attributes include the key "foobar"',
    'response': "SELECT * FROM records WHERE attributes ? 'foobar'",
  },
  {
    'request': 'show me records from yesterday',
    'response': "SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'",
  },
  {
    'request': 'show me error records with the tag "foobar"',
    'response': "SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)",
  },
]

@dataclass
classDeps:
  conn: asyncpg.Connection

classSuccess(BaseModel):
"""Response when SQL could be successfully generated."""
  sql_query: Annotated[str, MinLen(1)]
  explanation: str = Field(
    '', description='Explanation of the SQL query, as markdown'
  )

classInvalidRequest(BaseModel):
"""Response the user input didn't include enough information to generate SQL."""
  error_message: str

Response: TypeAlias = Union[Success, InvalidRequest]
agent: Agent[Deps, Response] = Agent(
  'google-gla:gemini-1.5-flash',
  # Type ignore while we wait for PEP-0747, nonetheless unions will work fine everywhere else
  result_type=Response, # type: ignore
  deps_type=Deps,
  instrument=True,
)

@agent.system_prompt
async defsystem_prompt() -> str:
  return f"""\
Given the following PostgreSQL table of records, your job is to
write a SQL query that suits the user's request.
Database schema:
{DB_SCHEMA}
today's date = {date.today()}
{format_as_xml(SQL_EXAMPLES)}
"""

@agent.result_validator
async defvalidate_result(ctx: RunContext[Deps], result: Response) -> Response:
  if isinstance(result, InvalidRequest):
    return result
  # gemini often adds extraneous backslashes to SQL
  result.sql_query = result.sql_query.replace('\\', '')
  if not result.sql_query.upper().startswith('SELECT'):
    raise ModelRetry('Please create a SELECT query')
  try:
    await ctx.deps.conn.execute(f'EXPLAIN {result.sql_query}')
  except asyncpg.exceptions.PostgresError as e:
    raise ModelRetry(f'Invalid query: {e}') frome
  else:
    return result

async defmain():
  if len(sys.argv) == 1:
    prompt = 'show me logs from yesterday, with level "error"'
  else:
    prompt = sys.argv[1]
  async with database_connect(
    'postgresql://postgres:postgres@localhost:54320', 'pydantic_ai_sql_gen'
  ) as conn:
    deps = Deps(conn)
    result = await agent.run(prompt, deps=deps)
  debug(result.data)

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async defdatabase_connect(server_dsn: str, database: str) -> AsyncGenerator[Any, None]:
  with logfire.span('check and create DB'):
    conn = await asyncpg.connect(server_dsn)
    try:
      db_exists = await conn.fetchval(
        'SELECT 1 FROM pg_database WHERE datname = $1', database
      )
      if not db_exists:
        await conn.execute(f'CREATE DATABASE {database}')
    finally:
      await conn.close()
  conn = await asyncpg.connect(f'{server_dsn}/{database}')
  try:
    with logfire.span('create schema'):
      async with conn.transaction():
        if not db_exists:
          await conn.execute(
            "CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical')"
          )
          await conn.execute(DB_SCHEMA)
    yield conn
  finally:
    await conn.close()

if __name__ == '__main__':
  asyncio.run(main())

```


# Stream markdown
This example shows how to stream markdown from an agent, using the [`rich`](https://github.com/Textualize/rich) library to highlight the output in the terminal.
It'll run the example with both OpenAI and Google Gemini models if the required environment variables are set.
Demonstrates:
  * [streaming text responses](https://ai.pydantic.dev/results/#streaming-text)


## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.stream_markdown

```

```
uvrun-mpydantic_ai_examples.stream_markdown

```

## Example Code
```
importasyncio
importos
importlogfire
fromrich.consoleimport Console, ConsoleOptions, RenderResult
fromrich.liveimport Live
fromrich.markdownimport CodeBlock, Markdown
fromrich.syntaximport Syntax
fromrich.textimport Text
frompydantic_aiimport Agent
frompydantic_ai.modelsimport KnownModelName
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
agent = Agent(instrument=True)
# models to try, and the appropriate env var
models: list[tuple[KnownModelName, str]] = [
  ('google-gla:gemini-1.5-flash', 'GEMINI_API_KEY'),
  ('openai:gpt-4o-mini', 'OPENAI_API_KEY'),
  ('groq:llama-3.3-70b-versatile', 'GROQ_API_KEY'),
]

async defmain():
  prettier_code_blocks()
  console = Console()
  prompt = 'Show me a short example of using Pydantic.'
  console.log(f'Asking: {prompt}...', style='cyan')
  for model, env_var in models:
    if env_var in os.environ:
      console.log(f'Using model: {model}')
      with Live('', console=console, vertical_overflow='visible') as live:
        async with agent.run_stream(prompt, model=model) as result:
          async for message in result.stream():
            live.update(Markdown(message))
      console.log(result.usage())
    else:
      console.log(f'{model} requires {env_var} to be set.')

defprettier_code_blocks():
"""Make rich code blocks prettier and easier to copy.
  From https://github.com/samuelcolvin/aicli/blob/v0.8.0/samuelcolvin_aicli.py#L22
  """
  classSimpleCodeBlock(CodeBlock):
    def__rich_console__(
      self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
      code = str(self.text).rstrip()
      yield Text(self.lexer_name, style='dim')
      yield Syntax(
        code,
        self.lexer_name,
        theme=self.theme,
        background_color='default',
        word_wrap=True,
      )
      yield Text(f'/{self.lexer_name}', style='dim')
  Markdown.elements['fence'] = SimpleCodeBlock

if __name__ == '__main__':
  asyncio.run(main())

```


# Stream whales
Information about whales — an example of streamed structured response validation.
Demonstrates:
  * [streaming structured responses](https://ai.pydantic.dev/results/#streaming-structured-responses)


This script streams structured responses from GPT-4 about whales, validates the data and displays it as a dynamic table using [`rich`](https://github.com/Textualize/rich) as the data is received.
## Running the Example
With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.stream_whales

```

```
uvrun-mpydantic_ai_examples.stream_whales

```

Should give an output like this:
## Example Code
stream_whales.py```
fromtypingimport Annotated
importlogfire
frompydanticimport Field, ValidationError
fromrich.consoleimport Console
fromrich.liveimport Live
fromrich.tableimport Table
fromtyping_extensionsimport NotRequired, TypedDict
frompydantic_aiimport Agent
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

classWhale(TypedDict):
  name: str
  length: Annotated[
    float, Field(description='Average length of an adult whale in meters.')
  ]
  weight: NotRequired[
    Annotated[
      float,
      Field(description='Average weight of an adult whale in kilograms.', ge=50),
    ]
  ]
  ocean: NotRequired[str]
  description: NotRequired[Annotated[str, Field(description='Short Description')]]

agent = Agent('openai:gpt-4', result_type=list[Whale], instrument=True)

async defmain():
  console = Console()
  with Live('\n' * 36, console=console) as live:
    console.print('Requesting data...', style='cyan')
    async with agent.run_stream(
      'Generate me details of 5 species of Whale.'
    ) as result:
      console.print('Response:', style='green')
      async for message, last in result.stream_structured(debounce_by=0.01):
        try:
          whales = await result.validate_structured_result(
            message, allow_partial=not last
          )
        except ValidationError as exc:
          if all(
            e['type'] == 'missing' and e['loc'] == ('response',)
            for e in exc.errors()
          ):
            continue
          else:
            raise
        table = Table(
          title='Species of Whale',
          caption='Streaming Structured responses from GPT-4',
          width=120,
        )
        table.add_column('ID', justify='right')
        table.add_column('Name')
        table.add_column('Avg. Length (m)', justify='right')
        table.add_column('Avg. Weight (kg)', justify='right')
        table.add_column('Ocean')
        table.add_column('Description', justify='right')
        for wid, whale in enumerate(whales, start=1):
          table.add_row(
            str(wid),
            whale['name'],
            f'{whale["length"]:0.0f}',
            f'{w:0.0f}' if (w := whale.get('weight')) else '…',
            whale.get('ocean') or '…',
            whale.get('description') or '…',
          )
        live.update(table)

if __name__ == '__main__':
  importasyncio
  asyncio.run(main())

```


# Weather agent
Example of PydanticAI with multiple tools which the LLM needs to call in turn to answer a question.
Demonstrates:
  * [agent dependencies](https://ai.pydantic.dev/dependencies/)
  * [streaming text responses](https://ai.pydantic.dev/results/#streaming-text)
  * Building a [Gradio](https://www.gradio.app/) UI for the agent


In this case the idea is a "weather" agent — the user can ask for the weather in multiple locations, the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use the `get_weather` tool to get the weather for those locations.
## Running the Example
To run this example properly, you might want to add two extra API keys **(Note if either key is missing, the code will fall back to dummy data, so they're not required)** :
  * A weather API key from [tomorrow.io](https://www.tomorrow.io/weather-api/) set via `WEATHER_API_KEY`
  * A geocoding API key from [geocode.maps.co](https://geocode.maps.co/) set via `GEO_API_KEY`


With [dependencies installed and environment variables set](https://ai.pydantic.dev/examples/#usage), run:
```
python-mpydantic_ai_examples.weather_agent

```

```
uvrun-mpydantic_ai_examples.weather_agent

```

## Example Code
pydantic_ai_examples/weather_agent.py```
from__future__import annotations as _annotations
importasyncio
importos
fromdataclassesimport dataclass
fromtypingimport Any
importlogfire
fromdevtoolsimport debug
fromhttpximport AsyncClient
frompydantic_aiimport Agent, ModelRetry, RunContext
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

@dataclass
classDeps:
  client: AsyncClient
  weather_api_key: str | None
  geo_api_key: str | None

weather_agent = Agent(
  'openai:gpt-4o',
  # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
  # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
  system_prompt=(
    'Be concise, reply with one sentence.'
    'Use the `get_lat_lng` tool to get the latitude and longitude of the locations, '
    'then use the `get_weather` tool to get the weather.'
  ),
  deps_type=Deps,
  retries=2,
  instrument=True,
)

@weather_agent.tool
async defget_lat_lng(
  ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
"""Get the latitude and longitude of a location.
  Args:
    ctx: The context.
    location_description: A description of a location.
  """
  if ctx.deps.geo_api_key is None:
    # if no API key is provided, return a dummy response (London)
    return {'lat': 51.1, 'lng': -0.1}
  params = {
    'q': location_description,
    'api_key': ctx.deps.geo_api_key,
  }
  with logfire.span('calling geocode API', params=params) as span:
    r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
    r.raise_for_status()
    data = r.json()
    span.set_attribute('response', data)
  if data:
    return {'lat': data[0]['lat'], 'lng': data[0]['lon']}
  else:
    raise ModelRetry('Could not find the location')

@weather_agent.tool
async defget_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
"""Get the weather at a location.
  Args:
    ctx: The context.
    lat: Latitude of the location.
    lng: Longitude of the location.
  """
  if ctx.deps.weather_api_key is None:
    # if no API key is provided, return a dummy response
    return {'temperature': '21 °C', 'description': 'Sunny'}
  params = {
    'apikey': ctx.deps.weather_api_key,
    'location': f'{lat},{lng}',
    'units': 'metric',
  }
  with logfire.span('calling weather API', params=params) as span:
    r = await ctx.deps.client.get(
      'https://api.tomorrow.io/v4/weather/realtime', params=params
    )
    r.raise_for_status()
    data = r.json()
    span.set_attribute('response', data)
  values = data['data']['values']
  # https://docs.tomorrow.io/reference/data-layers-weather-codes
  code_lookup = {
    1000: 'Clear, Sunny',
    1100: 'Mostly Clear',
    1101: 'Partly Cloudy',
    1102: 'Mostly Cloudy',
    1001: 'Cloudy',
    2000: 'Fog',
    2100: 'Light Fog',
    4000: 'Drizzle',
    4001: 'Rain',
    4200: 'Light Rain',
    4201: 'Heavy Rain',
    5000: 'Snow',
    5001: 'Flurries',
    5100: 'Light Snow',
    5101: 'Heavy Snow',
    6000: 'Freezing Drizzle',
    6001: 'Freezing Rain',
    6200: 'Light Freezing Rain',
    6201: 'Heavy Freezing Rain',
    7000: 'Ice Pellets',
    7101: 'Heavy Ice Pellets',
    7102: 'Light Ice Pellets',
    8000: 'Thunderstorm',
  }
  return {
    'temperature': f'{values["temperatureApparent"]:0.0f}°C',
    'description': code_lookup.get(values['weatherCode'], 'Unknown'),
  }

async defmain():
  async with AsyncClient() as client:
    # create a free API key at https://www.tomorrow.io/weather-api/
    weather_api_key = os.getenv('WEATHER_API_KEY')
    # create a free API key at https://geocode.maps.co/
    geo_api_key = os.getenv('GEO_API_KEY')
    deps = Deps(
      client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key
    )
    result = await weather_agent.run(
      'What is the weather like in London and in Wiltshire?', deps=deps
    )
    debug(result)
    print('Response:', result.data)

if __name__ == '__main__':
  asyncio.run(main())

```

## Running the UI
You can build multi-turn chat applications for your agent with [Gradio](https://www.gradio.app/), a framework for building AI web applications entirely in python. Gradio comes with built-in chat components and agent support so the entire UI will be implemented in a single python file!
Here's what the UI looks like for the weather agent:
Note, to run the UI, you'll need Python 3.10+.
```
pipinstallgradio>=5.9.0
python/uv-run-mpydantic_ai_examples.weather_agent_gradio

```

## UI Code
pydantic_ai_examples/weather_agent_gradio.py```
from__future__import annotations as _annotations
importjson
importos
fromhttpximport AsyncClient
frompydantic_ai.messagesimport ToolCallPart, ToolReturnPart
frompydantic_ai_examples.weather_agentimport Deps, weather_agent
try:
  importgradioasgr
except ImportError as e:
  raise ImportError(
    'Please install gradio with `pip install gradio`. You must use python>=3.10.'
  ) frome
TOOL_TO_DISPLAY_NAME = {'get_lat_lng': 'Geocoding API', 'get_weather': 'Weather API'}
client = AsyncClient()
weather_api_key = os.getenv('WEATHER_API_KEY')
# create a free API key at https://geocode.maps.co/
geo_api_key = os.getenv('GEO_API_KEY')
deps = Deps(client=client, weather_api_key=weather_api_key, geo_api_key=geo_api_key)

async defstream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
  chatbot.append({'role': 'user', 'content': prompt})
  yield gr.Textbox(interactive=False, value=''), chatbot, gr.skip()
  async with weather_agent.run_stream(
    prompt, deps=deps, message_history=past_messages
  ) as result:
    for message in result.new_messages():
      for call in message.parts:
        if isinstance(call, ToolCallPart):
          call_args = (
            call.args.args_json
            if hasattr(call.args, 'args_json')
            else json.dumps(call.args.args_dict)
          )
          metadata = {
            'title': f'🛠️ Using {TOOL_TO_DISPLAY_NAME[call.tool_name]}',
          }
          if call.tool_call_id is not None:
            metadata['id'] = {call.tool_call_id}
          gr_message = {
            'role': 'assistant',
            'content': 'Parameters: ' + call_args,
            'metadata': metadata,
          }
          chatbot.append(gr_message)
        if isinstance(call, ToolReturnPart):
          for gr_message in chatbot:
            if (
              gr_message.get('metadata', {}).get('id', '')
              == call.tool_call_id
            ):
              gr_message['content'] += (
                f'\nOutput: {json.dumps(call.content)}'
              )
        yield gr.skip(), chatbot, gr.skip()
    chatbot.append({'role': 'assistant', 'content': ''})
    async for message in result.stream_text():
      chatbot[-1]['content'] = message
      yield gr.skip(), chatbot, gr.skip()
    past_messages = result.all_messages()
    yield gr.Textbox(interactive=True), gr.skip(), past_messages

async defhandle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
  new_history = chatbot[: retry_data.index]
  previous_prompt = chatbot[retry_data.index]['content']
  past_messages = past_messages[: retry_data.index]
  async for update in stream_from_agent(previous_prompt, new_history, past_messages):
    yield update

defundo(chatbot, past_messages: list, undo_data: gr.UndoData):
  new_history = chatbot[: undo_data.index]
  past_messages = past_messages[: undo_data.index]
  return chatbot[undo_data.index]['content'], new_history, past_messages

defselect_data(message: gr.SelectData) -> str:
  return message.value['text']

with gr.Blocks() as demo:
  gr.HTML(
"""
<div style="display: flex; justify-content: center; align-items: center; gap: 2rem; padding: 1rem; width: 100%">
  <img src="https://ai.pydantic.dev/img/logo-white.svg" style="max-width: 200px; height: auto">
  <div>
    <h1 style="margin: 0 0 1rem 0">Weather Assistant</h1>
    <h3 style="margin: 0 0 0.5rem 0">
      This assistant answer your weather questions.
    </h3>
  </div>
</div>
"""
  )
  past_messages = gr.State([])
  chatbot = gr.Chatbot(
    label='Packing Assistant',
    type='messages',
    avatar_images=(None, 'https://ai.pydantic.dev/img/logo-white.svg'),
    examples=[
      {'text': 'What is the weather like in Miami?'},
      {'text': 'What is the weather like in London?'},
    ],
  )
  with gr.Row():
    prompt = gr.Textbox(
      lines=1,
      show_label=False,
      placeholder='What is the weather like in New York City?',
    )
  generation = prompt.submit(
    stream_from_agent,
    inputs=[prompt, chatbot, past_messages],
    outputs=[prompt, chatbot, past_messages],
  )
  chatbot.example_select(select_data, None, [prompt])
  chatbot.retry(
    handle_retry, [chatbot, past_messages], [prompt, chatbot, past_messages]
  )
  chatbot.undo(undo, [chatbot, past_messages], [prompt, chatbot, past_messages])

if __name__ == '__main__':
  demo.launch()

```


[ Skip to content ](https://ai.pydantic.dev/graph/#graphs)
# Graphs
Don't use a nail gun unless you need a nail gun
If PydanticAI [agents](https://ai.pydantic.dev/agents/) are a hammer, and [multi-agent workflows](https://ai.pydantic.dev/multi-agent-applications/) are a sledgehammer, then graphs are a nail gun:
  * sure, nail guns look cooler than hammers
  * but nail guns take a lot more setup than hammers
  * and nail guns don't make you a better builder, they make you a builder with a nail gun
  * Lastly, (and at the risk of torturing this metaphor), if you're a fan of medieval tools like mallets and untyped Python, you probably won't like nail guns or our approach to graphs. (But then again, if you're not a fan of type hints in Python, you've probably already bounced off PydanticAI to use one of the toy agent frameworks — good luck, and feel free to borrow my sledgehammer when you realize you need it)


In short, graphs are a powerful tool, but they're not the right tool for every job. Please consider other [multi-agent approaches](https://ai.pydantic.dev/multi-agent-applications/) before proceeding.
If you're not confident a graph-based approach is a good idea, it might be unnecessary.
Graphs and finite state machines (FSMs) are a powerful abstraction to model, execute, control and visualize complex workflows.
Alongside PydanticAI, we've developed `pydantic-graph` — an async graph and state machine library for Python where nodes and edges are defined using type hints.
While this library is developed as part of PydanticAI; it has no dependency on `pydantic-ai` and can be considered as a pure graph-based state machine library. You may find it useful whether or not you're using PydanticAI or even building with GenAI.
`pydantic-graph` is designed for advanced users and makes heavy use of Python generics and type hints. It is not designed to be as beginner-friendly as PydanticAI.
## Installation
`pydantic-graph` is a required dependency of `pydantic-ai`, and an optional dependency of `pydantic-ai-slim`, see [installation instructions](https://ai.pydantic.dev/install/#slim-install) for more information. You can also install it directly:
```
pipinstallpydantic-graph

```

```
uvaddpydantic-graph

```

## Graph Types
`pydantic-graph` is made up of a few key components:
### GraphRunContext
[`GraphRunContext`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.GraphRunContext) — The context for the graph run, similar to PydanticAI's [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext). This holds the state of the graph and dependencies and is passed to nodes when they're run.
`GraphRunContext` is generic in the state type of the graph it's used in, [`StateT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT).
### End
[`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End) — return value to indicate the graph run should end.
`End` is generic in the graph return type of the graph it's used in, [`RunEndT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT).
### Nodes
Subclasses of [`BaseNode`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode) define nodes for execution in the graph.
Nodes, which are generally [`dataclass`es](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass), generally consist of:
  * fields containing any parameters required/optional when calling the node
  * the business logic to execute the node, in the [`run`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.run) method
  * return annotations of the [`run`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.run) method, which are read by `pydantic-graph` to determine the outgoing edges of the node


Nodes are generic in:
  * **state** , which must have the same type as the state of graphs they're included in, [`StateT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT) has a default of `None`, so if you're not using state you can omit this generic parameter, see [stateful graphs](https://ai.pydantic.dev/graph/#stateful-graphs) for more information
  * **deps** , which must have the same type as the deps of the graph they're included in, [`DepsT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT) has a default of `None`, so if you're not using deps you can omit this generic parameter, see [dependency injection](https://ai.pydantic.dev/graph/#dependency-injection) for more information
  * **graph return type** — this only applies if the node returns [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End). [`RunEndT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT) has a default of [Never](https://docs.python.org/3/library/typing.html#typing.Never) so this generic parameter can be omitted if the node doesn't return `End`, but must be included if it does.


Here's an example of a start or intermediate node in a graph — it can't end the run as it doesn't return [`End`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.End):
intermediate_node.py```
fromdataclassesimport dataclass
frompydantic_graphimport BaseNode, GraphRunContext

@dataclass
classMyNode(BaseNode[MyState]): 
  foo: int 
  async defrun(
    self,
    ctx: GraphRunContext[MyState], 
  ) -> AnotherNode: 
    ...
    return AnotherNode()

```

We could extend `MyNode` to optionally end the run if `foo` is divisible by 5:
intermediate_or_end_node.py```
fromdataclassesimport dataclass
frompydantic_graphimport BaseNode, End, GraphRunContext

@dataclass
classMyNode(BaseNode[MyState, None, int]): 
  foo: int
  async defrun(
    self,
    ctx: GraphRunContext[MyState],
  ) -> AnotherNode | End[int]: 
    if self.foo % 5 == 0:
      return End(self.foo)
    else:
      return AnotherNode()

```

### Graph
[`Graph`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph) — this is the execution graph itself, made up of a set of [node classes](https://ai.pydantic.dev/graph/#nodes) (i.e., `BaseNode` subclasses).
`Graph` is generic in:
  * **state** the state type of the graph, [`StateT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.StateT)
  * **deps** the deps type of the graph, [`DepsT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.DepsT)
  * **graph return type** the return type of the graph run, [`RunEndT`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.RunEndT)


Here's an example of a simple graph:
graph_example.py```
from__future__import annotations
fromdataclassesimport dataclass
frompydantic_graphimport BaseNode, End, Graph, GraphRunContext

@dataclass
classDivisibleBy5(BaseNode[None, None, int]): 
  foo: int
  async defrun(
    self,
    ctx: GraphRunContext,
  ) -> Increment | End[int]:
    if self.foo % 5 == 0:
      return End(self.foo)
    else:
      return Increment(self.foo)

@dataclass
classIncrement(BaseNode): 
  foo: int
  async defrun(self, ctx: GraphRunContext) -> DivisibleBy5:
    return DivisibleBy5(self.foo + 1)

fives_graph = Graph(nodes=[DivisibleBy5, Increment]) 
result = fives_graph.run_sync(DivisibleBy5(4)) 
print(result.output)
#> 5

```

_(This example is complete, it can be run "as is" with Python 3.10+)_
A [mermaid diagram](https://ai.pydantic.dev/graph/#mermaid-diagrams) for this graph can be generated with the following code:
graph_example_diagram.py```
fromgraph_exampleimport DivisibleBy5, fives_graph
fives_graph.mermaid_code(start_node=DivisibleBy5)

```

In order to visualize a graph within a `jupyter-notebook`, `IPython.display` needs to be used:
jupyter_display_mermaid.py```
fromgraph_exampleimport DivisibleBy5, fives_graph
fromIPython.displayimport Image, display
display(Image(fives_graph.mermaid_image(start_node=DivisibleBy5)))

```

## Stateful Graphs
The "state" concept in `pydantic-graph` provides an optional way to access and mutate an object (often a `dataclass` or Pydantic model) as nodes run in a graph. If you think of Graphs as a production line, then your state is the engine being passed along the line and built up by each node as the graph is run.
In the future, we intend to extend `pydantic-graph` to provide state persistence with the state recorded after each node is run, see [#695](https://github.com/pydantic/pydantic-ai/issues/695).
Here's an example of a graph which represents a vending machine where the user may insert coins and select a product to purchase.
vending_machine.py```
from__future__import annotations
fromdataclassesimport dataclass
fromrich.promptimport Prompt
frompydantic_graphimport BaseNode, End, Graph, GraphRunContext

@dataclass
classMachineState: 
  user_balance: float = 0.0
  product: str | None = None

@dataclass
classInsertCoin(BaseNode[MachineState]): 
  async defrun(self, ctx: GraphRunContext[MachineState]) -> CoinsInserted: 
    return CoinsInserted(float(Prompt.ask('Insert coins'))) 

@dataclass
classCoinsInserted(BaseNode[MachineState]):
  amount: float 
  async defrun(
    self, ctx: GraphRunContext[MachineState]
  ) -> SelectProduct | Purchase: 
    ctx.state.user_balance += self.amount 
    if ctx.state.product is not None: 
      return Purchase(ctx.state.product)
    else:
      return SelectProduct()

@dataclass
classSelectProduct(BaseNode[MachineState]):
  async defrun(self, ctx: GraphRunContext[MachineState]) -> Purchase:
    return Purchase(Prompt.ask('Select product'))

PRODUCT_PRICES = { 
  'water': 1.25,
  'soda': 1.50,
  'crisps': 1.75,
  'chocolate': 2.00,
}

@dataclass
classPurchase(BaseNode[MachineState, None, None]): 
  product: str
  async defrun(
    self, ctx: GraphRunContext[MachineState]
  ) -> End | InsertCoin | SelectProduct:
    if price := PRODUCT_PRICES.get(self.product): 
      ctx.state.product = self.product 
      if ctx.state.user_balance >= price: 
        ctx.state.user_balance -= price
        return End(None)
      else:
        diff = price - ctx.state.user_balance
        print(f'Not enough money for {self.product}, need {diff:0.2f} more')
        #> Not enough money for crisps, need 0.75 more
        return InsertCoin() 
    else:
      print(f'No such product: {self.product}, try again')
      return SelectProduct() 

vending_machine_graph = Graph( 
  nodes=[InsertCoin, CoinsInserted, SelectProduct, Purchase]
)

async defmain():
  state = MachineState() 
  await vending_machine_graph.run(InsertCoin(), state=state) 
  print(f'purchase successful item={state.product} change={state.user_balance:0.2f}')
  #> purchase successful item=crisps change=0.25

```

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add`asyncio.run(main())` to run `main`)_
A [mermaid diagram](https://ai.pydantic.dev/graph/#mermaid-diagrams) for this graph can be generated with the following code:
vending_machine_diagram.py```
fromvending_machineimport InsertCoin, vending_machine_graph
vending_machine_graph.mermaid_code(start_node=InsertCoin)

```

The diagram generated by the above code is:
See [below](https://ai.pydantic.dev/graph/#mermaid-diagrams) for more information on generating diagrams.
## GenAI Example
So far we haven't shown an example of a Graph that actually uses PydanticAI or GenAI at all.
In this example, one agent generates a welcome email to a user and the other agent provides feedback on the email.
This graph has a very simple structure:
genai_email_feedback.py```
from__future__import annotations as _annotations
fromdataclassesimport dataclass, field
frompydanticimport BaseModel, EmailStr
frompydantic_aiimport Agent
frompydantic_ai.format_as_xmlimport format_as_xml
frompydantic_ai.messagesimport ModelMessage
frompydantic_graphimport BaseNode, End, Graph, GraphRunContext

@dataclass
classUser:
  name: str
  email: EmailStr
  interests: list[str]

@dataclass
classEmail:
  subject: str
  body: str

@dataclass
classState:
  user: User
  write_agent_messages: list[ModelMessage] = field(default_factory=list)

email_writer_agent = Agent(
  'google-vertex:gemini-1.5-pro',
  result_type=Email,
  system_prompt='Write a welcome email to our tech blog.',
)

@dataclass
classWriteEmail(BaseNode[State]):
  email_feedback: str | None = None
  async defrun(self, ctx: GraphRunContext[State]) -> Feedback:
    if self.email_feedback:
      prompt = (
        f'Rewrite the email for the user:\n'
        f'{format_as_xml(ctx.state.user)}\n'
        f'Feedback: {self.email_feedback}'
      )
    else:
      prompt = (
        f'Write a welcome email for the user:\n'
        f'{format_as_xml(ctx.state.user)}'
      )
    result = await email_writer_agent.run(
      prompt,
      message_history=ctx.state.write_agent_messages,
    )
    ctx.state.write_agent_messages += result.all_messages()
    return Feedback(result.data)

classEmailRequiresWrite(BaseModel):
  feedback: str

classEmailOk(BaseModel):
  pass

feedback_agent = Agent[None, EmailRequiresWrite | EmailOk](
  'openai:gpt-4o',
  result_type=EmailRequiresWrite | EmailOk, # type: ignore
  system_prompt=(
    'Review the email and provide feedback, email must reference the users specific interests.'
  ),
)

@dataclass
classFeedback(BaseNode[State, None, Email]):
  email: Email
  async defrun(
    self,
    ctx: GraphRunContext[State],
  ) -> WriteEmail | End[Email]:
    prompt = format_as_xml({'user': ctx.state.user, 'email': self.email})
    result = await feedback_agent.run(prompt)
    if isinstance(result.data, EmailRequiresWrite):
      return WriteEmail(email_feedback=result.data.feedback)
    else:
      return End(self.email)

async defmain():
  user = User(
    name='John Doe',
    email='john.joe@example.com',
    interests=['Haskel', 'Lisp', 'Fortran'],
  )
  state = State(user)
  feedback_graph = Graph(nodes=(WriteEmail, Feedback))
  result = await feedback_graph.run(WriteEmail(), state=state)
  print(result.output)
"""
  Email(
    subject='Welcome to our tech blog!',
    body='Hello John, Welcome to our tech blog! ...',

  """

```

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add`asyncio.run(main())` to run `main`)_
## Iterating Over a Graph
### Using `Graph.iter` for `async for` iteration
Sometimes you want direct control or insight into each node as the graph executes. The easiest way to do that is with the [`Graph.iter`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.iter) method, which returns a **context manager** that yields a [`GraphRun`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.GraphRun) object. The `GraphRun` is an async-iterable over the nodes of your graph, allowing you to record or modify them as they execute.
Here's an example:
count_down.py```
from__future__import annotations as _annotations
fromdataclassesimport dataclass
frompydantic_graphimport Graph, BaseNode, End, GraphRunContext

@dataclass
classCountDownState:
  counter: int

@dataclass
classCountDown(BaseNode[CountDownState, None, int]):
  async defrun(self, ctx: GraphRunContext[CountDownState]) -> CountDown | End[int]:
    if ctx.state.counter <= 0:
      return End(ctx.state.counter)
    ctx.state.counter -= 1
    return CountDown()

count_down_graph = Graph(nodes=[CountDown])

async defmain():
  state = CountDownState(counter=3)
  async with count_down_graph.iter(CountDown(), state=state) as run: 
    async for node in run: 
      print('Node:', node)
      #> Node: CountDown()
      #> Node: CountDown()
      #> Node: CountDown()
      #> Node: End(data=0)
  print('Final result:', run.result.output) 
  #> Final result: 0

```

### Using `GraphRun.next(node)` manually
Alternatively, you can drive iteration manually with the [`GraphRun.next`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.GraphRun.next) method, which allows you to pass in whichever node you want to run next. You can modify or selectively skip nodes this way.
Below is a contrived example that stops whenever the counter is at 2, ignoring any node runs beyond that:
count_down_next.py```
frompydantic_graphimport End, FullStatePersistence
fromcount_downimport CountDown, CountDownState, count_down_graph

async defmain():
  state = CountDownState(counter=5)
  persistence = FullStatePersistence() 
  async with count_down_graph.iter(
    CountDown(), state=state, persistence=persistence
  ) as run:
    node = run.next_node 
    while not isinstance(node, End): 
      print('Node:', node)
      #> Node: CountDown()
      #> Node: CountDown()
      #> Node: CountDown()
      #> Node: CountDown()
      if state.counter == 2:
        break 
      node = await run.next(node) 
    print(run.result) 
    #> None
    for step in persistence.history: 
      print('History Step:', step.state, step.state)
      #> History Step: CountDownState(counter=5) CountDownState(counter=5)
      #> History Step: CountDownState(counter=4) CountDownState(counter=4)
      #> History Step: CountDownState(counter=3) CountDownState(counter=3)
      #> History Step: CountDownState(counter=2) CountDownState(counter=2)

```

## State Persistence
One of the biggest benefits of finite state machine (FSM) graphs is how they simplify the handling of interrupted execution. This might happen for a variety of reasons:
  * the state machine logic might fundamentally need to be paused — e.g. the returns workflow for an e-commerce order needs to wait for the item to be posted to the returns center or because execution of the next node needs input from a user so needs to wait for a new http request,
  * the execution takes so long that the entire graph can't reliably be executed in a single continuous run — e.g. a deep research agent that might take hours to run,
  * you want to run multiple graph nodes in parallel in different processes / hardware instances (note: parallel node execution is not yet supported in `pydantic-graph`, see [#704](https://github.com/pydantic/pydantic-ai/issues/704)).


Trying to make a conventional control flow (i.e., boolean logic and nested function calls) implementation compatible with these usage scenarios generally results in brittle and over-complicated spaghetti code, with the logic required to interrupt and resume execution dominating the implementation.
To allow graph runs to be interrupted and resumed, `pydantic-graph` provides state persistence — a system for snapshotting the state of a graph run before and after each node is run, allowing a graph run to be resumed from any point in the graph.
`pydantic-graph` includes three state persistence implementations:
  * [`SimpleStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.in_mem.SimpleStatePersistence) — Simple in memory state persistence that just hold the latest snapshot. If no state persistence implementation is provided when running a graph, this is used by default.
  * [`FullStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.in_mem.FullStatePersistence) — In memory state persistence that hold a list of snapshots.
  * [`FileStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.file.FileStatePersistence) — File-based state persistence that saves snapshots to a JSON file.


In production applications, developers should implement their own state persistence by subclassing [`BaseStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.BaseStatePersistence) abstract base class, which might persist runs in a relational database like PostgresQL.
At a high level the role of `StatePersistence` implementations is to store and retrieve [`NodeSnapshot`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.NodeSnapshot) and [`EndSnapshot`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.EndSnapshot) objects.
[`graph.iter_from_persistence()`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.iter_from_persistence) may be used to run the graph based on the state stored in persistence.
We can run the `count_down_graph` from [above](https://ai.pydantic.dev/graph/#iterating-over-a-graph), using [`graph.iter_from_persistence()`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.iter_from_persistence) and [`FileStatePersistence`](https://ai.pydantic.dev/api/pydantic_graph/persistence/#pydantic_graph.persistence.file.FileStatePersistence).
As you can see in this code, `run_node` requires no external application state (apart from state persistence) to be run, meaning graphs can easily be executed by distributed execution and queueing systems.
count_down_from_persistence.py```
frompathlibimport Path
frompydantic_graphimport End
frompydantic_graph.persistence.fileimport FileStatePersistence
fromcount_downimport CountDown, CountDownState, count_down_graph

async defmain():
  run_id = 'run_abc123'
  persistence = FileStatePersistence(Path(f'count_down_{run_id}.json')) 
  state = CountDownState(counter=5)
  await count_down_graph.initialize( 
    CountDown(), state=state, persistence=persistence
  )
  done = False
  while not done:
    done = await run_node(run_id)

async defrun_node(run_id: str) -> bool: 
  persistence = FileStatePersistence(Path(f'count_down_{run_id}.json'))
  async with count_down_graph.iter_from_persistence(persistence) as run: 
    node_or_end = await run.next() 
  print('Node:', node_or_end)
  #> Node: CountDown()
  #> Node: CountDown()
  #> Node: CountDown()
  #> Node: CountDown()
  #> Node: CountDown()
  #> Node: End(data=0)
  return isinstance(node_or_end, End) 

```

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add`asyncio.run(main())` to run `main`)_
### Example: Human in the loop.
As noted above, state persistence allows graphs to be interrupted and resumed. One use case of this is to allow user input to continue.
In this example, an AI asks the user a question, the user provides an answer, the AI evaluates the answer and ends if the user got it right or asks another question if they got it wrong.
Instead of running the entire graph in a single process invocation, we run the graph by running the process repeatedly, optionally providing an answer to the question as a command line argument.
`ai_q_and_a_graph.py` — `question_graph` definition
ai_q_and_a_graph.py```
from__future__import annotations as _annotations
fromdataclassesimport dataclass, field
fromgroqimport BaseModel
frompydantic_graphimport (
  BaseNode,
  End,
  Graph,
  GraphRunContext,
)
frompydantic_aiimport Agent
frompydantic_ai.format_as_xmlimport format_as_xml
frompydantic_ai.messagesimport ModelMessage
ask_agent = Agent('openai:gpt-4o', result_type=str, instrument=True)

@dataclass
classQuestionState:
  question: str | None = None
  ask_agent_messages: list[ModelMessage] = field(default_factory=list)
  evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)

@dataclass
classAsk(BaseNode[QuestionState]):
  async defrun(self, ctx: GraphRunContext[QuestionState]) -> Answer:
    result = await ask_agent.run(
      'Ask a simple question with a single correct answer.',
      message_history=ctx.state.ask_agent_messages,
    )
    ctx.state.ask_agent_messages += result.all_messages()
    ctx.state.question = result.data
    return Answer(result.data)

@dataclass
classAnswer(BaseNode[QuestionState]):
  question: str
  async defrun(self, ctx: GraphRunContext[QuestionState]) -> Evaluate:
    answer = input(f'{self.question}: ')
    return Evaluate(answer)

classEvaluationResult(BaseModel, use_attribute_docstrings=True):
  correct: bool
"""Whether the answer is correct."""
  comment: str
"""Comment on the answer, reprimand the user if the answer is wrong."""

evaluate_agent = Agent(
  'openai:gpt-4o',
  result_type=EvaluationResult,
  system_prompt='Given a question and answer, evaluate if the answer is correct.',
)

@dataclass
classEvaluate(BaseNode[QuestionState, None, str]):
  answer: str
  async defrun(
    self,
    ctx: GraphRunContext[QuestionState],
  ) -> End[str] | Reprimand:
    assert ctx.state.question is not None
    result = await evaluate_agent.run(
      format_as_xml({'question': ctx.state.question, 'answer': self.answer}),
      message_history=ctx.state.evaluate_agent_messages,
    )
    ctx.state.evaluate_agent_messages += result.all_messages()
    if result.data.correct:
      return End(result.data.comment)
    else:
      return Reprimand(result.data.comment)

@dataclass
classReprimand(BaseNode[QuestionState]):
  comment: str
  async defrun(self, ctx: GraphRunContext[QuestionState]) -> Ask:
    print(f'Comment: {self.comment}')
    ctx.state.question = None
    return Ask()

question_graph = Graph(
  nodes=(Ask, Answer, Evaluate, Reprimand), state_type=QuestionState
)

```

_(This example is complete, it can be run "as is" with Python 3.10+)_
ai_q_and_a_run.py```
importsys
frompathlibimport Path
frompydantic_graphimport End
frompydantic_graph.persistence.fileimport FileStatePersistence
frompydantic_ai.messagesimport ModelMessage # noqa: F401
fromai_q_and_a_graphimport Ask, question_graph, Evaluate, QuestionState, Answer

async defmain():
  answer: str | None = sys.argv[2] if len(sys.argv)  2 else None 
  persistence = FileStatePersistence(Path('question_graph.json')) 
  persistence.set_graph_types(question_graph) 
  if snapshot := await persistence.load_next(): 
    state = snapshot.state
    assert answer is not None
    node = Evaluate(answer)
  else:
    state = QuestionState()
    node = Ask() 
  async with question_graph.iter(node, state=state, persistence=persistence) as run:
    while True:
      node = await run.next() 
      if isinstance(node, End): 
        print('END:', node.data)
        history = await persistence.load_all() 
        print([e.node for e in history])
        break
      elif isinstance(node, Answer): 
        print(node.question)
        #> What is the capital of France?
        break
      # otherwise just continue

```

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add`asyncio.run(main(answer))` to run `main`)_
For a complete example of this graph, see the [question graph example](https://ai.pydantic.dev/examples/question-graph/).
## Dependency Injection
As with PydanticAI, `pydantic-graph` supports dependency injection via a generic parameter on [`Graph`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph) and [`BaseNode`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode), and the [`GraphRunContext.deps`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.GraphRunContext.deps) field.
As an example of dependency injection, let's modify the `DivisibleBy5` example [above](https://ai.pydantic.dev/graph/#graph) to use a [`ProcessPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor) to run the compute load in a separate process (this is a contrived example, `ProcessPoolExecutor` wouldn't actually improve performance in this example):
deps_example.py```
from__future__import annotations
importasyncio
fromconcurrent.futuresimport ProcessPoolExecutor
fromdataclassesimport dataclass
frompydantic_graphimport BaseNode, End, Graph, GraphRunContext

@dataclass
classGraphDeps:
  executor: ProcessPoolExecutor

@dataclass
classDivisibleBy5(BaseNode[None, GraphDeps, int]):
  foo: int
  async defrun(
    self,
    ctx: GraphRunContext[None, GraphDeps],
  ) -> Increment | End[int]:
    if self.foo % 5 == 0:
      return End(self.foo)
    else:
      return Increment(self.foo)

@dataclass
classIncrement(BaseNode[None, GraphDeps]):
  foo: int
  async defrun(self, ctx: GraphRunContext[None, GraphDeps]) -> DivisibleBy5:
    loop = asyncio.get_running_loop()
    compute_result = await loop.run_in_executor(
      ctx.deps.executor,
      self.compute,
    )
    return DivisibleBy5(compute_result)
  defcompute(self) -> int:
    return self.foo + 1

fives_graph = Graph(nodes=[DivisibleBy5, Increment])

async defmain():
  with ProcessPoolExecutor() as executor:
    deps = GraphDeps(executor)
    result = await fives_graph.run(DivisibleBy5(3), deps=deps)
  print(result.output)
  #> 5
  # the full history is quite verbose (see below), so we'll just print the summary
  print([item.data_snapshot() for item in result.history])
"""

    DivisibleBy5(foo=3),
    Increment(foo=3),
    DivisibleBy5(foo=4),
    Increment(foo=4),
    DivisibleBy5(foo=5),
    End(data=5),

  """

```

_(This example is complete, it can be run "as is" with Python 3.10+ — you'll need to add`asyncio.run(main())` to run `main`)_
## Mermaid Diagrams
Pydantic Graph can generate [mermaid](https://mermaid.js.org/) [`stateDiagram-v2`](https://mermaid.js.org/syntax/stateDiagram.html) diagrams for graphs, as shown above.
These diagrams can be generated with:
  * [`Graph.mermaid_code`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_code) to generate the mermaid code for a graph
  * [`Graph.mermaid_image`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_image) to generate an image of the graph using [mermaid.ink](https://mermaid.ink/)
  * [`Graph.mermaid_save`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_save) to generate an image of the graph using [mermaid.ink](https://mermaid.ink/) and save it to a file


Beyond the diagrams shown above, you can also customize mermaid diagrams with the following options:
  * [`Edge`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.Edge) allows you to apply a label to an edge
  * [`BaseNode.docstring_notes`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.docstring_notes) and [`BaseNode.get_note`](https://ai.pydantic.dev/api/pydantic_graph/nodes/#pydantic_graph.nodes.BaseNode.get_note) allows you to add notes to nodes
  * The [`highlighted_nodes`](https://ai.pydantic.dev/api/pydantic_graph/graph/#pydantic_graph.graph.Graph.mermaid_code) parameter allows you to highlight specific node(s) in the diagram


Putting that together, we can edit the last [`ai_q_and_a_graph.py`](https://ai.pydantic.dev/graph/#example-human-in-the-loop) example to:
  * add labels to some edges
  * add a note to the `Ask` node
  * highlight the `Answer` node
  * save the diagram as a `PNG` image to file


ai_q_and_a_graph_extra.py```
...
fromtypingimport Annotated

frompydantic_graphimport BaseNode, End, Graph, GraphRunContext, Edge

...
@dataclass
classAsk(BaseNode[QuestionState]):
"""Generate question using GPT-4o."""
  docstring_notes = True
  async defrun(
    self, ctx: GraphRunContext[QuestionState]
  ) -> Annotated[Answer, Edge(label='Ask the question')]:
    ...
...
@dataclass
classEvaluate(BaseNode[QuestionState]):
  answer: str
  async defrun(
      self,
      ctx: GraphRunContext[QuestionState],
  ) -> Annotated[End[str], Edge(label='success')] | Reprimand:
    ...
...
question_graph.mermaid_save('image.png', highlighted_nodes=[Answer])

```

_(This example is not complete and cannot be run directly)_
This would generate an image that looks like this:
### Setting Direction of the State Diagram
You can specify the direction of the state diagram using one of the following values:
  * `'TB'`: Top to bottom, the diagram flows vertically from top to bottom.
  * `'LR'`: Left to right, the diagram flows horizontally from left to right.
  * `'RL'`: Right to left, the diagram flows horizontally from right to left.
  * `'BT'`: Bottom to top, the diagram flows vertically from bottom to top.


Here is an example of how to do this using 'Left to Right' (LR) instead of the default 'Top to Bottom' (TB): 
vending_machine_diagram.py```
fromvending_machineimport InsertCoin, vending_machine_graph
vending_machine_graph.mermaid_code(start_node=InsertCoin, direction='LR')

```


[ Skip to content ](https://ai.pydantic.dev/help/#getting-help)
# Getting Help
If you need help getting started with PydanticAI or with advanced usage, the following sources may be useful.
##  Slack
Join the `#pydantic-ai` channel in the [Pydantic Slack](https://logfire.pydantic.dev/docs/join-slack/) to ask questions, get help, and chat about PydanticAI. There's also channels for Pydantic, Logfire, and FastUI.
If you're on a [Logfire](https://pydantic.dev/logfire) Pro plan, you can also get a dedicated private slack collab channel with us.
##  GitHub Issues
The [PydanticAI GitHub Issues](https://github.com/pydantic/pydantic-ai/issues) are a great place to ask questions and give us feedback.

# Image, Audio & Document Input
Some LLMs are now capable of understanding both audio, image and document content.
## Image Input
Info
Some models do not support image input. Please check the model's documentation to confirm whether it supports image input.
If you have a direct URL for the image, you can use [`ImageUrl`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ImageUrl):
main.py```
frompydantic_aiimport Agent, ImageUrl
agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(
  [
    'What company is this logo from?',
    ImageUrl(url='https://iili.io/3Hs4FMg.png'),
  ]
)
print(result.data)
#> This is the logo for Pydantic, a data validation and settings management library in Python.

```

If you have the image locally, you can also use [`BinaryContent`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.BinaryContent):
main.py```
importhttpx
frompydantic_aiimport Agent, BinaryContent
image_response = httpx.get('https://iili.io/3Hs4FMg.png') # Pydantic logo
agent = Agent(model='openai:gpt-4o')
result = agent.run_sync(
  [
    'What company is this logo from?',
    BinaryContent(data=image_response.content, media_type='image/png'), 
  ]
)
print(result.data)
#> This is the logo for Pydantic, a data validation and settings management library in Python.

```

## Audio Input
Info
Some models do not support audio input. Please check the model's documentation to confirm whether it supports audio input.
You can provide audio input using either [`AudioUrl`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.AudioUrl) or [`BinaryContent`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.BinaryContent). The process is analogous to the examples above.
## Document Input
Info
Some models do not support document input. Please check the model's documentation to confirm whether it supports document input.
Warning
When using Gemini models, the document content will always be sent as binary data, regardless of whether you use `DocumentUrl` or `BinaryContent`. This is due to differences in how Vertex AI and Google AI handle document inputs.
For more details, see [this discussion](https://discuss.ai.google.dev/t/i-am-using-google-generative-ai-model-gemini-1-5-pro-for-image-analysis-but-getting-error/34866/4).
If you are unsatisfied with this behavior, please let us know by opening an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).
You can provide document input using either [`DocumentUrl`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.DocumentUrl) or [`BinaryContent`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.BinaryContent). The process is similar to the examples above.
If you have a direct URL for the document, you can use [`DocumentUrl`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.DocumentUrl):
main.py```
frompydantic_aiimport Agent, DocumentUrl
agent = Agent(model='anthropic:claude-3-sonnet')
result = agent.run_sync(
  [
    'What is the main content of this document?',
    DocumentUrl(url='https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf'),
  ]
)
print(result.data)
#> This document is the technical report introducing Gemini 1.5, Google's latest large language model...

```

The supported document formats vary by model.
You can also use [`BinaryContent`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.BinaryContent) to pass document data directly:
main.py```
frompathlibimport Path
frompydantic_aiimport Agent, BinaryContent
pdf_path = Path('document.pdf')
agent = Agent(model='anthropic:claude-3-sonnet')
result = agent.run_sync(
  [
    'What is the main content of this document?',
    BinaryContent(data=pdf_path.read_bytes(), media_type='application/pdf'),
  ]
)
print(result.data)
#> The document discusses...

```


[ Skip to content ](https://ai.pydantic.dev/install/#installation)
# Installation
PydanticAI is available on PyPI as [`pydantic-ai`](https://pypi.org/project/pydantic-ai/) so installation is as simple as:
```
pipinstallpydantic-ai

```

```
uvaddpydantic-ai

```

(Requires Python 3.9+)
This installs the `pydantic_ai` package, core dependencies, and libraries required to use all the models included in PydanticAI. If you want to use a specific model, you can install the ["slim"](https://ai.pydantic.dev/install/#slim-install) version of PydanticAI.
## Use with Pydantic Logfire
PydanticAI has an excellent (but completely optional) integration with [Pydantic Logfire](https://pydantic.dev/logfire) to help you view and understand agent runs.
To use Logfire with PydanticAI, install `pydantic-ai` or `pydantic-ai-slim` with the `logfire` optional group:
```
pipinstall"pydantic-ai[logfire]"

```

```
uvadd"pydantic-ai[logfire]"

```

From there, follow the [Logfire setup docs](https://ai.pydantic.dev/logfire/#using-logfire) to configure Logfire.
## Running Examples
We distribute the [`pydantic_ai_examples`](https://github.com/pydantic/pydantic-ai/tree/main/examples/pydantic_ai_examples) directory as a separate PyPI package ([`pydantic-ai-examples`](https://pypi.org/project/pydantic-ai-examples/)) to make examples extremely easy to customize and run.
To install examples, use the `examples` optional group:
```
pipinstall"pydantic-ai[examples]"

```

```
uvadd"pydantic-ai[examples]"

```

To run the examples, follow instructions in the [examples docs](https://ai.pydantic.dev/examples/).
## Slim Install
If you know which model you're going to use and want to avoid installing superfluous packages, you can use the [`pydantic-ai-slim`](https://pypi.org/project/pydantic-ai-slim/) package. For example, if you're using just [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel), you would run:
```
pipinstall"pydantic-ai-slim[openai]"

```

```
uvadd"pydantic-ai-slim[openai]"

```

`pydantic-ai-slim` has the following optional groups:
  * `logfire` — installs [`logfire`](https://ai.pydantic.dev/logfire/) [PyPI ↗](https://pypi.org/project/logfire)
  * `openai` — installs `openai` [PyPI ↗](https://pypi.org/project/openai)
  * `vertexai` — installs `google-auth` [PyPI ↗](https://pypi.org/project/google-auth) and `requests` [PyPI ↗](https://pypi.org/project/requests)
  * `anthropic` — installs `anthropic` [PyPI ↗](https://pypi.org/project/anthropic)
  * `groq` — installs `groq` [PyPI ↗](https://pypi.org/project/groq)
  * `mistral` — installs `mistralai` [PyPI ↗](https://pypi.org/project/mistralai)
  * `cohere` - installs `cohere` [PyPI ↗](https://pypi.org/project/cohere)
  * `duckduckgo` - installs `duckduckgo-search` [PyPI ↗](https://pypi.org/project/duckduckgo-search)
  * `tavily` - installs `tavily-python` [PyPI ↗](https://pypi.org/project/tavily-python)


See the [models](https://ai.pydantic.dev/models/) documentation for information on which optional dependencies are required for each model.
You can also install dependencies for multiple models and use cases, for example:
```
pipinstall"pydantic-ai-slim[openai,vertexai,logfire]"

```

```
uvadd"pydantic-ai-slim[openai,vertexai,logfire]"

```


# Debugging and Monitoring
Applications that use LLMs have some challenges that are well known and understood: LLMs are **slow** , **unreliable** and **expensive**.
These applications also have some challenges that most developers have encountered much less often: LLMs are **fickle** and **non-deterministic**. Subtle changes in a prompt can completely change a model's performance, and there's no `EXPLAIN` query you can run to understand why.
Warning
From a software engineers point of view, you can think of LLMs as the worst database you've ever heard of, but worse.
If LLMs weren't so bloody useful, we'd never touch them.
To build successful applications with LLMs, we need new tools to understand both model performance, and the behavior of applications that rely on them.
LLM Observability tools that just let you understand how your model is performing are useless: making API calls to an LLM is easy, it's building that into an application that's hard.
## Pydantic Logfire
[Pydantic Logfire](https://pydantic.dev/logfire) is an observability platform developed by the team who created and maintain Pydantic and PydanticAI. Logfire aims to let you understand your entire application: Gen AI, classic predictive AI, HTTP traffic, database queries and everything else a modern application needs.
Pydantic Logfire is a commercial product
Logfire is a commercially supported, hosted platform with an extremely generous and perpetual [free tier](https://pydantic.dev/pricing/). You can sign up and start using Logfire in a couple of minutes.
PydanticAI has built-in (but optional) support for Logfire. That means if the `logfire` package is installed and configured and agent instrumentation is enabled then detailed information about agent runs is sent to Logfire. Otherwise there's virtually no overhead and nothing is sent.
Here's an example showing details of running the [Weather Agent](https://ai.pydantic.dev/examples/weather-agent/) in Logfire:
## Using Logfire
To use logfire, you'll need a logfire [account](https://logfire.pydantic.dev), and logfire installed:
```
pipinstall"pydantic-ai[logfire]"

```

```
uvadd"pydantic-ai[logfire]"

```

Then authenticate your local environment with logfire:
```
logfireauth

```

```
uvrunlogfireauth

```

And configure a project to send data to:
```
logfireprojectsnew

```

```
uvrunlogfireprojectsnew

```

(Or use an existing project with `logfire projects use`)
Then add logfire to your code:
adding_logfire.py```
importlogfire
logfire.configure()

```

and enable instrumentation in your agent:
instrument_agent.py```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o', instrument=True)
# or instrument all agents to avoid needing to add `instrument=True` to each agent:
Agent.instrument_all()

```

The [logfire documentation](https://logfire.pydantic.dev/docs/) has more details on how to use logfire, including how to instrument other libraries like [Pydantic](https://logfire.pydantic.dev/docs/integrations/pydantic/), [HTTPX](https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/) and [FastAPI](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/).
Since Logfire is built on [OpenTelemetry](https://opentelemetry.io/), you can use the Logfire Python SDK to send data to any OpenTelemetry collector.
Once you have logfire set up, there are two primary ways it can help you understand your application:
  * **Debugging** — Using the live view to see what's happening in your application in real-time.
  * **Monitoring** — Using SQL and dashboards to observe the behavior of your application, Logfire is effectively a SQL database that stores information about how your application is running.


### Debugging
To demonstrate how Logfire can let you visualise the flow of a PydanticAI run, here's the view you get from Logfire while running the [chat app examples](https://ai.pydantic.dev/examples/chat-app/):
### Monitoring Performance
We can also query data with SQL in Logfire to monitor the performance of an application. Here's a real world example of using Logfire to monitor PydanticAI runs inside Logfire itself:
### Monitoring HTTPX Requests
In order to monitor HTTPX requests made by models, you can use `logfire`'s [HTTPX](https://logfire.pydantic.dev/docs/integrations/http-clients/httpx/) integration.
Instrumentation is as easy as adding the following three lines to your application:
instrument_httpx.py```
importlogfire
logfire.configure()
logfire.instrument_httpx(capture_all=True) 

```

In particular, this can help you to trace specific requests, responses, and headers:
instrument_httpx_example.py```
importlogfire
frompydantic_aiimport Agent
logfire.configure()
logfire.instrument_httpx(capture_all=True) 
agent = Agent('openai:gpt-4o', instrument=True)
result = agent.run_sync('What is the capital of France?')
print(result.data)
# > The capital of France is Paris.

```

[With `httpx` instrumentation](https://ai.pydantic.dev/logfire/#__tabbed_4_1)[Without `httpx` instrumentation](https://ai.pydantic.dev/logfire/#__tabbed_4_2)
Tip
`httpx` instrumentation might be of particular utility if you're using a custom `httpx` client in your model in order to get insights into your custom requests.
## Using OpenTelemetry
PydanticAI's instrumentation uses [OpenTelemetry](https://opentelemetry.io/), which Logfire is based on. You can use the Logfire SDK completely freely and follow the [Alternative backends](https://logfire.pydantic.dev/docs/how-to-guides/alternative-backends/) guide to send the data to any OpenTelemetry collector, such as a self-hosted Jaeger instance. Or you can skip Logfire entirely and use the OpenTelemetry Python SDK directly.
## Data format
PydanticAI follows the [OpenTelemetry Semantic Conventions for Generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/), with one caveat. The semantic conventions specify that messages should be captured as individual events (logs) that are children of the request span. By default, PydanticAI instead collects these events into a JSON array which is set as a single large attribute called `events` on the request span. To change this, use [`InstrumentationSettings(event_mode='logs')`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.InstrumentationSettings).
instrumentation_settings_event_mode.py```
frompydantic_aiimport Agent
frompydantic_ai.agentimport InstrumentationSettings
instrumentation_settings = InstrumentationSettings(event_mode='logs')
agent = Agent('openai:gpt-4o', instrument=instrumentation_settings)
# or instrument all agents:
Agent.instrument_all(instrumentation_settings)

```

For now, this won't look as good in the Logfire UI, but we're working on it.
If you have very long conversations, the `events` span attribute may be truncated. Using `event_mode='logs'` will help avoid this issue.
Note that the OpenTelemetry Semantic Conventions are still experimental and are likely to change.
## Setting OpenTelemetry SDK providers
By default, the global `TracerProvider` and `EventLoggerProvider` are used. These are set automatically by `logfire.configure()`. They can also be set by the `set_tracer_provider` and `set_event_logger_provider` functions in the OpenTelemetry Python SDK. You can set custom providers with [`InstrumentationSettings`](https://ai.pydantic.dev/api/models/instrumented/#pydantic_ai.models.instrumented.InstrumentationSettings).
instrumentation_settings_providers.py```
fromopentelemetry.sdk._eventsimport EventLoggerProvider
fromopentelemetry.sdk.traceimport TracerProvider
frompydantic_ai.agentimport InstrumentationSettings
instrumentation_settings = InstrumentationSettings(
  tracer_provider=TracerProvider(),
  event_logger_provider=EventLoggerProvider(),
)

```

## Instrumenting a specific `Model`
instrumented_model_example.py```
frompydantic_aiimport Agent
frompydantic_ai.models.instrumentedimport InstrumentationSettings, InstrumentedModel
settings = InstrumentationSettings()
model = InstrumentedModel('gpt-4o', settings)
agent = Agent(model)

```


# Messages and chat history
PydanticAI provides access to messages exchanged during an agent run. These messages can be used both to continue a coherent conversation, and to understand how an agent performed.
### Accessing Messages from Results
After running an agent, you can access the messages exchanged during that run from the `result` object.
Both [`RunResult`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult) (returned by [`Agent.run`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run), [`Agent.run_sync`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync)) and [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult) (returned by [`Agent.run_stream`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream)) have the following methods:
  * [`all_messages()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult.all_messages): returns all messages, including messages from prior runs. There's also a variant that returns JSON bytes, [`all_messages_json()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult.all_messages_json).
  * [`new_messages()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult.new_messages): returns only the messages from the current run. There's also a variant that returns JSON bytes, [`new_messages_json()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult.new_messages_json).


StreamedRunResult and complete messages
On [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult), the messages returned from these methods will only include the final result message once the stream has finished.
E.g. you've awaited one of the following coroutines:
  * [`StreamedRunResult.stream()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.stream)
  * [`StreamedRunResult.stream_text()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.stream_text)
  * [`StreamedRunResult.stream_structured()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.stream_structured)
  * [`StreamedRunResult.get_data()`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.get_data)


**Note:** The final result message will NOT be added to result messages if you use [`.stream_text(delta=True)`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult.stream_text) since in this case the result content is never built as one string.
Example of accessing methods on a [`RunResult`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult) :
run_result_messages.py```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
result = agent.run_sync('Tell me a joke.')
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
# all messages from the run
print(result.all_messages())
"""
[
  ModelRequest(
    parts=[
      SystemPromptPart(
        content='Be a helpful assistant.',
        timestamp=datetime.datetime(...),
        dynamic_ref=None,
        part_kind='system-prompt',

      UserPromptPart(
        content='Tell me a joke.',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',

    ],
    kind='request',
  ModelResponse(
    parts=[
      TextPart(
        content='Did you hear about the toothpaste scandal? They called it Colgate.',
        part_kind='text',

    ],
    model_name='gpt-4o',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
]
"""

```

_(This example is complete, it can be run "as is")_
Example of accessing methods on a [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult) :
streamed_run_result_messages.py```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

async defmain():
  async with agent.run_stream('Tell me a joke.') as result:
    # incomplete messages before the stream finishes
    print(result.all_messages())
"""

      ModelRequest(
        parts=[
          SystemPromptPart(
            content='Be a helpful assistant.',
            timestamp=datetime.datetime(...),
            dynamic_ref=None,
            part_kind='system-prompt',

          UserPromptPart(
            content='Tell me a joke.',
            timestamp=datetime.datetime(...),
            part_kind='user-prompt',


        kind='request',


    """
    async for text in result.stream_text():
      print(text)
      #> Did you hear
      #> Did you hear about the toothpaste
      #> Did you hear about the toothpaste scandal? They called
      #> Did you hear about the toothpaste scandal? They called it Colgate.
    # complete messages once the stream finishes
    print(result.all_messages())
"""

      ModelRequest(
        parts=[
          SystemPromptPart(
            content='Be a helpful assistant.',
            timestamp=datetime.datetime(...),
            dynamic_ref=None,
            part_kind='system-prompt',

          UserPromptPart(
            content='Tell me a joke.',
            timestamp=datetime.datetime(...),
            part_kind='user-prompt',


        kind='request',

      ModelResponse(
        parts=[
          TextPart(
            content='Did you hear about the toothpaste scandal? They called it Colgate.',
            part_kind='text',


        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
        kind='response',


    """

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
### Using Messages as Input for Further Agent Runs
The primary use of message histories in PydanticAI is to maintain context across multiple agent runs.
To use existing messages in a run, pass them to the `message_history` parameter of [`Agent.run`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run), [`Agent.run_sync`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_sync) or [`Agent.run_stream`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run_stream).
If `message_history` is set and not empty, a new system prompt is not generated — we assume the existing message history includes a system prompt.
Reusing messages in a conversation```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
result1 = agent.run_sync('Tell me a joke.')
print(result1.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.data)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.
print(result2.all_messages())
"""
[
  ModelRequest(
    parts=[
      SystemPromptPart(
        content='Be a helpful assistant.',
        timestamp=datetime.datetime(...),
        dynamic_ref=None,
        part_kind='system-prompt',

      UserPromptPart(
        content='Tell me a joke.',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',

    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='Did you hear about the toothpaste scandal? They called it Colgate.',
        part_kind='text',

    ],
    model_name='gpt-4o',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
  ModelRequest(
    parts=[
      UserPromptPart(
        content='Explain?',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',

    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.',
        part_kind='text',

    ],
    model_name='gpt-4o',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
]
"""

```

_(This example is complete, it can be run "as is")_
## Storing and loading messages (to JSON)
While maintaining conversation state in memory is enough for many applications, often times you may want to store the messages history of an agent run on disk or in a database. This might be for evals, for sharing data between Python and JavaScript/TypeScript, or any number of other use cases.
The intended way to do this is using a `TypeAdapter`.
We export [`ModelMessagesTypeAdapter`](https://ai.pydantic.dev/api/messages/#pydantic_ai.messages.ModelMessagesTypeAdapter) that can be used for this, or you can create your own.
Here's an example showing how:
serialize messages to json```
frompydantic_coreimport to_jsonable_python
frompydantic_aiimport Agent
frompydantic_ai.messagesimport ModelMessagesTypeAdapter 
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
result1 = agent.run_sync('Tell me a joke.')
history_step_1 = result1.all_messages()
as_python_objects = to_jsonable_python(history_step_1) 
same_history_as_step_1 = ModelMessagesTypeAdapter.validate_python(as_python_objects)
result2 = agent.run_sync( 
  'Tell me a different joke.', message_history=same_history_as_step_1
)

```

_(This example is complete, it can be run "as is")_
## Other ways of using messages
Since messages are defined by simple dataclasses, you can manually create and manipulate, e.g. for testing.
The message format is independent of the model used, so you can use messages in different agents, or the same agent with different models.
In the example below, we reuse the message from the first agent run, which uses the `openai:gpt-4o` model, in a second agent run using the `google-gla:gemini-1.5-pro` model.
Reusing messages with a different model```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')
result1 = agent.run_sync('Tell me a joke.')
print(result1.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
result2 = agent.run_sync(
  'Explain?',
  model='google-gla:gemini-1.5-pro',
  message_history=result1.new_messages(),
)
print(result2.data)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.
print(result2.all_messages())
"""
[
  ModelRequest(
    parts=[
      SystemPromptPart(
        content='Be a helpful assistant.',
        timestamp=datetime.datetime(...),
        dynamic_ref=None,
        part_kind='system-prompt',

      UserPromptPart(
        content='Tell me a joke.',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',

    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='Did you hear about the toothpaste scandal? They called it Colgate.',
        part_kind='text',

    ],
    model_name='gpt-4o',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
  ModelRequest(
    parts=[
      UserPromptPart(
        content='Explain?',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',

    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.',
        part_kind='text',

    ],
    model_name='gemini-1.5-pro',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
]
"""

```

## Examples
For a more complete example of using messages in conversations, see the [chat app](https://ai.pydantic.dev/examples/chat-app/) example.

# Models
PydanticAI is Model-agnostic and has built in support for the following model providers:
  * Gemini via two different APIs: [Generative Language API](https://ai.pydantic.dev/models/#gemini) and [VertexAI API](https://ai.pydantic.dev/models/#gemini-via-vertexai)


See [OpenAI-compatible models](https://ai.pydantic.dev/models/#openai-compatible-models) for more examples on how to use models such as [OpenRouter](https://ai.pydantic.dev/models/#openrouter), and [Grok (xAI)](https://ai.pydantic.dev/models/#grok-xai) that support the OpenAI SDK.
You can also [add support for other models](https://ai.pydantic.dev/models/#implementing-custom-models).
PydanticAI also comes with [`TestModel`](https://ai.pydantic.dev/api/models/test/) and [`FunctionModel`](https://ai.pydantic.dev/api/models/function/) for testing and development.
To use each model provider, you need to configure your local environment and make sure you have the right packages installed.
## Models, Interfaces, and Providers
PydanticAI uses a few key terms to describe how it interacts with different LLMs:
  * **Model** : This refers to the specific LLM model you want to handle your requests (e.g., `gpt-4o`, `claude-3-5-sonnet-latest`, `gemini-1.5-flash`). It's the "brain" that processes your prompts and generates responses. You specify the _Model_ as a parameter to the _Interface_.
  * **Interface** : This refers to a PydanticAI class used to make requests following a specific LLM API (generally by wrapping a vendor-provided SDK, like the `openai` python SDK). These classes implement a vendor-SDK-agnostic API, ensuring a single PydanticAI agent is portable to different LLM vendors without any other code changes just by swapping out the _Interface_ it uses. Currently, interface classes are named roughly in the format `<VendorSdk>Model`, for example, we have `OpenAIModel`, `AnthropicModel`, `GeminiModel`, etc. These `Model` classes will soon be renamed to `<VendorSdk>Interface` to reflect this terminology better.
  * **Provider** : This refers to _Interface_ -specific classes which handle the authentication and connections to an LLM vendor. Passing a non-default _Provider_ as a parameter to an _Interface_ is how you can ensure that your agent will make requests to a specific endpoint, or make use of a specific approach to authentication (e.g., you can use Vertex-specific auth with the `GeminiModel` by way of the `VertexProvider`). In particular, this is how you can make use of an AI gateway, or an LLM vendor that offers API compatibility with the vendor SDK used by an existing interface (such as `OpenAIModel`).


In short, you select a _model_ , PydanticAI uses the appropriate _interface_ class, and the _provider_ handles the connection and authentication to the underlying service.
## OpenAI
### Install
To use OpenAI models, you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `openai` optional group:
```
pipinstall"pydantic-ai-slim[openai]"

```

```
uvadd"pydantic-ai-slim[openai]"

```

### Configuration
To use [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel) through their main API, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find the place to generate an API key.
### Environment variable
Once you have the API key, you can set it as an environment variable:
```
exportOPENAI_API_KEY='your-api-key'

```

You can then use [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel) by name:
openai_model_by_name.py```
frompydantic_aiimport Agent
agent = Agent('openai:gpt-4o')
...

```

Or initialise the model directly with just the model name:
openai_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
model = OpenAIModel('gpt-4o')
agent = Agent(model)
...

```

By default, the `OpenAIModel` uses the [`OpenAIProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.openai.OpenAIProvider.__init__) with the `base_url` set to `https://api.openai.com/v1`. 
### `provider` argument
You can provide a custom [`Provider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider) via the [`provider` argument](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel.__init__):
openai_model_provider.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key='your-api-key'))
agent = Agent(model)
...

```

### Custom OpenAI Client
`OpenAIProvider` also accepts a custom `AsyncOpenAI` client via the [`openai_client` parameter](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.openai.OpenAIProvider.__init__), so you can customise the `organization`, `project`, `base_url` etc. as defined in the [OpenAI API docs](https://platform.openai.com/docs/api-reference).
You could also use the [`AsyncAzureOpenAI`](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints) client to use the Azure OpenAI API.
openai_azure.py```
fromopenaiimport AsyncAzureOpenAI
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
client = AsyncAzureOpenAI(
  azure_endpoint='...',
  api_version='2024-07-01-preview',
  api_key='your-api-key',
)
model = OpenAIModel(
  'gpt-4o',
  provider=OpenAIProvider(openai_client=client),
)
agent = Agent(model)
...

```

## Anthropic
### Install
To use [`AnthropicModel`](https://ai.pydantic.dev/api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel) models, you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `anthropic` optional group:
```
pipinstall"pydantic-ai-slim[anthropic]"

```

```
uvadd"pydantic-ai-slim[anthropic]"

```

### Configuration
To use [Anthropic](https://anthropic.com) through their API, go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) to generate an API key.
[`AnthropicModelName`](https://ai.pydantic.dev/api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModelName) contains a list of available Anthropic models.
### Environment variable
Once you have the API key, you can set it as an environment variable:
```
exportANTHROPIC_API_KEY='your-api-key'

```

You can then use [`AnthropicModel`](https://ai.pydantic.dev/api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel) by name:
anthropic_model_by_name.py```
frompydantic_aiimport Agent
agent = Agent('anthropic:claude-3-5-sonnet-latest')
...

```

Or initialise the model directly with just the model name:
anthropic_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.anthropicimport AnthropicModel
model = AnthropicModel('claude-3-5-sonnet-latest')
agent = Agent(model)
...

```

### `provider` argument
You can provide a custom [`Provider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider) via the [`provider` argument](https://ai.pydantic.dev/api/models/anthropic/#pydantic_ai.models.anthropic.AnthropicModel.__init__):
anthropic_model_provider.py```
frompydantic_aiimport Agent
frompydantic_ai.models.anthropicimport AnthropicModel
frompydantic_ai.providers.anthropicimport AnthropicProvider
model = AnthropicModel(
  'claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key='your-api-key')
)
agent = Agent(model)
...

```

### Custom HTTP Client
You can customize the `AnthropicProvider` with a custom `httpx.AsyncClient`:
anthropic_model_custom_provider.py```
fromhttpximport AsyncClient
frompydantic_aiimport Agent
frompydantic_ai.models.anthropicimport AnthropicModel
frompydantic_ai.providers.anthropicimport AnthropicProvider
custom_http_client = AsyncClient(timeout=30)
model = AnthropicModel(
  'claude-3-5-sonnet-latest',
  provider=AnthropicProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...

```

## Gemini
### Install
To use [`GeminiModel`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) models, you just need to install [`pydantic-ai`](https://ai.pydantic.dev/install/) or [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install), no extra dependencies are required.
### Configuration
[`GeminiModel`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) let's you use the Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods), `generativelanguage.googleapis.com`.
[`GeminiModelName`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModelName) contains a list of available Gemini models that can be used through this interface.
To use `GeminiModel`, go to [aistudio.google.com](https://aistudio.google.com/apikey) and select "Create API key".
### Environment variable
Once you have the API key, you can set it as an environment variable:
```
exportGEMINI_API_KEY=your-api-key

```

You can then use [`GeminiModel`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) by name:
gemini_model_by_name.py```
frompydantic_aiimport Agent
agent = Agent('google-gla:gemini-2.0-flash')
...

```

Note
The `google-gla` provider prefix represents the [Google **G** enerative **L** anguage **A** PI](https://ai.google.dev/api/all-methods) for `GeminiModel`s. `google-vertex` is used with [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).
Or initialise the model directly with just the model name and provider:
gemini_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.geminiimport GeminiModel
model = GeminiModel('gemini-2.0-flash', provider='google-gla')
agent = Agent(model)
...

```

### `provider` argument
You can provide a custom [`Provider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider) via the [`provider` argument](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel.__init__):
gemini_model_provider.py```
frompydantic_aiimport Agent
frompydantic_ai.models.geminiimport GeminiModel
frompydantic_ai.providers.google_glaimport GoogleGLAProvider
model = GeminiModel(
  'gemini-2.0-flash', provider=GoogleGLAProvider(api_key='your-api-key')
)
agent = Agent(model)
...

```

You can also customize the `GoogleGLAProvider` with a custom `http_client`: 
gemini_model_custom_provider.py```
fromhttpximport AsyncClient
frompydantic_aiimport Agent
frompydantic_ai.models.geminiimport GeminiModel
frompydantic_ai.providers.google_glaimport GoogleGLAProvider
custom_http_client = AsyncClient(timeout=30)
model = GeminiModel(
  'gemini-2.0-flash',
  provider=GoogleGLAProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...

```

## Gemini via VertexAI
If you are an enterprise user, you should use the `google-vertex` provider with [`GeminiModel`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel) which uses the `*-aiplatform.googleapis.com` API.
[`GeminiModelName`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModelName) contains a list of available Gemini models that can be used through this interface.
### Install
To use the `google-vertex` provider with [`GeminiModel`](https://ai.pydantic.dev/api/models/gemini/#pydantic_ai.models.gemini.GeminiModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `vertexai` optional group:
```
pipinstall"pydantic-ai-slim[vertexai]"

```

```
uvadd"pydantic-ai-slim[vertexai]"

```

### Configuration
This interface has a number of advantages over `generativelanguage.googleapis.com` documented above:
  1. The VertexAI API comes with more enterprise readiness guarantees.
  2. You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with VertexAI to guarantee capacity.
  3. If you're running PydanticAI inside GCP, you don't need to set up authentication, it should "just work".
  4. You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.


The big disadvantage is that for local development you may need to create and configure a "service account", which I've found extremely painful to get right in the past.
Whichever way you authenticate, you'll need to have VertexAI enabled in your GCP account.
### Application default credentials
Luckily if you're running PydanticAI inside GCP, or you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you should be able to use `VertexAIModel` without any additional setup.
To use `VertexAIModel`, with [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) configured (e.g. with `gcloud`), you can simply use:
vertexai_application_default_credentials.py```
frompydantic_aiimport Agent
frompydantic_ai.models.geminiimport GeminiModel
model = GeminiModel('gemini-2.0-flash', provider='google-vertex')
agent = Agent(model)
...

```

Internally this uses [`google.auth.default()`](https://google-auth.readthedocs.io/en/master/reference/google.auth.html) from the `google-auth` package to obtain credentials.
Won't fail until `agent.run()`
Because `google.auth.default()` requires network requests and can be slow, it's not run until you call `agent.run()`.
You may also need to pass the [`project_id` argument to `GoogleVertexProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.google_vertex.GoogleVertexProvider) if application default credentials don't set a project, if you pass `project_id` and it conflicts with the project set by application default credentials, an error is raised.
### Service account
If instead of application default credentials, you want to authenticate with a service account, you'll need to create a service account, add it to your GCP project (note: AFAIK this step is necessary even if you created the service account within the project), give that service account the "Vertex AI Service Agent" role, and download the service account JSON file.
Once you have the JSON file, you can use it thus:
vertexai_service_account.py```
frompydantic_aiimport Agent
frompydantic_ai.models.geminiimport GeminiModel
frompydantic_ai.providers.google_verteximport GoogleVertexProvider
model = GeminiModel(
  'gemini-2.0-flash',
  provider=GoogleVertexProvider(service_account_file='path/to/service-account.json'),
)
agent = Agent(model)
...

```

Alternatively, if you already have the service account information in memory, you can pass it as a dictionary:
vertexai_service_account.py```
importjson
frompydantic_aiimport Agent
frompydantic_ai.models.geminiimport GeminiModel
frompydantic_ai.providers.google_verteximport GoogleVertexProvider
service_account_info = json.loads(
  '{"type": "service_account", "project_id": "my-project-id"}'
model = GeminiModel(
  'gemini-2.0-flash',
  provider=GoogleVertexProvider(service_account_info=service_account_info),
)
agent = Agent(model)
...

```

### Customising region
Whichever way you authenticate, you can specify which region requests will be sent to via the [`region` argument](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.google_vertex.GoogleVertexProvider).
Using a region close to your application can improve latency and might be important from a regulatory perspective.
vertexai_region.py```
frompydantic_aiimport Agent
frompydantic_ai.models.geminiimport GeminiModel
frompydantic_ai.providers.google_verteximport GoogleVertexProvider
model = GeminiModel(
  'gemini-2.0-flash', provider=GoogleVertexProvider(region='asia-east1')
)
agent = Agent(model)
...

```

You can also customize the `GoogleVertexProvider` with a custom `http_client`: 
vertexai_custom_provider.py```
fromhttpximport AsyncClient
frompydantic_aiimport Agent
frompydantic_ai.models.geminiimport GeminiModel
frompydantic_ai.providers.google_verteximport GoogleVertexProvider
custom_http_client = AsyncClient(timeout=30)
model = GeminiModel(
  'gemini-2.0-flash',
  provider=GoogleVertexProvider(region='asia-east1', http_client=custom_http_client),
)
agent = Agent(model)
...

```

## Groq
### Install
To use [`GroqModel`](https://ai.pydantic.dev/api/models/groq/#pydantic_ai.models.groq.GroqModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `groq` optional group:
```
pipinstall"pydantic-ai-slim[groq]"

```

```
uvadd"pydantic-ai-slim[groq]"

```

### Configuration
To use [Groq](https://groq.com/) through their API, go to [console.groq.com/keys](https://console.groq.com/keys) and follow your nose until you find the place to generate an API key.
[`GroqModelName`](https://ai.pydantic.dev/api/models/groq/#pydantic_ai.models.groq.GroqModelName) contains a list of available Groq models.
### Environment variable
Once you have the API key, you can set it as an environment variable:
```
exportGROQ_API_KEY='your-api-key'

```

You can then use [`GroqModel`](https://ai.pydantic.dev/api/models/groq/#pydantic_ai.models.groq.GroqModel) by name:
groq_model_by_name.py```
frompydantic_aiimport Agent
agent = Agent('groq:llama-3.3-70b-versatile')
...

```

Or initialise the model directly with just the model name:
groq_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.groqimport GroqModel
model = GroqModel('llama-3.3-70b-versatile')
agent = Agent(model)
...

```

### `provider` argument
You can provide a custom [`Provider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider) via the [`provider` argument](https://ai.pydantic.dev/api/models/groq/#pydantic_ai.models.groq.GroqModel.__init__):
groq_model_provider.py```
frompydantic_aiimport Agent
frompydantic_ai.models.groqimport GroqModel
frompydantic_ai.providers.groqimport GroqProvider
model = GroqModel(
  'llama-3.3-70b-versatile', provider=GroqProvider(api_key='your-api-key')
)
agent = Agent(model)
...

```

You can also customize the [`GroqProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.groq.GroqProvider) with a custom `httpx.AsyncHTTPClient`:
groq_model_custom_provider.py```
fromhttpximport AsyncClient
frompydantic_aiimport Agent
frompydantic_ai.models.groqimport GroqModel
frompydantic_ai.providers.groqimport GroqProvider
custom_http_client = AsyncClient(timeout=30)
model = GroqModel(
  'llama-3.3-70b-versatile',
  provider=GroqProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...

```

## Mistral
### Install
To use [`MistralModel`](https://ai.pydantic.dev/api/models/mistral/#pydantic_ai.models.mistral.MistralModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `mistral` optional group:
```
pipinstall"pydantic-ai-slim[mistral]"

```

```
uvadd"pydantic-ai-slim[mistral]"

```

### Configuration
To use [Mistral](https://mistral.ai) through their API, go to [console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/) and follow your nose until you find the place to generate an API key.
[`LatestMistralModelNames`](https://ai.pydantic.dev/api/models/mistral/#pydantic_ai.models.mistral.LatestMistralModelNames) contains a list of the most popular Mistral models.
### Environment variable
Once you have the API key, you can set it as an environment variable:
```
exportMISTRAL_API_KEY='your-api-key'

```

You can then use [`MistralModel`](https://ai.pydantic.dev/api/models/mistral/#pydantic_ai.models.mistral.MistralModel) by name:
mistral_model_by_name.py```
frompydantic_aiimport Agent
agent = Agent('mistral:mistral-large-latest')
...

```

Or initialise the model directly with just the model name:
mistral_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.mistralimport MistralModel
model = MistralModel('mistral-small-latest')
agent = Agent(model)
...

```

### `provider` argument
You can provide a custom [`Provider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider) via the [`provider` argument](https://ai.pydantic.dev/api/models/mistral/#pydantic_ai.models.mistral.MistralModel.__init__):
groq_model_provider.py```
frompydantic_aiimport Agent
frompydantic_ai.models.mistralimport MistralModel
frompydantic_ai.providers.mistralimport MistralProvider
model = MistralModel(
  'mistral-large-latest', provider=MistralProvider(api_key='your-api-key')
)
agent = Agent(model)
...

```

You can also customize the provider with a custom `httpx.AsyncHTTPClient`:
groq_model_custom_provider.py```
fromhttpximport AsyncClient
frompydantic_aiimport Agent
frompydantic_ai.models.mistralimport MistralModel
frompydantic_ai.providers.mistralimport MistralProvider
custom_http_client = AsyncClient(timeout=30)
model = MistralModel(
  'mistral-large-latest',
  provider=MistralProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...

```

## Cohere
### Install
To use [`CohereModel`](https://ai.pydantic.dev/api/models/cohere/#pydantic_ai.models.cohere.CohereModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `cohere` optional group:
```
pipinstall"pydantic-ai-slim[cohere]"

```

```
uvadd"pydantic-ai-slim[cohere]"

```

### Configuration
To use [Cohere](https://cohere.com/) through their API, go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and follow your nose until you find the place to generate an API key.
[`CohereModelName`](https://ai.pydantic.dev/api/models/cohere/#pydantic_ai.models.cohere.CohereModelName) contains a list of the most popular Cohere models.
### Environment variable
Once you have the API key, you can set it as an environment variable:
```
exportCO_API_KEY='your-api-key'

```

You can then use [`CohereModel`](https://ai.pydantic.dev/api/models/cohere/#pydantic_ai.models.cohere.CohereModel) by name:
cohere_model_by_name.py```
frompydantic_aiimport Agent
agent = Agent('cohere:command')
...

```

Or initialise the model directly with just the model name:
cohere_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.cohereimport CohereModel
model = CohereModel('command')
agent = Agent(model)
...

```

### `provider` argument
You can provide a custom [`Provider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.Provider) via the [`provider` argument](https://ai.pydantic.dev/api/models/cohere/#pydantic_ai.models.cohere.CohereModel.__init__):
cohere_model_provider.py```
frompydantic_aiimport Agent
frompydantic_ai.models.cohereimport CohereModel
frompydantic_ai.providers.cohereimport CohereProvider
model = CohereModel('command', provider=CohereProvider(api_key='your-api-key'))
agent = Agent(model)
...

```

You can also customize the `CohereProvider` with a custom `http_client`:
cohere_model_custom_provider.py```
fromhttpximport AsyncClient
frompydantic_aiimport Agent
frompydantic_ai.models.cohereimport CohereModel
frompydantic_ai.providers.cohereimport CohereProvider
custom_http_client = AsyncClient(timeout=30)
model = CohereModel(
  'command',
  provider=CohereProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...

```

## Bedrock
### Install
To use [`BedrockConverseModel`](https://ai.pydantic.dev/api/models/bedrock/#pydantic_ai.models.bedrock.BedrockConverseModel), you need to either install [`pydantic-ai`](https://ai.pydantic.dev/install/), or install [`pydantic-ai-slim`](https://ai.pydantic.dev/install/#slim-install) with the `bedrock` optional group:
```
pipinstall"pydantic-ai-slim[bedrock]"

```

```
uvadd"pydantic-ai-slim[bedrock]"

```

### Configuration
To use [AWS Bedrock](https://aws.amazon.com/bedrock/), you'll need an AWS account with Bedrock enabled and appropriate credentials. You can use either AWS credentials directly or a pre-configured boto3 client.
[`BedrockModelName`](https://ai.pydantic.dev/api/models/bedrock/#pydantic_ai.models.bedrock.BedrockModelName) contains a list of available Bedrock models, including models from Anthropic, Amazon, Cohere, Meta, and Mistral.
### Environment variables
You can set your AWS credentials as environment variables ([among other options](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables):
```
exportAWS_ACCESS_KEY_ID='your-access-key'
exportAWS_SECRET_ACCESS_KEY='your-secret-key'
exportAWS_DEFAULT_REGION='us-east-1'# or your preferred region

```

You can then use [`BedrockConverseModel`](https://ai.pydantic.dev/api/models/bedrock/#pydantic_ai.models.bedrock.BedrockConverseModel) by name:
bedrock_model_by_name.py```
frompydantic_aiimport Agent
agent = Agent('bedrock:anthropic.claude-3-sonnet-20240229-v1:0')
...

```

Or initialize the model directly with just the model name:
bedrock_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.bedrockimport BedrockConverseModel
model = BedrockConverseModel('anthropic.claude-3-sonnet-20240229-v1:0')
agent = Agent(model)
...

```

### `provider` argument
You can provide a custom [`BedrockProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.bedrock.BedrockProvider) via the [`provider` argument](https://ai.pydantic.dev/api/models/bedrock/#pydantic_ai.models.bedrock.BedrockConverseModel.__init__). This is useful when you want to specify credentials directly or use a custom boto3 client:
bedrock_model_provider.py```
frompydantic_aiimport Agent
frompydantic_ai.models.bedrockimport BedrockConverseModel
frompydantic_ai.providers.bedrockimport BedrockProvider
# Using AWS credentials directly
model = BedrockConverseModel(
  'anthropic.claude-3-sonnet-20240229-v1:0',
  provider=BedrockProvider(
    region_name='us-east-1',
    aws_access_key_id='your-access-key',
    aws_secret_access_key='your-secret-key',
  ),
)
agent = Agent(model)
...

```

You can also pass a pre-configured boto3 client:
bedrock_model_boto3.py```
importboto3
frompydantic_aiimport Agent
frompydantic_ai.models.bedrockimport BedrockConverseModel
frompydantic_ai.providers.bedrockimport BedrockProvider
# Using a pre-configured boto3 client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
model = BedrockConverseModel(
  'anthropic.claude-3-sonnet-20240229-v1:0',
  provider=BedrockProvider(bedrock_client=bedrock_client),
)
agent = Agent(model)
...

```

## OpenAI-compatible Models
Many of the models are compatible with OpenAI API, and thus can be used with [`OpenAIModel`](https://ai.pydantic.dev/api/models/openai/#pydantic_ai.models.openai.OpenAIModel) in PydanticAI. Before getting started, check the [OpenAI](https://ai.pydantic.dev/models/#openai) section for installation and configuration instructions.
To use another OpenAI-compatible API, you can make use of the [`base_url`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.openai.OpenAIProvider.__init__) and [`api_key`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.openai.OpenAIProvider.__init__) arguments from `OpenAIProvider`:
deepseek_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
model = OpenAIModel(
  'model_name',
  provider=OpenAIProvider(
    base_url='https://<openai-compatible-api-endpoint>.com', api_key='your-api-key'
  ),
)
agent = Agent(model)
...

```

You can also use the `provider` argument with a custom provider class like the [`DeepSeekProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.deepseek.DeepSeekProvider):
deepseek_model_init_provider_class.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.deepseekimport DeepSeekProvider
model = OpenAIModel(
  'deepseek-chat',
  provider=DeepSeekProvider(api_key='your-deepseek-api-key'),
)
agent = Agent(model)
...

```

You can also customize any provider with a custom `http_client`: 
deepseek_model_init_provider_custom.py```
fromhttpximport AsyncClient
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.deepseekimport DeepSeekProvider
custom_http_client = AsyncClient(timeout=30)
model = OpenAIModel(
  'deepseek-chat',
  provider=DeepSeekProvider(
    api_key='your-deepseek-api-key', http_client=custom_http_client
  ),
)
agent = Agent(model)
...

```

### Ollama
To use [Ollama](https://ollama.com/), you must first download the Ollama client, and then download a model using the [Ollama model library](https://ollama.com/library).
You must also ensure the Ollama server is running when trying to make requests to it. For more information, please see the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs).
#### Example local usage
With `ollama` installed, you can run the server with the model you want to use:
terminal-run-ollama```
ollamarunllama3.2

```

(this will pull the `llama3.2` model if you don't already have it downloaded)
Then run your code, here's a minimal example:
ollama_example.py```
frompydanticimport BaseModel
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider

classCityLocation(BaseModel):
  city: str
  country: str

ollama_model = OpenAIModel(
  model_name='llama3.2', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
agent = Agent(ollama_model, result_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""

```

#### Example using a remote server
ollama_example_with_remote_server.py```
frompydanticimport BaseModel
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
ollama_model = OpenAIModel(
  model_name='qwen2.5-coder:7b', 
  provider=OpenAIProvider(base_url='http://192.168.1.74:11434/v1'), 
)

classCityLocation(BaseModel):
  city: str
  country: str

agent = Agent(model=ollama_model, result_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""

```

### Azure AI Foundry
If you want to use [Azure AI Foundry](https://ai.azure.com/) as your provider, you can do so by using the [`AzureProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.azure.AzureProvider) class.
azure_provider_example.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.azureimport AzureProvider
model = OpenAIModel(
  'gpt-4o',
  provider=AzureProvider(
    azure_endpoint='your-azure-endpoint',
    api_version='your-api-version',
    api_key='your-api-key',
  ),
)
agent = Agent(model)
...

```

### OpenRouter
To use [OpenRouter](https://openrouter.ai), first create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).
Once you have the API key, you can use it with the [`OpenAIProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.openai.OpenAIProvider):
openrouter_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
model = OpenAIModel(
  'anthropic/claude-3.5-sonnet',
  provider=OpenAIProvider(
    base_url='https://openrouter.ai/api/v1',
    api_key='your-openrouter-api-key',
  ),
)
agent = Agent(model)
...

```

### Grok (xAI)
Go to [xAI API Console](https://console.x.ai/) and create an API key. Once you have the API key, you can use it with the [`OpenAIProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.openai.OpenAIProvider):
grok_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
model = OpenAIModel(
  'grok-2-1212',
  provider=OpenAIProvider(base_url='https://api.x.ai/v1', api_key='your-xai-api-key'),
)
agent = Agent(model)
...

```

### Perplexity
Follow the Perplexity [getting started](https://docs.perplexity.ai/guides/getting-started) guide to create an API key. Then, you can query the Perplexity API with the following:
perplexity_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
model = OpenAIModel(
  'sonar-pro',
  provider=OpenAIProvider(
    base_url='https://api.perplexity.ai',
    api_key='your-perplexity-api-key',
  ),
)
agent = Agent(model)
...

```

### Fireworks AI
Go to [Fireworks.AI](https://fireworks.ai/) and create an API key in your account settings. Once you have the API key, you can use it with the [`OpenAIProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.openai.OpenAIProvider):
fireworks_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
model = OpenAIModel(
  'accounts/fireworks/models/qwq-32b', # model library available at https://fireworks.ai/models
  provider=OpenAIProvider(
    base_url='https://api.fireworks.ai/inference/v1',
    api_key='your-fireworks-api-key',
  ),
)
agent = Agent(model)
...

```

### Together AI
Go to [Together.ai](https://www.together.ai/) and create an API key in your account settings. Once you have the API key, you can use it with the [`OpenAIProvider`](https://ai.pydantic.dev/api/providers/#pydantic_ai.providers.openai.OpenAIProvider):
together_model_init.py```
frompydantic_aiimport Agent
frompydantic_ai.models.openaiimport OpenAIModel
frompydantic_ai.providers.openaiimport OpenAIProvider
model = OpenAIModel(
  'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', # model library available at https://www.together.ai/models
  provider=OpenAIProvider(
    base_url='https://api.together.xyz/v1',
    api_key='your-together-api-key',
  ),
)
agent = Agent(model)
...

```

## Implementing Custom Models
To implement support for models not already supported, you will need to subclass the [`Model`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.Model) abstract base class.
For streaming, you'll also need to implement the following abstract base class:


The best place to start is to review the source code for existing implementations, e.g. [`OpenAIModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).
For details on when we'll accept contributions adding new models to PydanticAI, see the [contributing guidelines](https://ai.pydantic.dev/contributing/#new-model-rules).
## Fallback
You can use [`FallbackModel`](https://ai.pydantic.dev/api/models/fallback/#pydantic_ai.models.fallback.FallbackModel) to attempt multiple models in sequence until one returns a successful result. Under the hood, PydanticAI automatically switches from one model to the next if the current model returns a 4xx or 5xx status code.
In the following example, the agent first makes a request to the OpenAI model (which fails due to an invalid API key), and then falls back to the Anthropic model.
fallback_model.py```
frompydantic_aiimport Agent
frompydantic_ai.models.anthropicimport AnthropicModel
frompydantic_ai.models.fallbackimport FallbackModel
frompydantic_ai.models.openaiimport OpenAIModel
openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)
agent = Agent(fallback_model)
response = agent.run_sync('What is the capital of France?')
print(response.data)
#> Paris
print(response.all_messages())
"""
[
  ModelRequest(
    parts=[
      UserPromptPart(
        content='What is the capital of France?',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',

    ],
    kind='request',
  ),
  ModelResponse(
    parts=[TextPart(content='Paris', part_kind='text')],
    model_name='claude-3-5-sonnet-latest',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
]
"""

```

The `ModelResponse` message above indicates in the `model_name` field that the result was returned by the Anthropic model, which is the second model specified in the `FallbackModel`.
Note
Each model's options should be configured individually. For example, `base_url`, `api_key`, and custom clients should be set on each model itself, not on the `FallbackModel`.
In this next example, we demonstrate the exception-handling capabilities of `FallbackModel`. If all models fail, a [`FallbackExceptionGroup`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.FallbackExceptionGroup) is raised, which contains all the exceptions encountered during the `run` execution.
[Python >=3.11](https://ai.pydantic.dev/models/#__tabbed_8_1)
fallback_model_failure.py```
frompydantic_aiimport Agent
frompydantic_ai.exceptionsimport ModelHTTPError
frompydantic_ai.models.anthropicimport AnthropicModel
frompydantic_ai.models.fallbackimport FallbackModel
frompydantic_ai.models.openaiimport OpenAIModel
openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)
agent = Agent(fallback_model)
try:
  response = agent.run_sync('What is the capital of France?')
except* ModelHTTPError as exc_group:
  for exc in exc_group.exceptions:
    print(exc)

```

Since [`except*`](https://docs.python.org/3/reference/compound_stmts.html#except-star) is only supported in Python 3.11+, we use the [`exceptiongroup`](https://github.com/agronholm/exceptiongroup) backport package for earlier Python versions:
fallback_model_failure.py```
fromexceptiongroupimport catch
frompydantic_aiimport Agent
frompydantic_ai.exceptionsimport ModelHTTPError
frompydantic_ai.models.anthropicimport AnthropicModel
frompydantic_ai.models.fallbackimport FallbackModel
frompydantic_ai.models.openaiimport OpenAIModel

defmodel_status_error_handler(exc_group: BaseExceptionGroup) -> None:
  for exc in exc_group.exceptions:
    print(exc)

openai_model = OpenAIModel('gpt-4o')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)
agent = Agent(fallback_model)
with catch({ModelHTTPError: model_status_error_handler}):
  response = agent.run_sync('What is the capital of France?')

```

By default, the `FallbackModel` only moves on to the next model if the current model raises a [`ModelHTTPError`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.ModelHTTPError). You can customize this behavior by passing a custom `fallback_on` argument to the `FallbackModel` constructor.

# Multi-agent Applications
There are roughly four levels of complexity when building applications with PydanticAI:
  1. Single agent workflows — what most of the `pydantic_ai` documentation covers
  2. [Agent delegation](https://ai.pydantic.dev/multi-agent-applications/#agent-delegation) — agents using another agent via tools
  3. [Programmatic agent hand-off](https://ai.pydantic.dev/multi-agent-applications/#programmatic-agent-hand-off) — one agent runs, then application code calls another agent
  4. [Graph based control flow](https://ai.pydantic.dev/graph/) — for the most complex cases, a graph-based state machine can be used to control the execution of multiple agents


Of course, you can combine multiple strategies in a single application.
## Agent delegation
"Agent delegation" refers to the scenario where an agent delegates work to another agent, then takes back control when the delegate agent (the agent called from within a tool) finishes.
Since agents are stateless and designed to be global, you do not need to include the agent itself in agent [dependencies](https://ai.pydantic.dev/dependencies/).
You'll generally want to pass [`ctx.usage`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext.usage) to the [`usage`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.run) keyword argument of the delegate agent run so usage within that run counts towards the total usage of the parent agent run.
Multiple models
Agent delegation doesn't need to use the same model for each agent. If you choose to use different models within a run, calculating the monetary cost from the final [`result.usage()`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult.usage) of the run will not be possible, but you can still use [`UsageLimits`](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.UsageLimits) to avoid unexpected costs.
agent_delegation_simple.py```
frompydantic_aiimport Agent, RunContext
frompydantic_ai.usageimport UsageLimits
joke_selection_agent = Agent( 
  'openai:gpt-4o',
  system_prompt=(
    'Use the `joke_factory` to generate some jokes, then choose the best. '
    'You must return just a single joke.'
  ),
)
joke_generation_agent = Agent( 
  'google-gla:gemini-1.5-flash', result_type=list[str]
)

@joke_selection_agent.tool
async defjoke_factory(ctx: RunContext[None], count: int) -> list[str]:
  r = await joke_generation_agent.run( 
    f'Please generate {count} jokes.',
    usage=ctx.usage, 
  )
  return r.data 

result = joke_selection_agent.run_sync(
  'Tell me a joke.',
  usage_limits=UsageLimits(request_limit=5, total_tokens_limit=300),
)
print(result.data)
#> Did you hear about the toothpaste scandal? They called it Colgate.
print(result.usage())
"""
Usage(
  requests=3, request_tokens=204, response_tokens=24, total_tokens=228, details=None
)
"""

```

_(This example is complete, it can be run "as is")_
The control flow for this example is pretty simple and can be summarised as follows:
### Agent delegation and dependencies
Generally the delegate agent needs to either have the same [dependencies](https://ai.pydantic.dev/dependencies/) as the calling agent, or dependencies which are a subset of the calling agent's dependencies.
Initializing dependencies
We say "generally" above since there's nothing to stop you initializing dependencies within a tool call and therefore using interdependencies in a delegate agent that are not available on the parent, this should often be avoided since it can be significantly slower than reusing connections etc. from the parent agent.
agent_delegation_deps.py```
fromdataclassesimport dataclass
importhttpx
frompydantic_aiimport Agent, RunContext

@dataclass
classClientAndKey: 
  http_client: httpx.AsyncClient
  api_key: str

joke_selection_agent = Agent(
  'openai:gpt-4o',
  deps_type=ClientAndKey, 
  system_prompt=(
    'Use the `joke_factory` tool to generate some jokes on the given subject, '
    'then choose the best. You must return just a single joke.'
  ),
)
joke_generation_agent = Agent(
  'gemini-1.5-flash',
  deps_type=ClientAndKey, 
  result_type=list[str],
  system_prompt=(
    'Use the "get_jokes" tool to get some jokes on the given subject, '
    'then extract each joke into a list.'
  ),
)

@joke_selection_agent.tool
async defjoke_factory(ctx: RunContext[ClientAndKey], count: int) -> list[str]:
  r = await joke_generation_agent.run(
    f'Please generate {count} jokes.',
    deps=ctx.deps, 
    usage=ctx.usage,
  )
  return r.data

@joke_generation_agent.tool 
async defget_jokes(ctx: RunContext[ClientAndKey], count: int) -> str:
  response = await ctx.deps.http_client.get(
    'https://example.com',
    params={'count': count},
    headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
  )
  response.raise_for_status()
  return response.text

async defmain():
  async with httpx.AsyncClient() as client:
    deps = ClientAndKey(client, 'foobar')
    result = await joke_selection_agent.run('Tell me a joke.', deps=deps)
    print(result.data)
    #> Did you hear about the toothpaste scandal? They called it Colgate.
    print(result.usage()) 
"""
    Usage(
      requests=4,
      request_tokens=309,
      response_tokens=32,
      total_tokens=341,
      details=None,

    """

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
This example shows how even a fairly simple agent delegation can lead to a complex control flow:
## Programmatic agent hand-off
"Programmatic agent hand-off" refers to the scenario where multiple agents are called in succession, with application code and/or a human in the loop responsible for deciding which agent to call next.
Here agents don't need to use the same deps.
Here we show two agents used in succession, the first to find a flight and the second to extract the user's seat preference.
programmatic_handoff.py```
fromtypingimport Literal, Union
frompydanticimport BaseModel, Field
fromrich.promptimport Prompt
frompydantic_aiimport Agent, RunContext
frompydantic_ai.messagesimport ModelMessage
frompydantic_ai.usageimport Usage, UsageLimits

classFlightDetails(BaseModel):
  flight_number: str

classFailed(BaseModel):
"""Unable to find a satisfactory choice."""

flight_search_agent = Agent[None, Union[FlightDetails, Failed]]( 
  'openai:gpt-4o',
  result_type=Union[FlightDetails, Failed], # type: ignore
  system_prompt=(
    'Use the "flight_search" tool to find a flight '
    'from the given origin to the given destination.'
  ),
)

@flight_search_agent.tool 
async defflight_search(
  ctx: RunContext[None], origin: str, destination: str
) -> Union[FlightDetails, None]:
  # in reality, this would call a flight search API or
  # use a browser to scrape a flight search website
  return FlightDetails(flight_number='AK456')

usage_limits = UsageLimits(request_limit=15) 

async deffind_flight(usage: Usage) -> Union[FlightDetails, None]: 
  message_history: Union[list[ModelMessage], None] = None
  for _ in range(3):
    prompt = Prompt.ask(
      'Where would you like to fly from and to?',
    )
    result = await flight_search_agent.run(
      prompt,
      message_history=message_history,
      usage=usage,
      usage_limits=usage_limits,
    )
    if isinstance(result.data, FlightDetails):
      return result.data
    else:
      message_history = result.all_messages(
        result_tool_return_content='Please try again.'
      )

classSeatPreference(BaseModel):
  row: int = Field(ge=1, le=30)
  seat: Literal['A', 'B', 'C', 'D', 'E', 'F']

# This agent is responsible for extracting the user's seat selection
seat_preference_agent = Agent[None, Union[SeatPreference, Failed]]( 
  'openai:gpt-4o',
  result_type=Union[SeatPreference, Failed], # type: ignore
  system_prompt=(
    "Extract the user's seat preference. "
    'Seats A and F are window seats. '
    'Row 1 is the front row and has extra leg room. '
    'Rows 14, and 20 also have extra leg room. '
  ),
)

async deffind_seat(usage: Usage) -> SeatPreference: 
  message_history: Union[list[ModelMessage], None] = None
  while True:
    answer = Prompt.ask('What seat would you like?')
    result = await seat_preference_agent.run(
      answer,
      message_history=message_history,
      usage=usage,
      usage_limits=usage_limits,
    )
    if isinstance(result.data, SeatPreference):
      return result.data
    else:
      print('Could not understand seat preference. Please try again.')
      message_history = result.all_messages()

async defmain(): 
  usage: Usage = Usage()
  opt_flight_details = await find_flight(usage)
  if opt_flight_details is not None:
    print(f'Flight found: {opt_flight_details.flight_number}')
    #> Flight found: AK456
    seat_preference = await find_seat(usage)
    print(f'Seat preference: {seat_preference}')
    #> Seat preference: row=1 seat='A'

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
The control flow for this example can be summarised as follows:
## Pydantic Graphs
See the [graph](https://ai.pydantic.dev/graph/) documentation on when and how to use graphs.
## Examples
The following examples demonstrate how to use dependencies in PydanticAI:



# Results
Results are the final values returned from [running an agent](https://ai.pydantic.dev/agents/#running-agents). The result values are wrapped in [`AgentRunResult`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.AgentRunResult) and [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult) so you can access other data like [usage](https://ai.pydantic.dev/api/usage/#pydantic_ai.usage.Usage) of the run and [message history](https://ai.pydantic.dev/message-history/#accessing-messages-from-results)
Both `RunResult` and `StreamedRunResult` are generic in the data they wrap, so typing information about the data returned by the agent is preserved.
olympics.py```
frompydanticimport BaseModel
frompydantic_aiimport Agent

classCityLocation(BaseModel):
  city: str
  country: str

agent = Agent('google-gla:gemini-1.5-flash', result_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""

```

_(This example is complete, it can be run "as is")_
Runs end when either a plain text response is received or the model calls a tool associated with one of the structured result types. We will add limits to make sure a run doesn't go on indefinitely, see [#70](https://github.com/pydantic/pydantic-ai/issues/70).
## Result data
When the result type is `str`, or a union including `str`, plain text responses are enabled on the model, and the raw text response from the model is used as the response data.
If the result type is a union with multiple members (after remove `str` from the members), each member is registered as a separate tool with the model in order to reduce the complexity of the tool schemas and maximise the chances a model will respond correctly.
If the result type schema is not of type `"object"`, the result type is wrapped in a single element object, so the schema of all tools registered with the model are object schemas.
Structured results (like tools) use Pydantic to build the JSON schema used for the tool, and to validate the data returned by the model.
Bring on PEP-747
Until [PEP-747](https://peps.python.org/pep-0747/) "Annotating Type Forms" lands, unions are not valid as `type`s in Python.
When creating the agent we need to `# type: ignore` the `result_type` argument, and add a type hint to tell type checkers about the type of the agent.
Here's an example of returning either text or a structured value
box_or_error.py```
fromtypingimport Union
frompydanticimport BaseModel
frompydantic_aiimport Agent

classBox(BaseModel):
  width: int
  height: int
  depth: int
  units: str

agent: Agent[None, Union[Box, str]] = Agent(
  'openai:gpt-4o-mini',
  result_type=Union[Box, str], # type: ignore
  system_prompt=(
    "Extract me the dimensions of a box, "
    "if you can't extract all data, ask the user to try again."
  ),
)
result = agent.run_sync('The box is 10x20x30')
print(result.data)
#> Please provide the units for the dimensions (e.g., cm, in, m).
result = agent.run_sync('The box is 10x20x30 cm')
print(result.data)
#> width=10 height=20 depth=30 units='cm'

```

_(This example is complete, it can be run "as is")_
Here's an example of using a union return type which registered multiple tools, and wraps non-object schemas in an object:
colors_or_sizes.py```
fromtypingimport Union
frompydantic_aiimport Agent
agent: Agent[None, Union[list[str], list[int]]] = Agent(
  'openai:gpt-4o-mini',
  result_type=Union[list[str], list[int]], # type: ignore
  system_prompt='Extract either colors or sizes from the shapes provided.',
)
result = agent.run_sync('red square, blue circle, green triangle')
print(result.data)
#> ['red', 'blue', 'green']
result = agent.run_sync('square size 10, circle size 20, triangle size 30')
print(result.data)
#> [10, 20, 30]

```

_(This example is complete, it can be run "as is")_
### Result validators functions
Some validation is inconvenient or impossible to do in Pydantic validators, in particular when the validation requires IO and is asynchronous. PydanticAI provides a way to add validation functions via the [`agent.result_validator`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.result_validator) decorator.
Here's a simplified variant of the [SQL Generation example](https://ai.pydantic.dev/examples/sql-gen/):
sql_gen.py```
fromtypingimport Union
fromfake_databaseimport DatabaseConn, QueryError
frompydanticimport BaseModel
frompydantic_aiimport Agent, RunContext, ModelRetry

classSuccess(BaseModel):
  sql_query: str

classInvalidRequest(BaseModel):
  error_message: str

Response = Union[Success, InvalidRequest]
agent: Agent[DatabaseConn, Response] = Agent(
  'google-gla:gemini-1.5-flash',
  result_type=Response, # type: ignore
  deps_type=DatabaseConn,
  system_prompt='Generate PostgreSQL flavored SQL queries based on user input.',
)

@agent.result_validator
async defvalidate_result(ctx: RunContext[DatabaseConn], result: Response) -> Response:
  if isinstance(result, InvalidRequest):
    return result
  try:
    await ctx.deps.execute(f'EXPLAIN {result.sql_query}')
  except QueryError as e:
    raise ModelRetry(f'Invalid query: {e}') frome
  else:
    return result

result = agent.run_sync(
  'get me users who were last active yesterday.', deps=DatabaseConn()
)
print(result.data)
#> sql_query='SELECT * FROM users WHERE last_active::date = today() - interval 1 day'

```

_(This example is complete, it can be run "as is")_
## Streamed Results
There two main challenges with streamed results:
  1. Validating structured responses before they're complete, this is achieved by "partial validation" which was recently added to Pydantic in [pydantic/pydantic#10748](https://github.com/pydantic/pydantic/pull/10748).
  2. When receiving a response, we don't know if it's the final response without starting to stream it and peeking at the content. PydanticAI streams just enough of the response to sniff out if it's a tool call or a result, then streams the whole thing and calls tools, or returns the stream as a [`StreamedRunResult`](https://ai.pydantic.dev/api/result/#pydantic_ai.result.StreamedRunResult).


### Streaming Text
Example of streamed text result:
streamed_hello_world.py```
frompydantic_aiimport Agent
agent = Agent('google-gla:gemini-1.5-flash') 

async defmain():
  async with agent.run_stream('Where does "hello world" come from?') as result: 
    async for message in result.stream_text(): 
      print(message)
      #> The first known
      #> The first known use of "hello,
      #> The first known use of "hello, world" was in
      #> The first known use of "hello, world" was in a 1974 textbook
      #> The first known use of "hello, world" was in a 1974 textbook about the C
      #> The first known use of "hello, world" was in a 1974 textbook about the C programming language.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
We can also stream text as deltas rather than the entire text in each item:
streamed_delta_hello_world.py```
frompydantic_aiimport Agent
agent = Agent('google-gla:gemini-1.5-flash')

async defmain():
  async with agent.run_stream('Where does "hello world" come from?') as result:
    async for message in result.stream_text(delta=True): 
      print(message)
      #> The first known
      #> use of "hello,
      #> world" was in
      #> a 1974 textbook
      #> about the C
      #> programming language.

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
Result message not included in `messages`
The final result message will **NOT** be added to result messages if you use `.stream_text(delta=True)`, see [Messages and chat history](https://ai.pydantic.dev/message-history/) for more information.
### Streaming Structured Responses
Not all types are supported with partial validation in Pydantic, see [pydantic/pydantic#10748](https://github.com/pydantic/pydantic/pull/10748), generally for model-like structures it's currently best to use `TypeDict`.
Here's an example of streaming a use profile as it's built:
streamed_user_profile.py```
fromdatetimeimport date
fromtyping_extensionsimport TypedDict
frompydantic_aiimport Agent

classUserProfile(TypedDict, total=False):
  name: str
  dob: date
  bio: str

agent = Agent(
  'openai:gpt-4o',
  result_type=UserProfile,
  system_prompt='Extract a user profile from the input',
)

async defmain():
  user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
  async with agent.run_stream(user_input) as result:
    async for profile in result.stream():
      print(profile)
      #> {'name': 'Ben'}
      #> {'name': 'Ben'}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes'}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the '}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyr'}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
If you want fine-grained control of validation, particularly catching validation errors, you can use the following pattern:
streamed_user_profile.py```
fromdatetimeimport date
frompydanticimport ValidationError
fromtyping_extensionsimport TypedDict
frompydantic_aiimport Agent

classUserProfile(TypedDict, total=False):
  name: str
  dob: date
  bio: str

agent = Agent('openai:gpt-4o', result_type=UserProfile)

async defmain():
  user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
  async with agent.run_stream(user_input) as result:
    async for message, last in result.stream_structured(debounce_by=0.01): 
      try:
        profile = await result.validate_structured_result( 
          message,
          allow_partial=not last,
        )
      except ValidationError:
        continue
      print(profile)
      #> {'name': 'Ben'}
      #> {'name': 'Ben'}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes'}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the '}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyr'}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
      #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}

```

_(This example is complete, it can be run "as is" — you'll need to add`asyncio.run(main())` to run `main`)_
## Examples
The following examples demonstrate how to use streamed responses in PydanticAI:
  * [Stream markdown](https://ai.pydantic.dev/examples/stream-markdown/)



# Testing and Evals
With PydanticAI and LLM integrations in general, there are two distinct kinds of test:
  1. **Unit tests** — tests of your application code, and whether it's behaving correctly
  2. **Evals** — tests of the LLM, and how good or bad its responses are


For the most part, these two kinds of tests have pretty separate goals and considerations.
## Unit tests
Unit tests for PydanticAI code are just like unit tests for any other Python code.
Because for the most part they're nothing new, we have pretty well established tools and patterns for writing and running these kinds of tests.
Unless you're really sure you know better, you'll probably want to follow roughly this strategy:
  * Use [`pytest`](https://docs.pytest.org/en/stable/) as your test harness
  * If you find yourself typing out long assertions, use [inline-snapshot](https://15r10nk.github.io/inline-snapshot/latest/)
  * Similarly, [dirty-equals](https://dirty-equals.helpmanual.io/latest/) can be useful for comparing large data structures
  * Use [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel) or [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) in place of your actual model to avoid the usage, latency and variability of real LLM calls
  * Use [`Agent.override`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.override) to replace your model inside your application logic
  * Set [`ALLOW_MODEL_REQUESTS=False`](https://ai.pydantic.dev/api/models/base/#pydantic_ai.models.ALLOW_MODEL_REQUESTS) globally to block any requests from being made to non-test models accidentally


### Unit testing with `TestModel`
The simplest and fastest way to exercise most of your application code is using [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel), this will (by default) call all tools in the agent, then return either plain text or a structured response depending on the return type of the agent.
`TestModel` is not magic
The "clever" (but not too clever) part of `TestModel` is that it will attempt to generate valid structured data for [function tools](https://ai.pydantic.dev/tools/) and [result types](https://ai.pydantic.dev/results/#structured-result-validation) based on the schema of the registered tools.
There's no ML or AI in `TestModel`, it's just plain old procedural Python code that tries to generate data that satisfies the JSON schema of a tool.
The resulting data won't look pretty or relevant, but it should pass Pydantic's validation in most cases. If you want something more sophisticated, use [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) and write your own data generation logic.
Let's write unit tests for the following application code:
weather_app.py```
importasyncio
fromdatetimeimport date
frompydantic_aiimport Agent, RunContext
fromfake_databaseimport DatabaseConn 
fromweather_serviceimport WeatherService 
weather_agent = Agent(
  'openai:gpt-4o',
  deps_type=WeatherService,
  system_prompt='Providing a weather forecast at the locations the user provides.',
)

@weather_agent.tool
defweather_forecast(
  ctx: RunContext[WeatherService], location: str, forecast_date: date
) -> str:
  if forecast_date  date.today(): 
    return ctx.deps.get_historic_weather(location, forecast_date)
  else:
    return ctx.deps.get_forecast(location, forecast_date)

async defrun_weather_forecast( 
  user_prompts: list[tuple[str, int]], conn: DatabaseConn
):
"""Run weather forecast for a list of user prompts and save."""
  async with WeatherService() as weather_service:
    async defrun_forecast(prompt: str, user_id: int):
      result = await weather_agent.run(prompt, deps=weather_service)
      await conn.store_forecast(user_id, result.data)
    # run all prompts in parallel
    await asyncio.gather(
      *(run_forecast(prompt, user_id) for (prompt, user_id) in user_prompts)
    )

```

Here we have a function that takes a list of `(user_prompt, user_id)` tuples, gets a weather forecast for each prompt, and stores the result in the database.
**We want to test this code without having to mock certain objects or modify our code so we can pass test objects in.**
Here's how we would write tests using [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel):
test_weather_app.py```
fromdatetimeimport timezone
importpytest
fromdirty_equalsimport IsNow, IsStr
frompydantic_aiimport models, capture_run_messages
frompydantic_ai.models.testimport TestModel
frompydantic_ai.messagesimport (
  ModelResponse,
  SystemPromptPart,
  TextPart,
  ToolCallPart,
  ToolReturnPart,
  UserPromptPart,
  ModelRequest,
)
fromfake_databaseimport DatabaseConn
fromweather_appimport run_weather_forecast, weather_agent
pytestmark = pytest.mark.anyio 
models.ALLOW_MODEL_REQUESTS = False 

async deftest_forecast():
  conn = DatabaseConn()
  user_id = 1
  with capture_run_messages() as messages:
    with weather_agent.override(model=TestModel()): 
      prompt = 'What will the weather be like in London on 2024-11-28?'
      await run_weather_forecast([(prompt, user_id)], conn) 
  forecast = await conn.get_forecast(user_id)
  assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}' 
  assert messages == [ 
    ModelRequest(
      parts=[
        SystemPromptPart(
          content='Providing a weather forecast at the locations the user provides.',
          timestamp=IsNow(tz=timezone.utc),
        ),
        UserPromptPart(
          content='What will the weather be like in London on 2024-11-28?',
          timestamp=IsNow(tz=timezone.utc), 
        ),
      ]
    ),
    ModelResponse(
      parts=[
        ToolCallPart(
          tool_name='weather_forecast',
          args={
            'location': 'a',
            'forecast_date': '2024-01-01', 
          },
          tool_call_id=IsStr(),
        )
      ],
      model_name='test',
      timestamp=IsNow(tz=timezone.utc),
    ),
    ModelRequest(
      parts=[
        ToolReturnPart(
          tool_name='weather_forecast',
          content='Sunny with a chance of rain',
          tool_call_id=IsStr(),
          timestamp=IsNow(tz=timezone.utc),
        ),
      ],
    ),
    ModelResponse(
      parts=[
        TextPart(
          content='{"weather_forecast":"Sunny with a chance of rain"}',
        )
      ],
      model_name='test',
      timestamp=IsNow(tz=timezone.utc),
    ),
  ]

```

### Unit testing with `FunctionModel`
The above tests are a great start, but careful readers will notice that the `WeatherService.get_forecast` is never called since `TestModel` calls `weather_forecast` with a date in the past.
To fully exercise `weather_forecast`, we need to use [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) to customise how the tools is called.
Here's an example of using `FunctionModel` to test the `weather_forecast` tool with custom inputs
test_weather_app2.py```
importre
importpytest
frompydantic_aiimport models
frompydantic_ai.messagesimport (
  ModelMessage,
  ModelResponse,
  TextPart,
  ToolCallPart,
)
frompydantic_ai.models.functionimport AgentInfo, FunctionModel
fromfake_databaseimport DatabaseConn
fromweather_appimport run_weather_forecast, weather_agent
pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False

defcall_weather_forecast( 
  messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
  if len(messages) == 1:
    # first call, call the weather forecast tool
    user_prompt = messages[0].parts[-1]
    m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
    assert m is not None
    args = {'location': 'London', 'forecast_date': m.group()} 
    return ModelResponse(parts=[ToolCallPart('weather_forecast', args)])
  else:
    # second call, return the forecast
    msg = messages[-1].parts[0]
    assert msg.part_kind == 'tool-return'
    return ModelResponse(parts=[TextPart(f'The forecast is: {msg.content}')])

async deftest_forecast_future():
  conn = DatabaseConn()
  user_id = 1
  with weather_agent.override(model=FunctionModel(call_weather_forecast)): 
    prompt = 'What will the weather be like in London on 2032-01-01?'
    await run_weather_forecast([(prompt, user_id)], conn)
  forecast = await conn.get_forecast(user_id)
  assert forecast == 'The forecast is: Rainy with a chance of sun'

```

### Overriding model via pytest fixtures
If you're writing lots of tests that all require model to be overridden, you can use [pytest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) to override the model with [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel) or [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) in a reusable way.
Here's an example of a fixture that overrides the model with `TestModel`:
tests.py```
importpytest
fromweather_appimport weather_agent
frompydantic_ai.models.testimport TestModel

@pytest.fixture
defoverride_weather_agent():
  with weather_agent.override(model=TestModel()):
    yield

async deftest_forecast(override_weather_agent: None):
  ...
  # test code here

```

## Evals
"Evals" refers to evaluating a models performance for a specific application.
Warning
Unlike unit tests, evals are an emerging art/science; anyone who claims to know for sure exactly how your evals should be defined can safely be ignored.
Evals are generally more like benchmarks than unit tests, they never "pass" although they do "fail"; you care mostly about how they change over time.
Since evals need to be run against the real model, then can be slow and expensive to run, you generally won't want to run them in CI for every commit.
### Measuring performance
The hardest part of evals is measuring how well the model has performed.
In some cases (e.g. an agent to generate SQL) there are simple, easy to run tests that can be used to measure performance (e.g. is the SQL valid? Does it return the right results? Does it return just the right results?).
In other cases (e.g. an agent that gives advice on quitting smoking) it can be very hard or impossible to make quantitative measures of performance — in the smoking case you'd really need to run a double-blind trial over months, then wait 40 years and observe health outcomes to know if changes to your prompt were an improvement.
There are a few different strategies you can use to measure performance:
  * **End to end, self-contained tests** — like the SQL example, we can test the final result of the agent near-instantly
  * **Synthetic self-contained tests** — writing unit test style checks that the output is as expected, checks like `'chewing gum' in response`, while these checks might seem simplistic they can be helpful, one nice characteristic is that it's easy to tell what's wrong when they fail
  * **LLMs evaluating LLMs** — using another models, or even the same model with a different prompt to evaluate the performance of the agent (like when the class marks each other's homework because the teacher has a hangover), while the downsides and complexities of this approach are obvious, some think it can be a useful tool in the right circumstances
  * **Evals in prod** — measuring the end results of the agent in production, then creating a quantitative measure of performance, so you can easily measure changes over time as you change the prompt or model used, [logfire](https://ai.pydantic.dev/logfire/) can be extremely useful in this case since you can write a custom query to measure the performance of your agent


### System prompt customization
The system prompt is the developer's primary tool in controlling an agent's behavior, so it's often useful to be able to customise the system prompt and see how performance changes. This is particularly relevant when the system prompt contains a list of examples and you want to understand how changing that list affects the model's performance.
Let's assume we have the following app for running SQL generated from a user prompt (this examples omits a lot of details for brevity, see the [SQL gen](https://ai.pydantic.dev/examples/sql-gen/) example for a more complete code):
sql_app.py```
importjson
frompathlibimport Path
fromtypingimport Union
frompydantic_aiimport Agent, RunContext
fromfake_databaseimport DatabaseConn

classSqlSystemPrompt: 
  def__init__(
    self, examples: Union[list[dict[str, str]], None] = None, db: str = 'PostgreSQL'
  ):
    if examples is None:
      # if examples aren't provided, load them from file, this is the default
      with Path('examples.json').open('rb') as f:
        self.examples = json.load(f)
    else:
      self.examples = examples
    self.db = db
  defbuild_prompt(self) -> str: 
    return f"""\
Given the following {self.db} table of records, your job is to
write a SQL query that suits the user's request.
Database schema:
CREATE TABLE records (
 ...
);
{''.join(self.format_example(example)forexampleinself.examples)}
"""
  @staticmethod
  defformat_example(example: dict[str, str]) -> str: 
    return f"""\
<example>
 <request>{example['request']}</request>
 <sql>{example['sql']}</sql>
</example>
"""

sql_agent = Agent(
  'google-gla:gemini-1.5-flash',
  deps_type=SqlSystemPrompt,
)

@sql_agent.system_prompt
async defsystem_prompt(ctx: RunContext[SqlSystemPrompt]) -> str:
  return ctx.deps.build_prompt()

async defuser_search(user_prompt: str) -> list[dict[str, str]]:
"""Search the database based on the user's prompts."""
  ... 
  result = await sql_agent.run(user_prompt, deps=SqlSystemPrompt())
  conn = DatabaseConn()
  return await conn.execute(result.data)

```

`examples.json` looks something like this:
```
request: show me error records with the tag "foobar"
response: SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)

```

examples.json```
{
"examples":[
{
"request":"Show me all records",
"sql":"SELECT * FROM records;"
},
{
"request":"Show me all records from 2021",
"sql":"SELECT * FROM records WHERE date_trunc('year', date) = '2021-01-01';"
},
{
"request":"show me error records with the tag 'foobar'",
"sql":"SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags);"
},
...
]
}

```

Now we want a way to quantify the success of the SQL generation so we can judge how changes to the agent affect its performance.
We can use [`Agent.override`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.override) to replace the system prompt with a custom one that uses a subset of examples, and then run the application code (in this case `user_search`). We also run the actual SQL from the examples and compare the "correct" result from the example SQL to the SQL generated by the agent. (We compare the results of running the SQL rather than the SQL itself since the SQL might be semantically equivalent but written in a different way).
To get a quantitative measure of performance, we assign points to each run as follows:
  * **-100** points if the generated SQL is invalid
  * **-1** point for each row returned by the agent (so returning lots of results is discouraged)
  * **+5** points for each row returned by the agent that matches the expected result


We use 5-fold cross-validation to judge the performance of the agent using our existing set of examples.
sql_app_evals.py```
importjson
importstatistics
frompathlibimport Path
fromitertoolsimport chain
fromfake_databaseimport DatabaseConn, QueryError
fromsql_appimport sql_agent, SqlSystemPrompt, user_search

async defmain():
  with Path('examples.json').open('rb') as f:
    examples = json.load(f)
  # split examples into 5 folds
  fold_size = len(examples) // 5
  folds = [examples[i : i + fold_size] for i in range(0, len(examples), fold_size)]
  conn = DatabaseConn()
  scores = []
  for i, fold in enumerate(folds):
    fold_score = 0
    # build all other folds into a list of examples
    other_folds = list(chain(*(f for j, f in enumerate(folds) if j != i)))
    # create a new system prompt with the other fold examples
    system_prompt = SqlSystemPrompt(examples=other_folds)
    # override the system prompt with the new one
    with sql_agent.override(deps=system_prompt):
      for case in fold:
        try:
          agent_results = await user_search(case['request'])
        except QueryError as e:
          print(f'Fold {i}{case}: {e}')
          fold_score -= 100
        else:
          # get the expected results using the SQL from this case
          expected_results = await conn.execute(case['sql'])
        agent_ids = [r['id'] for r in agent_results]
        # each returned value has a score of -1
        fold_score -= len(agent_ids)
        expected_ids = {r['id'] for r in expected_results}
        # each return value that matches the expected value has a score of 3
        fold_score += 5 * len(set(agent_ids)  expected_ids)
    scores.append(fold_score)
  overall_score = statistics.mean(scores)
  print(f'Overall score: {overall_score:0.2f}')
  #> Overall score: 12.00

```

We can then change the prompt, the model, or the examples and see how the score changes over time.

[ Skip to content ](https://ai.pydantic.dev/tools/#function-tools)
# Function Tools
Function tools provide a mechanism for models to retrieve extra information to help them generate a response.
They're useful when it is impractical or impossible to put all the context an agent might need into the system prompt, or when you want to make agents' behavior more deterministic or reliable by deferring some of the logic required to generate a response to another (not necessarily AI-powered) tool.
Function tools vs. RAG
Function tools are basically the "R" of RAG (Retrieval-Augmented Generation) — they augment what the model can do by letting it request extra information.
The main semantic difference between PydanticAI Tools and RAG is RAG is synonymous with vector search, while PydanticAI tools are more general-purpose. (Note: we may add support for vector search functionality in the future, particularly an API for generating embeddings. See [#58](https://github.com/pydantic/pydantic-ai/issues/58))
There are a number of ways to register tools with an agent:
  * via the [`@agent.tool`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool) decorator — for tools that need access to the agent [context](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext)
  * via the [`@agent.tool_plain`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool_plain) decorator — for tools that do not need access to the agent [context](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext)
  * via the [`tools`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__) keyword argument to `Agent` which can take either plain functions, or instances of [`Tool`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.Tool)


`@agent.tool` is considered the default decorator since in the majority of cases tools will need access to the agent context.
Here's an example using both:
dice_game.py```
importrandom
frompydantic_aiimport Agent, RunContext
agent = Agent(
  'google-gla:gemini-1.5-flash', 
  deps_type=str, 
  system_prompt=(
    "You're a dice game, you should roll the die and see if the number "
    "you get back matches the user's guess. If so, tell them they're a winner. "
    "Use the player's name in the response."
  ),
)

@agent.tool_plain 
defroll_die() -> str:
"""Roll a six-sided die and return the result."""
  return str(random.randint(1, 6))

@agent.tool 
defget_player_name(ctx: RunContext[str]) -> str:
"""Get the player's name."""
  return ctx.deps

dice_result = agent.run_sync('My guess is 4', deps='Anne') 
print(dice_result.data)
#> Congratulations Anne, you guessed correctly! You're a winner!

```

_(This example is complete, it can be run "as is")_
Let's print the messages from that game to see what happened:
dice_game_messages.py```
fromdice_gameimport dice_result
print(dice_result.all_messages())
"""
[
  ModelRequest(
    parts=[
      SystemPromptPart(
        content="You're a dice game, you should roll the die and see if the number you get back matches the user's guess. If so, tell them they're a winner. Use the player's name in the response.",
        timestamp=datetime.datetime(...),
        dynamic_ref=None,
        part_kind='system-prompt',

      UserPromptPart(
        content='My guess is 4',
        timestamp=datetime.datetime(...),
        part_kind='user-prompt',

    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      ToolCallPart(
        tool_name='roll_die',
        args={},
        tool_call_id='pyd_ai_tool_call_id',
        part_kind='tool-call',

    ],
    model_name='gemini-1.5-flash',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
  ModelRequest(
    parts=[
      ToolReturnPart(
        tool_name='roll_die',
        content='4',
        tool_call_id='pyd_ai_tool_call_id',
        timestamp=datetime.datetime(...),
        part_kind='tool-return',

    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      ToolCallPart(
        tool_name='get_player_name',
        args={},
        tool_call_id='pyd_ai_tool_call_id',
        part_kind='tool-call',

    ],
    model_name='gemini-1.5-flash',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
  ModelRequest(
    parts=[
      ToolReturnPart(
        tool_name='get_player_name',
        content='Anne',
        tool_call_id='pyd_ai_tool_call_id',
        timestamp=datetime.datetime(...),
        part_kind='tool-return',

    ],
    kind='request',
  ),
  ModelResponse(
    parts=[
      TextPart(
        content="Congratulations Anne, you guessed correctly! You're a winner!",
        part_kind='text',

    ],
    model_name='gemini-1.5-flash',
    timestamp=datetime.datetime(...),
    kind='response',
  ),
]
"""

```

We can represent this with a diagram:
## Registering Function Tools via kwarg
As well as using the decorators, we can register tools via the `tools` argument to the [`Agent` constructor](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.__init__). This is useful when you want to reuse tools, and can also give more fine-grained control over the tools.
dice_game_tool_kwarg.py```
importrandom
frompydantic_aiimport Agent, RunContext, Tool

defroll_die() -> str:
"""Roll a six-sided die and return the result."""
  return str(random.randint(1, 6))

defget_player_name(ctx: RunContext[str]) -> str:
"""Get the player's name."""
  return ctx.deps

agent_a = Agent(
  'google-gla:gemini-1.5-flash',
  deps_type=str,
  tools=[roll_die, get_player_name], 
)
agent_b = Agent(
  'google-gla:gemini-1.5-flash',
  deps_type=str,
  tools=[ 
    Tool(roll_die, takes_ctx=False),
    Tool(get_player_name, takes_ctx=True),
  ],
)
dice_result = agent_b.run_sync('My guess is 4', deps='Anne')
print(dice_result.data)
#> Congratulations Anne, you guessed correctly! You're a winner!

```

_(This example is complete, it can be run "as is")_
## Function Tools vs. Structured Results
As the name suggests, function tools use the model's "tools" or "functions" API to let the model know what is available to call. Tools or functions are also used to define the schema(s) for structured responses, thus a model might have access to many tools, some of which call function tools while others end the run and return a result.
## Function tools and schema
Function parameters are extracted from the function signature, and all parameters except `RunContext` are used to build the schema for that tool call.
Even better, PydanticAI extracts the docstring from functions and (thanks to [griffe](https://mkdocstrings.github.io/griffe/)) extracts parameter descriptions from the docstring and adds them to the schema.
[Griffe supports](https://mkdocstrings.github.io/griffe/reference/docstrings/#docstrings) extracting parameter descriptions from `google`, `numpy`, and `sphinx` style docstrings. PydanticAI will infer the format to use based on the docstring, but you can explicitly set it using [`docstring_format`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.DocstringFormat). You can also enforce parameter requirements by setting `require_parameter_descriptions=True`. This will raise a [`UserError`](https://ai.pydantic.dev/api/exceptions/#pydantic_ai.exceptions.UserError) if a parameter description is missing.
To demonstrate a tool's schema, here we use [`FunctionModel`](https://ai.pydantic.dev/api/models/function/#pydantic_ai.models.function.FunctionModel) to print the schema a model would receive:
tool_schema.py```
frompydantic_aiimport Agent
frompydantic_ai.messagesimport ModelMessage, ModelResponse, TextPart
frompydantic_ai.models.functionimport AgentInfo, FunctionModel
agent = Agent()

@agent.tool_plain(docstring_format='google', require_parameter_descriptions=True)
deffoobar(a: int, b: str, c: dict[str, list[float]]) -> str:
"""Get me foobar.
  Args:
    a: apple pie
    b: banana cake
    c: carrot smoothie
  """
  return f'{a}{b}{c}'

defprint_schema(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
  tool = info.function_tools[0]
  print(tool.description)
  #> Get me foobar.
  print(tool.parameters_json_schema)
"""

    'additionalProperties': False,
    'properties': {
      'a': {'description': 'apple pie', 'type': 'integer'},
      'b': {'description': 'banana cake', 'type': 'string'},
      'c': {
        'additionalProperties': {'items': {'type': 'number'}, 'type': 'array'},
        'description': 'carrot smoothie',
        'type': 'object',

    },
    'required': ['a', 'b', 'c'],
    'type': 'object',

  """
  return ModelResponse(parts=[TextPart('foobar')])

agent.run_sync('hello', model=FunctionModel(print_schema))

```

_(This example is complete, it can be run "as is")_
The return type of tool can be anything which Pydantic can serialize to JSON as some models (e.g. Gemini) support semi-structured return values, some expect text (OpenAI) but seem to be just as good at extracting meaning from the data. If a Python object is returned and the model expects a string, the value will be serialized to JSON.
If a tool has a single parameter that can be represented as an object in JSON schema (e.g. dataclass, TypedDict, pydantic model), the schema for the tool is simplified to be just that object.
Here's an example where we use [`TestModel.last_model_request_parameters`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel.last_model_request_parameters) to inspect the tool schema that would be passed to the model.
single_parameter_tool.py```
frompydanticimport BaseModel
frompydantic_aiimport Agent
frompydantic_ai.models.testimport TestModel
agent = Agent()

classFoobar(BaseModel):
"""This is a Foobar"""
  x: int
  y: str
  z: float = 3.14

@agent.tool_plain
deffoobar(f: Foobar) -> str:
  return str(f)

test_model = TestModel()
result = agent.run_sync('hello', model=test_model)
print(result.data)
#> {"foobar":"x=0 y='a' z=3.14"}
print(test_model.last_model_request_parameters.function_tools)
"""
[
  ToolDefinition(
    name='foobar',
    description='This is a Foobar',
    parameters_json_schema={
      'properties': {
        'x': {'type': 'integer'},
        'y': {'type': 'string'},
        'z': {'default': 3.14, 'type': 'number'},

      'required': ['x', 'y'],
      'title': 'Foobar',
      'type': 'object',
    },
    outer_typed_dict_key=None,

]
"""

```

_(This example is complete, it can be run "as is")_
## Dynamic Function tools
Tools can optionally be defined with another function: `prepare`, which is called at each step of a run to customize the definition of the tool passed to the model, or omit the tool completely from that step.
A `prepare` method can be registered via the `prepare` kwarg to any of the tool registration mechanisms:
  * [`@agent.tool`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool) decorator
  * [`@agent.tool_plain`](https://ai.pydantic.dev/api/agent/#pydantic_ai.agent.Agent.tool_plain) decorator
  * [`Tool`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.Tool) dataclass


The `prepare` method, should be of type [`ToolPrepareFunc`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolPrepareFunc), a function which takes [`RunContext`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.RunContext) and a pre-built [`ToolDefinition`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.ToolDefinition), and should either return that `ToolDefinition` with or without modifying it, return a new `ToolDefinition`, or return `None` to indicate this tools should not be registered for that step.
Here's a simple `prepare` method that only includes the tool if the value of the dependency is `42`.
As with the previous example, we use [`TestModel`](https://ai.pydantic.dev/api/models/test/#pydantic_ai.models.test.TestModel) to demonstrate the behavior without calling a real model.
tool_only_if_42.py```
fromtypingimport Union
frompydantic_aiimport Agent, RunContext
frompydantic_ai.toolsimport ToolDefinition
agent = Agent('test')

async defonly_if_42(
  ctx: RunContext[int], tool_def: ToolDefinition
) -> Union[ToolDefinition, None]:
  if ctx.deps == 42:
    return tool_def

@agent.tool(prepare=only_if_42)
defhitchhiker(ctx: RunContext[int], answer: str) -> str:
  return f'{ctx.deps}{answer}'

result = agent.run_sync('testing...', deps=41)
print(result.data)
#> success (no tool calls)
result = agent.run_sync('testing...', deps=42)
print(result.data)
#> {"hitchhiker":"42 a"}

```

_(This example is complete, it can be run "as is")_
Here's a more complex example where we change the description of the `name` parameter to based on the value of `deps`
For the sake of variation, we create this tool using the [`Tool`](https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.Tool) dataclass.
customize_name.py```
from__future__import annotations
fromtypingimport Literal
frompydantic_aiimport Agent, RunContext
frompydantic_ai.models.testimport TestModel
frompydantic_ai.toolsimport Tool, ToolDefinition

defgreet(name: str) -> str:
  return f'hello {name}'

async defprepare_greet(
  ctx: RunContext[Literal['human', 'machine']], tool_def: ToolDefinition
) -> ToolDefinition | None:
  d = f'Name of the {ctx.deps} to greet.'
  tool_def.parameters_json_schema['properties']['name']['description'] = d
  return tool_def

greet_tool = Tool(greet, prepare=prepare_greet)
test_model = TestModel()
agent = Agent(test_model, tools=[greet_tool], deps_type=Literal['human', 'machine'])
result = agent.run_sync('testing...', deps='human')
print(result.data)
#> {"greet":"hello a"}
print(test_model.last_model_request_parameters.function_tools)
"""
[
  ToolDefinition(
    name='greet',
    description='',
    parameters_json_schema={
      'additionalProperties': False,
      'properties': {
        'name': {'type': 'string', 'description': 'Name of the human to greet.'}

      'required': ['name'],
      'type': 'object',
    },
    outer_typed_dict_key=None,

]
"""

```

_(This example is complete, it can be run "as is")_

# Troubleshooting
Below are suggestions on how to fix some common errors you might encounter while using PydanticAI. If the issue you're experiencing is not listed below or addressed in the documentation, please feel free to ask in the [Pydantic Slack](https://ai.pydantic.dev/help/) or create an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).
## Jupyter Notebook Errors
### `RuntimeError: This event loop is already running`
This error is caused by conflicts between the event loops in Jupyter notebook and PydanticAI's. One way to manage these conflicts is by using [`nest-asyncio`](https://pypi.org/project/nest-asyncio/). Namely, before you execute any agent runs, do the following: 
```
importnest_asyncio
nest_asyncio.apply()

```

Note: This fix also applies to Google Colab. 
## API Key Configuration
### `UserError: API key must be provided or set in the [MODEL]_API_KEY environment variable`
If you're running into issues with setting the API key for your model, visit the [Models](https://ai.pydantic.dev/models/) page to learn more about how to set an environment variable and/or pass in an `api_key` argument.
## Monitoring HTTPX Requests
You can use custom `httpx` clients in your models in order to access specific requests, responses, and headers at runtime.
It's particularly helpful to use `logfire`'s [HTTPX integration](https://ai.pydantic.dev/logfire/#monitoring-httpx-requests) to monitor the above.
