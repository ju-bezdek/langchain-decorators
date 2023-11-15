# Changelog

## Version 0.0.1 (2023-06-10)

- Initial release of the package.

## Version 0.0.2 (2023-06-10)
- fixed typo that caused broken async executions

## Version 0.0.3 (2023-06-11)
- fix https://github.com/ju-bezdek/langchain-decorators/issues/2 
   (shoutout to @lukestanley who fixed it using Anthropic's LLM :)
- better pydantic and markdown output parser... now can self fix casing problems with keys (i.e. field in schema is "ceo_salary", now will accept "CEO salary" as well as "CeoSalary" and any other variations)

## Version 0.0.4 (2023-06-12)
- improved `pydantic` parser is not more tolerant to the casing (accepts pascalCase, snake_case, CamelCase field names, no matter what casing uses model)
- added boolean output parser


## Version 0.0.5 (2023-06-14)
- support for openAI functions 🚀 

## Version 0.0.6 (2023-06-15)
- fix some issues with async prompts

## Version 0.0.7 (2023-06-15)
- fixed streaming capture
- better handling for missing docs for llm_function

## Version 0.0.8 (2023-06-16)
- support for parsing via OpenAI functions 🚀
- support for controlling function_call
- add BIG_CONTEXT prompt type
- ton of bugfixes

## Version 0.0.9 (2023-06-17)
- fix some scenarios of LLM response that raised error
- save AIMessage with function call in output wrapper
- fix logging that we are out or stream context, when stream is not on

## Version 0.0.10 (2023-06-20)
- async screaming callback support
- LlmSelector for automatic selection of LLM based on the model context window and prompt length

## Version 0.0.11 (2023-07-03)
- fixed streaming
- multiple little bugfixes
- option to set the expected generated token count as a hint for LLM selector
- add argument schema option for llm_function

## Version 0.0.12 (2023-07-09)
New parameters in llm decorator
- support for `llm_selector_rule_key` to sub selection of LLM's to for consideration during selection. This enables you to enforce pick only some models (like GPT4 for instance) for particular prompts, or even for particular runs
- support for `function_source` and `memory_source` to point pick properties/attributes of the instance prompt is bound to (aka `self`) as source of functions and memories, so we wont need to send pass it in every time


## Version 0.1.0 (2023-08-09)
- Support for dynamic function schema, that allows augment the function schema dynamically based on the input [more here](./README.MD#dynamic-function-schemas)
- Support Functions provider, that allows control function/tool selection that will be fed into LLM [more here](./README.MD#functions-provider)
- Minor fix for JSON output parser for array scenarios

## Version 0.2.0 (2023-09-20)
- Support for custom template building, to support any kind of prompt block types (https://github.com/ju-bezdek/langchain-decorators/issues/5)
- Support for retrieving a chain object with preconfigured kwargs for more convenient use with the rest of LangChain ecosystem
- support for followup handle for convenient simple followup to response without using a history object
- hotfix support for pydantic v2


## Version 0.2.1 (2023-09-21)
- Hotfix of bug causing simple (without prompt blocks) prompts not working

## Version 0.2.2 (2023-09-25)
- Minor bugfix of LlmSelector causing error in specific cases

## Version 0.2.3 (2023-10-04)
- Fix verbose result longing when not verbose mode
- fix langchain logging warnings for using deprecated imports

## Version 0.3.0 (2023-11-15)
- Support for new OpenAI models (set as default, you can turn it off by setting env variable `LANGCHAIN_DECORATORS_USE_PREVIEW_MODELS=0` )
- automatically turn on new OpenAI JSON mode if `dict` is the output type / JSON output parser
- added timeouts to default models definitions
- you can now reference input variables from `__self__` of the object the `llm_function` is bound to (not only the `llm_prompt`)
- few bug fixes