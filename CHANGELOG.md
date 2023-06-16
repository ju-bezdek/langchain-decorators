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

## Version 0.0.7 (2023-06-16)
- support for parsing via OpenAI functions 🚀
- support for controlling function_call
- add BIG_CONTEXT prompt type
- ton of bugfixes

