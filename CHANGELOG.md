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
- improved `pydantic` parser is not more tolerant to casing (accepts pascalCase, snake_case, CamelCase field names, no matter what what casing uses model)
- added boolean output parser