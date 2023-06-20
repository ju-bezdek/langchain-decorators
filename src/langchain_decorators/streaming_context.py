import contextvars

from typing import Any, Callable, Coroutine


class StreamingContext():
    from langchain.callbacks.base import AsyncCallbackHandler

    class StreamingContextCallback(AsyncCallbackHandler):

        async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            await StreamingContext.get_context().on_new_token(token)

        async def on_llm_end(self, response, *args, **kwargs):
            if StreamingContext.get_context().stream_to_stdout:
                print()

    context_var = contextvars.ContextVar('streaming_context')

    def __init__(self, callback: Callable[[str], None] = None, callback_async: Callable[[str], Coroutine[None, None, None]]=None, stream_to_stdout: bool = False) -> None:
        self.callback = callback
        self.callback_async = callback_async
        self.stream_to_stdout = stream_to_stdout
        self.token_colors = ['\033[90m', '\033[0m']

    def __enter__(self):
        self.__class__.context_var.set(self)

    @classmethod
    def get_context(cls) -> 'StreamingContext':
        return cls.context_var.get(None)

    async def on_new_token(self, token: str):
        if token:
            if self.callback:
                self.callback(token)
            if self.callback_async:
                await self.callback_async(token)
            if self.stream_to_stdout:
                reset_color = '\033[0m'
                if token.strip() and token!="\n":
                    current_color = self.token_colors[0]
                    self.token_colors.reverse()
                else:
                    current_color='\033[40m'
                print('{}{}{}'.format(current_color, token if token !=
                    "" else '\u2022', reset_color), end='')

    def __exit__(self, exc_type, exc_value, traceback):
        self.context_var.set(None)
