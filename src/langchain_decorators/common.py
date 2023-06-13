import logging
import yaml
from enum import Enum
from typing import Any, Union, Optional
from pydantic import BaseModel, Extra

from langchain.llms.base import BaseLanguageModel
from langchain.chat_models import ChatOpenAI


class GlobalSettings(BaseModel):
    default_llm: Optional[BaseLanguageModel] = None
    default_streaming_llm: Optional[BaseLanguageModel] = None
    logging_level: int = logging.INFO
    stdout_logging: bool = True

    verbose: bool = False

    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow

    @classmethod
    def define_settings(cls,
                        settings_type="default",
                        default_llm=None,
                        default_streaming_llm=None,
                        logging_level=logging.INFO,
                        stdout_logging: bool = True,
                        verbose=False,
                        **kwargs
                        ):
        if default_llm is None:
            default_llm = ChatOpenAI(temperature=0.0)
        if default_streaming_llm is None:
            default_streaming_llm = ChatOpenAI(temperature=0.0, streaming=True)
        settings = cls(default_llm=default_llm, default_streaming_llm=default_streaming_llm,
                       logging_level=logging_level, stdout_logging=stdout_logging, verbose=verbose,  **kwargs)
        if not hasattr(GlobalSettings, "registry"):
            setattr(GlobalSettings, "registry", {})
        GlobalSettings.registry[settings_type] = settings

    @classmethod
    def get_current_settings(cls) -> "GlobalSettings":
        if not hasattr(GlobalSettings, "settings_type"):
            setattr(GlobalSettings, "settings_type", "default")
        if not hasattr(GlobalSettings, "registry"):
            GlobalSettings.define_settings()
        return GlobalSettings.registry[GlobalSettings.settings_type]

    @classmethod
    def switch_settings(cls, project_name):
        GlobalSettings.settings_type = project_name


class LogColors(Enum):
    WHITE_BOLD = "\033[1m"
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    DARK_GRAY = '\033[90m'
    WHITE = '\033[39m'
    BLACK_AND_WHITE = '\033[40m'

    # Define some reset codes to restore the default text color
    RESET = '\033[0m'


def print_log(log_object: Any, log_level: int, color: LogColors = None):
    settings = GlobalSettings.get_current_settings()
    if settings.logging_level <= log_level or settings.verbose:
        if isinstance(log_object, str):
            pass
        elif isinstance(log_object, dict):
            log_object = yaml.safe_dump(log_object)
        elif isinstance(log_object, BaseModel):
            log_object = yaml.safe_dump(log_object.dict())

        if color is None:
            if log_level >= logging.ERROR:
                color = LogColors.RED
            elif log_level >= logging.WARNING:
                color = LogColors.YELLOW
            elif log_level >= logging.INFO:
                color = LogColors.GREEN
            else:
                color = LogColors.DARK_GRAY
        if type(color) is LogColors:
            color = color.value
        reset = LogColors.RESET.value if color else ""
        print(f"{color}{log_object}{reset}\n", flush=True)


class PromptTypeSettings:
    def __init__(self, llm: BaseLanguageModel = None,  color: LogColors = None, log_level: Union[int, str] = "info", capture_stream: bool = False):
        self.color = color
        if isinstance(log_level, str):
            log_level = getattr(logging, log_level.upper())
        self.log_level = log_level
        self.capture_stream = capture_stream
        self.llm = llm

    def as_verbose(self):
        return PromptTypeSettings(llm=self.llm, color=self.color, log_level=100, capture_stream=self.capture_stream)


class PromptTypes:
    UNDEFINED: PromptTypeSettings = PromptTypeSettings(
        color=LogColors.BLACK_AND_WHITE, log_level=logging.DEBUG)
    AGENT_REASONING: PromptTypeSettings = PromptTypeSettings(
        color=LogColors.GREEN, log_level=logging.INFO)
    TOOL: PromptTypeSettings = PromptTypeSettings(
        color=LogColors.BLUE, log_level=logging.INFO)
    FINAL_OUTPUT: PromptTypeSettings = PromptTypeSettings(
        color=LogColors.YELLOW, log_level=logging.INFO)
