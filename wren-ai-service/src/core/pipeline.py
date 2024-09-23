import asyncio
from abc import ABCMeta, abstractmethod
from typing import Any, Dict

from hamilton.experimental.h_async import AsyncDriver
from haystack import Pipeline

class BasicPipeline(metaclass=ABCMeta):
    """
    Abstract class that defines the structure of a pipeline.
    Each pipeline should accept a Pipeline or AsyncDriver and define how the `run` method is executed.
    """
    def __init__(self, pipe: Pipeline | AsyncDriver):
        self._pipe = pipe

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Runs the pipeline with the provided arguments.
        Subclasses must implement this method to define pipeline-specific logic.
        :return: A dictionary containing the results of the pipeline run.
        """
        ...


def async_validate(task: callable):
    """
    Runs an asynchronous task and prints the result. Intended for task validation.
    :param task: A callable function representing an async task.
    :return: The result of the task.
    """
    result = asyncio.run(task())
    print(result)
    return result
