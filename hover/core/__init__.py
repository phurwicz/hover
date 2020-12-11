"""
Core module: contains classes which are centerpieces of the vast majority of use cases.

- dataset: defines the primary data structure to work with.
- explorer: defines high-level building blocks of the interactive UI.
- neural: defines sub-applications that involve neural networks.
"""
from rich.console import Console


class Loggable:
    """
    Base class that provides consistently templated logging.

    Inspired by wasabi's good/info/warn/fail methods.
    """

    CONSOLE = Console()

    def _print(self, *args, **kwargs):
        self.__class__.CONSOLE.print(*args, **kwargs)

    def _good(self, message):
        self.__class__.CONSOLE.print(
            f":green_circle: {self.__class__.__name__}: {message}", style="green"
        )

    def _info(self, message):
        self.__class__.CONSOLE.print(
            f":blue_circle: {self.__class__.__name__}: {message}", style="blue"
        )

    def _warn(self, message):
        self.__class__.CONSOLE.print(
            f":yellow_circle: {self.__class__.__name__}: {message}", style="yellow"
        )

    def _fail(self, message):
        self.__class__.CONSOLE.print(
            f":red_circle: {self.__class__.__name__}: {message}", style="red"
        )
