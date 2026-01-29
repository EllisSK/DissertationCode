from abc import ABC, abstractmethod
from pathlib import Path

class BaseModel(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def _equation(self, *params):
        pass

    def write_report(self, report_path: Path):
        pass

    def create_figure(self):
        pass