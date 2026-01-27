from argparse import _FormatterClass, ArgumentParser
from typing import Any, Sequence

class CustomParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument("--analysis", action="store_true", help="Run the lab data analysis script")
        self.add_argument("--visualisation", action="store_true", help="Run the lab data visualisation script")