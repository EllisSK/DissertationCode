# SPDX-FileCopyrightText: 2026 Ellis Sinclair-Kent
#
# SPDX-License-Identifier: GPL-2.0-only

from argparse import ArgumentParser
from typing import Any, Sequence


class CustomParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "--analysis", action="store_true", help="Run the lab data analysis script"
        )
        self.add_argument(
            "--visualisation",
            action="store_true",
            help="Run the lab data visualisation script",
        )
