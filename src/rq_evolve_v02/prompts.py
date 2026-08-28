"""Centralized prompt loading and rendering."""

from __future__ import annotations

from pathlib import Path
from string import Template

from .concepts import DOMAIN_DEFINITIONS
from .models import ProblemRecord

DEFAULT_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompt_templates"


class PromptBook:
    def __init__(self, root: str | Path = DEFAULT_PROMPT_DIR) -> None:
        self.root = Path(root)
        self.crossover_system = self._read("crossover_system.txt")
        self.crossover_user = Template(self._read("crossover_user.txt"))
        self.solver_system = self._read("solver_system.txt").strip()
        self.domain_system = self._read("domain_labeling_system.txt")
        self.domain_user = Template(self._read("domain_labeling_user.txt"))

    def _read(self, name: str) -> str:
        path = self.root / name
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"empty prompt template: {path}")
        return text.strip()

    def crossover_messages(
        self, left: ProblemRecord, right: ProblemRecord
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.crossover_system},
            {
                "role": "user",
                "content": self.crossover_user.substitute(
                    left_question=left.question,
                    left_answer=left.pseudo_gold,
                    right_question=right.question,
                    right_answer=right.pseudo_gold,
                ),
            },
        ]

    def solver_messages(self, question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.solver_system},
            {"role": "user", "content": question},
        ]

    def domain_messages(self, question: str, domain: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.domain_system},
            {
                "role": "user",
                "content": self.domain_user.substitute(
                    question=question,
                    domain=domain,
                    domain_definition=DOMAIN_DEFINITIONS[domain],
                ),
            },
        ]
