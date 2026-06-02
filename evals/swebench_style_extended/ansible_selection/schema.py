from __future__ import annotations

from dataclasses import dataclass

from .config import WINDOW_LABELS


@dataclass(frozen=True)
class BenchmarkInstance:
    instance_id: str
    date: str
    base_commit: str
    difficulty: str
    gold_files: list[str]
    problem_statement: str
    repo: str = "ansible/ansible"

    def to_json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "date": self.date,
            "base_commit": self.base_commit,
            "num_files": len(self.gold_files),
            "difficulty": self.difficulty,
            "gold_files": self.gold_files,
            "problem_statement": self.problem_statement,
            "repo": self.repo,
        }


@dataclass(frozen=True)
class BenchmarkFile:
    fixed_commit: str
    window: str
    instances: list[BenchmarkInstance]
    repo: str = "ansible/ansible"

    def to_json(self) -> dict:
        instances = [instance.to_json() for instance in self.instances]
        return {
            "repo": self.repo,
            "fixed_commit": self.fixed_commit,
            "window": WINDOW_LABELS[self.window],
            "total_instances": len(instances),
            "hard_instances": sum(instance["difficulty"] == "hard" for instance in instances),
            "instances": instances,
        }
