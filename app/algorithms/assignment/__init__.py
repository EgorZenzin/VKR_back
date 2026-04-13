"""Алгоритмы для задачи назначения (Assignment Problem)."""

from app.algorithms.assignment.greedy import GreedyAssignment
from app.algorithms.assignment.brute_force import BruteForceAssignment
from app.algorithms.assignment.hungarian import HungarianAssignment
from app.algorithms.assignment.genetic import GeneticAssignment

__all__ = [
    "GreedyAssignment",
    "BruteForceAssignment",
    "HungarianAssignment",
    "GeneticAssignment",
]
