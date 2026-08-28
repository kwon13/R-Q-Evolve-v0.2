"""Closed vocabularies for the DOMAIN x PROBLEM_TYPE map."""

from __future__ import annotations

DOMAINS: tuple[str, ...] = (
    "algebra",
    "geometry",
    "number_theory",
    "discrete_mathematics",
    "applied_mathematics",
    "calculus",
    "precalculus",
)

PROBLEM_TYPES: tuple[str, ...] = (
    "decision",
    "search",
    "counting",
    "optimization",
    "function",
)

AXES: tuple[str, ...] = ("domain", "problem_type")

DOMAIN_DEFINITIONS: dict[str, str] = {
    "algebra": (
        "equations, polynomials, identities, algebraic functions, linear or "
        "abstract algebra, and symbolic algebraic structure"
    ),
    "geometry": (
        "Euclidean or non-Euclidean figures, incidence, transformations, "
        "metric, distance, angle, area, and spatial configurations"
    ),
    "number_theory": (
        "integers with indispensable prime, divisibility, congruence, "
        "Diophantine, valuation, gcd/lcm, or arithmetic-function structure"
    ),
    "discrete_mathematics": (
        "combinatorics, graph theory, finite configurations, arrangements, "
        "selections, discrete processes, and enumerative structures"
    ),
    "applied_mathematics": (
        "probability, statistics, numerical mathematics, modeling, mathematical "
        "physics, finance, information theory, or application-driven mathematics"
    ),
    "calculus": (
        "limits, continuity, derivatives, integrals, differential equations, "
        "infinite series, or continuous analysis requiring calculus"
    ),
    "precalculus": (
        "elementary functions, trigonometry, logarithms, exponentials, conics, "
        "function graphs, and elementary sequences without calculus"
    ),
}


def validate_cell(domain: str | None, problem_type: str | None) -> list[str]:
    errors: list[str] = []
    if domain not in DOMAINS:
        errors.append(f"invalid domain: {domain!r}")
    if problem_type not in PROBLEM_TYPES:
        errors.append(f"invalid problem_type: {problem_type!r}")
    return errors


def cell_key(domain: str, problem_type: str) -> str:
    errors = validate_cell(domain, problem_type)
    if errors:
        raise ValueError("; ".join(errors))
    return f"{domain}/{problem_type}"
