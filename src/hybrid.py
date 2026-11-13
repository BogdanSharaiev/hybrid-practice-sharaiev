"""HYBRID (neuro-symbolic) pipeline:

    problem text --(LLM formalizer)--> IR --(validation)--> Z3 solver --(presentation)--> answer

This is the main thing your team builds. The plumbing is here; the parts marked TODO(team) are yours.
Suggested order of work:
    1. write FORMALIZER_PROMPT and get formalize() returning sensible IR for one simple problem;
    2. implement validate_ir();
    3. add ONE retry with feedback in solve();
    4. fix run_solver() so that "unique" and "multiple" can be told apart;
    5. (extension) replace the LLM-based present() with deterministic code for some problem types.

See docs/ir.md for the IR format.
"""
import ast
import json
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

import ir_solver
import llm
from answer import Answer


class HybridError(Exception):
    """The hybrid pipeline could not produce an answer (recorded by run.py as an error, not as status "none")."""


class IRModel(BaseModel):
    """JSON schema the formalizer must follow. Keep it in sync with docs/ir.md."""
    variables: dict[str, Any]
    all_different: list[list[str]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    objective: Optional[dict[str, str]] = None


# Compact IR description for the model. You may edit it, but keep it consistent with ir_solver.py.
IR_SPEC = """IR format (JSON):
{
  "variables": {"<name>": [lo, hi]  or  {"values": [v1, v2, ...]}},     integer variables and their domains
  "all_different": [["<name>", ...], ...],                                 groups whose values must all differ
  "constraints": ["<expression>", ...],                                    every expression must be true
  "objective": {"maximize": "<expression>"}  or  {"minimize": "<expression>"}   ONLY for optimisation problems
}
Expressions may use: integer numbers, variable names, + - * // %, == != < <= > >=, and, or, not, abs(), parentheses.
Inside arithmetic a comparison counts as 1 (true) or 0 (false), e.g. "(A == 1) + (B == 1) == 1".
Variable names must be identifiers: letters, digits and underscores only, no spaces."""

SYSTEM = "You translate logic puzzles into a formal constraint representation. You answer only with JSON."

FORMALIZER_PROMPT = """Translate the puzzle below into the IR format. Do not solve the puzzle.

{ir_spec}

Rules:
1. VARIABLES. Make one variable for EVERY entity or letter mentioned in the puzzle, also letters that appear only in the conditions.
   Its value is what the puzzle asks for: a position, a slot number, a digit, an amount.
2. LABELS AND SLOTS. If entities are matched with labels (people with drinks), the variable is the entity and its value is the
   number of the label: number the labels 1, 2, 3, ... in the order they are mentioned. Ordered slots (days, time slots) are numbered
   1, 2, 3, ... in time order. The domain is then [1, number of labels or slots].
3. EVERY CLAUSE. Write each condition of the puzzle as a separate constraint, exactly as stated. Do not invent conditions.
   Add facts the text relies on but does not state: the order of days, what "last" or "between" mean, that a leading digit is not 0,
   the total number of slots. Use all_different when entities must take different values.
4. ENCODINGS (position 1 is the front).
   - "A is in front of B" or "A is before B": "A < B". "A is behind B" or "A is after B": "A > B".
   - "A is directly behind B": "A == B + 1". "A is next to B": "abs(A - B) == 1".
   - "A is between B and C": "((B < A) and (A < C)) or ((C < A) and (A < B))". "at most k": "<= k". "at least k": ">= k".
   - "either A is first or B is first": "(A == 1) or (B == 1)".
   - A superlative ("the youngest", "the heaviest", "the last") is a comparison with EVERY other variable of the group, one
     constraint each, never a fixed number: if X is the smallest among X, Y and Z write "X < Y" and "X < Z".
   - A number with digits a, b, c is "100*a + 10*b + c": give EVERY digit its own place value.
   - Days with k slots each: number all slots 1, 2, ..., N in time order. The day of slot s is "(s - 1) // k + 1".
   - A statement S made by a person P who is either a knight (always true) or a knave (always false) is "P == (S)",
     with knight = 1 and knave = 0.
5. FORMAT. Integers only: words such as tea or Monday never appear in domains or constraints (no quotes, no strings), replace them by
   their numbers from rule 2. A domain is [lo, hi] with exactly two numbers: for 3 labels write [1, 3], never [1, 2, 3].
   For a list of allowed values use {{"values": [1, 2, 4]}}. Variable names are single words of letters, digits and underscores:
   "Design review" becomes Design_review. Use only the syntax listed above: "==" for equality, never "=", and no max(), min(),
   sum(), lists or other functions except abs(). Every constraint is one short expression. Never write comments or repeated text.
   Add "objective" only if the puzzle asks to maximize or minimize something.
6. CHECK before you answer (do not write the check): every name used in a constraint is declared in "variables"; every domain has two
   numbers; every clause of the puzzle has its own constraint.

Examples:

Example 1.
Puzzle: Three trains leave from platforms 1, 2 and 3: an intercity, a regional and an express, one per platform. The express does not leave from platform 1. The regional leaves from a platform with a lower number than the intercity. The intercity does not leave from platform 3.
IR:
{{
  "variables": {{"Intercity": [1, 3], "Regional": [1, 3], "Express": [1, 3]}},
  "all_different": [["Intercity", "Regional", "Express"]],
  "constraints": ["Express != 1", "Regional < Intercity", "Intercity != 3"]
}}

Example 2 (labels become numbers).
Puzzle: Ann and Ben each own a different pet, a cat or a dog. Ann does not own the cat.
IR (labels numbered in the order they are mentioned: cat = 1, dog = 2):
{{
  "variables": {{"Ann": [1, 2], "Ben": [1, 2]}},
  "all_different": [["Ann", "Ben"]],
  "constraints": ["Ann != 1"]
}}

Now the real puzzle. Answer with the IR JSON only.

Puzzle:
{text}
{feedback}"""


def formalize(problem: dict, feedback: Optional[str] = None) -> dict:
    """Ask the model for an IR encoding of the problem. `feedback` = error messages from a previous attempt."""
    fb = f"\nYour previous encoding was rejected:\n{feedback}\nFix these problems.\n" if feedback else ""
    prompt = FORMALIZER_PROMPT.format(ir_spec=IR_SPEC, text=problem["text"], feedback=fb)
    parsed, raw = llm.ask_json(prompt, IRModel, system=SYSTEM)
    return parsed.model_dump(exclude_none=True)


def is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def validate_ir(ir: dict) -> list:
    if not isinstance(ir, dict):
        return ["the IR must be a JSON object"]

    errors = []
    allowed_keys = {"variables", "all_different", "constraints", "objective"}
    extra_keys = set(ir) - allowed_keys
    if extra_keys:
        errors.append(
            f"unknown IR field(s): {', '.join(sorted(map(str, extra_keys)))}"
        )

    variables = ir.get("variables")
    if not isinstance(variables, dict) or not variables:
        return errors + ['"variables" must be a non-empty object {name: domain}']

    for name, dom in variables.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ):
            errors.append(
                f"variable name {name!r} is not an identifier; "
                "use letters, digits and underscores, starting with a letter or underscore"
            )
            continue

        if isinstance(dom, list):
            if len(dom) != 2 or not all(is_int(x) for x in dom):
                errors.append(
                    f"domain of {name} must be [lo, hi] with exactly two integers, "
                    f"got {dom!r}; use {{'values': [...]}} for a non-contiguous domain"
                )
            elif dom[0] > dom[1]:
                errors.append(
                    f"domain of {name} is empty: lo {dom[0]} is greater than hi {dom[1]}"
                )

        elif isinstance(dom, dict):
            if set(dom) != {"values"}:
                errors.append(
                    f'domain of {name} must contain only the "values" field, got {dom!r}'
                )
                continue

            vals = dom.get("values")
            if not isinstance(vals, list) or not vals or not all(
                is_int(x) for x in vals
            ):
                errors.append(
                    f'domain of {name} must be {{"values": [integers]}}, got {dom!r}'
                )
            elif len(vals) != len(set(vals)):
                errors.append(
                    f"domain of {name} contains duplicate values: {vals!r}"
                )
        else:
            errors.append(
                f"domain of {name} must be [lo, hi] or "
                + '{"values": [...]}'
                + f", got {dom!r}"
            )

    groups = ir.get("all_different", [])
    if not isinstance(groups, list):
        errors.append('"all_different" must be a list of lists of variable names')
    else:
        for index, group in enumerate(groups, start=1):
            if not isinstance(group, list) or len(group) < 2 or not all(
                isinstance(item, str) for item in group
            ):
                errors.append(
                    f"all_different group #{index} must contain at least two variable names"
                )
                continue

            if len(group) != len(set(group)):
                errors.append(
                    f"all_different group #{index} contains duplicate variable names"
                )

            for name in group:
                if name not in variables:
                    errors.append(f"all_different uses unknown variable: {name}")

    constraints = ir.get("constraints", [])
    if not isinstance(constraints, list):
        errors.append('"constraints" must be a list of strings')
        constraints = []

    structurally_valid = not any(
        message.startswith(("variable name ", "domain of "))
        for message in errors
    )

    if structurally_valid:
        base = {"variables": variables}

        for index, constraint in enumerate(constraints, start=1):
            if not isinstance(constraint, str) or not constraint.strip():
                errors.append(
                    f"constraint #{index} must be a non-empty string"
                )
                continue

            try:
                ir_solver.build({
                    **base,
                    "constraints": [constraint]
                })
            except ValueError as exc:
                errors.append(f"constraint {constraint!r}: {exc}")
            except SyntaxError:
                errors.append(
                    f"constraint {constraint!r}: invalid syntax "
                    "(use == for equality; only numbers, variables, arithmetic, "
                    "comparisons, and/or/not, abs() and parentheses are allowed)"
                )
            except Exception as exc:
                errors.append(
                    f"constraint {constraint!r}: {exc.__class__.__name__}: {exc}"
                )

    objective = ir.get("objective")
    if objective is not None:
        if not (
            isinstance(objective, dict)
            and len(objective) == 1
            and next(iter(objective), None) in ("maximize", "minimize")
            and isinstance(next(iter(objective.values()), None), str)
            and next(iter(objective.values())).strip()
        ):
            errors.append(
                '"objective" must be {"maximize": "<expression>"} '
                'or {"minimize": "<expression>"}'
            )
        elif structurally_valid:
            expression = next(iter(objective.values()))
            try:
                _, _, convert = ir_solver.build({"variables": variables})
                convert(ast.parse(expression, mode="eval"))
            except ValueError as exc:
                errors.append(f"objective {expression!r}: {exc}")
            except SyntaxError:
                errors.append(f"objective {expression!r}: invalid syntax")
            except Exception as exc:
                errors.append(
                    f"objective {expression!r}: {exc.__class__.__name__}: {exc}"
                )

    return errors


def run_solver(ir: dict) -> dict:
    return ir_solver.solve(ir, limit=2)


PRESENT_PROMPT = """A puzzle has been solved by a constraint solver. Rewrite the solver's result in the required
answer format. Do NOT solve the puzzle again - only translate the solver's assignment.

Puzzle:
{text}

Encoding used by the solver:
{ir}

Solver assignment (variable -> value):
{assignment}

{answer_format}"""


def present(problem: dict, ir: dict, result: dict) -> dict:
    status = result["status"]
    if status == "none":
        return Answer(status="none").model_dump()
    prompt = PRESENT_PROMPT.format(text=problem["text"], ir=json.dumps(ir, ensure_ascii=False),
                                   assignment=json.dumps(result["solutions"][0], ensure_ascii=False),
                                   answer_format=problem["answer_format"])
    parsed, raw = llm.ask_json(prompt, Answer)
    return Answer(status=status, solution=parsed.solution, objective=result.get("objective")).model_dump()


def solve(problem: dict) -> dict:
    """Returns {"answer": <Answer as dict>, "trace": {...}}. Raise HybridError if no answer can be produced."""
    trace = {"attempts": []}

    ir = formalize(problem)
    errors = validate_ir(ir)
    trace["attempts"].append({"ir": ir, "errors": errors})

    if errors:
        ir = formalize(problem, feedback="\n".join(errors))
        errors = validate_ir(ir)
        trace["attempts"].append({"ir": ir, "errors": errors})
    if errors:
        raise HybridError("invalid IR: " + "; ".join(errors))

    try:
        result = run_solver(ir)
    except ValueError as e:  # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {"status": result["status"], "n_solutions": len(result["solutions"]),
                       "objective": result.get("objective"),
                       "first_solution": result["solutions"][0] if result["solutions"] else None}

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}
