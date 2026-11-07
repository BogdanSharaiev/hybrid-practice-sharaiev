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

# TODO(team): this prompt is the heart of the hybrid pipeline. Things worth considering:
#   * how should the model choose variables? (one variable per entity? per attribute value?)
#   * how should it encode positions, days, "morning/afternoon", "next to", "between", "at most"?
#   * facts that the text does not state but that the model must know (calendar, working week, ...);
#   * a worked example (few-shot) on a puzzle that is NOT in your set;
#   * how to encode knights and knaves: a statement S by person X becomes  X == (S).
FORMALIZER_PROMPT = """Translate the puzzle below into the IR format. Do not solve the puzzle.

{ir_spec}

Rules:
1. Make one variable for EVERY entity or letter mentioned in the puzzle (also letters that appear only in the conditions). Its value is what the puzzle asks for: a position, a slot number, a digit, an amount.
2. If entities are matched with labels (for example people with drinks), the variable is the entity and its value
   is the number of the label. Number the labels 1, 2, 3, ... in the order they are mentioned in the puzzle.
   The same rule applies to ordered slots (days, time slots): number them 1, 2, 3, ... in time order.
3. If the puzzle says that entities take different values, add them to all_different.
4. Write every condition of the puzzle as a separate constraint, exactly as stated. Do not invent conditions.
5. Also write down facts that the text relies on but does not state (the order of days, what "last", "between"
   or "directly after" mean, that a leading digit is not 0, the total number of slots, and so on).
6. Encodings (position 1 is the front): "A is in front of B" or "A is before B" -> "A < B"; "A is behind B" or "A is after B" -> "A > B";
   "A is directly behind B" -> "A == B + 1"; "A or B" -> "(X == 1) or (Y == 1)"; a multi-digit number with digits a, b, c -> "100*a + 10*b + c"
   (give EVERY digit its own place value); a statement S made by a person P who is either a knight (always true) or a knave
   (always false) -> "P == (S)" with knight = 1, knave = 0.
7. Use only the syntax listed above: "==" for equality, no "=", no words, no strings, no lists, no function except abs(). Never use max(), min() or sum().
8. A domain [lo, hi] is a range of exactly two numbers, for example [1, 5]. To list allowed values use {{"values": [1, 2, 4]}}. Never write a list of three or more numbers directly.
9. Variable names are single words made of letters, digits and underscores. Replace spaces by underscores: "Design review" -> Design_review.
10. The IR contains integers only. Words such as tea, coffee or Monday never appear in domains or constraints (no quotes, no strings): replace every label by its number from rule 2, and write the domain of the variable as [1, number of labels].
11. A superlative ("the youngest", "the smallest", "the largest", "the last") means a comparison with EVERY other variable of the group, one constraint each: if X is the youngest among A, B and C write "X < A", "X < B", "X < C" (use > for the largest). Never fix it to a constant.
12. If time is divided into days with several slots each, give every slot its own number 1, 2, ..., N in time order, so the domain is [1, N] with N the total number of slots. The day of slot s is computed with integer division, for example (s + 1) // 2 for two slots per day.
13. Every constraint is one short expression. Never write comments, explanations or repeated text inside the JSON.

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

Example 3 (days and slots).
Puzzle: Four talks P, Q, R and S are given on Thursday and Friday. Each day has a first and a second slot, and each talk takes one slot. P is on Thursday. Q is in a second slot. R is on the day after S.
IR (slots numbered in time order: Thu first = 1, Thu second = 2, Fri first = 3, Fri second = 4; the day of slot s is (s + 1) // 2):
{{
  "variables": {{"P": [1, 4], "Q": [1, 4], "R": [1, 4], "S": [1, 4]}},
  "all_different": [["P", "Q", "R", "S"]],
  "constraints": ["P <= 2", "Q % 2 == 0", "(R + 1) // 2 == (S + 1) // 2 + 1"]
}}

Example 4 (optimization).
Choosing items within a budget: each variable 0 or 1 means "do not take / take".
{{
  "variables": {{"Tent": [0, 1], "Stove": [0, 1], "Camera": [0, 1]}},
  "constraints": ["4*Tent + 3*Stove + 5*Camera <= 8"],
  "objective": {{"maximize": "7*Tent + 4*Stove + 6*Camera"}}
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
    """Return a list of human-readable error messages; an empty list means the IR is valid.

    TODO(team). Things to check:
      * every domain is [lo, hi] with lo <= hi, or {"values": [...]} with integers;
      * every name used in all_different / constraints / objective is a declared variable;
      * every constraint parses and uses only allowed syntax.
    Hint: ir_solver.build(ir) raises ValueError with a readable message for unknown variables and
    unsupported syntax - you can call it inside try/except and turn the exception into an error message.
    """
    errors = []
    if not isinstance(ir, dict):
        return ["the IR must be a JSON object"]
    variables = ir.get("variables")
    if not isinstance(variables, dict) or not variables:
        return ['"variables" must be a non-empty object {name: domain}']

    for name, dom in variables.items():
        if not re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$").match(name):
            errors.append(f"variable name {name!r} is not an identifier; rename it using letters, digits and underscores only, "
                          f"for example {re.sub(r'[^A-Za-z0-9_]', '_', name)!r}, and use the new name everywhere")
        if isinstance(dom, list):
            if len(dom) != 2 or not all(is_int(x) for x in dom):
                errors.append(f"domain of {name} must be [lo, hi] with exactly two integers, got {dom!r}; "
                              'write the range as [smallest, largest], e.g. [1, 3], or use {"values": [1, 2, 3]}. '
                              "Only integers are allowed: replace words (tea, coffee, ...) by numbers 1, 2, 3, ... "
                              "in the order they are mentioned and use these numbers in the constraints")
            elif dom[0] > dom[1]:
                errors.append(f"domain of {name} is empty: lo {dom[0]} is greater than hi {dom[1]}")
        elif isinstance(dom, dict):
            vals = dom.get("values")
            if not isinstance(vals, list) or not vals or not all(is_int(x) for x in vals):
                errors.append(f'domain of {name} must be {{"values": [integers]}}, got {dom!r}')
        else:
            errors.append(f"domain of {name} must be [lo, hi] or " + '{"values": [...]}' + f", got {dom!r}")

    groups = ir.get("all_different", [])
    if not isinstance(groups, list):
        errors.append('"all_different" must be a list of lists of variable names')
    else:
        for grp in groups:
            if not isinstance(grp, list) or not all(isinstance(g, str) for g in grp):
                errors.append(f"all_different group {grp!r} must be a list of variable names")
                continue
            for g in grp:
                if g not in variables:
                    errors.append(f"all_different uses unknown variable: {g}")

    cons = ir.get("constraints", [])
    if not isinstance(cons, list):
        errors.append('"constraints" must be a list of strings')
        cons = []
    if errors:
        return errors

    base = {"variables": variables}
    for c in cons:
        if not isinstance(c, str):
            errors.append(f"constraint {c!r} must be a string")
            continue
        try:
            ir_solver.build({**base, "constraints": [c]})
        except ValueError as e:
            errors.append(f"constraint {c!r}: {e}")
        except SyntaxError:
            errors.append(f"constraint {c!r}: invalid syntax (use == for equality; only numbers, variables, "
                          "+ - * // %, comparisons, and/or/not, abs() and parentheses are allowed)")
        except Exception as e:  # noqa: BLE001
            errors.append(f"constraint {c!r}: {e.__class__.__name__}: {e}")

    obj = ir.get("objective")
    if obj is not None:
        if not (isinstance(obj, dict) and len(obj) == 1 and next(iter(obj)) in ("maximize", "minimize")
                and isinstance(next(iter(obj.values())), str)):
            errors.append('"objective" must be {"maximize": "<expression>"} or {"minimize": "<expression>"}')
        else:
            expr = next(iter(obj.values()))
            try:
                _, _, conv = ir_solver.build(base)
                conv(ast.parse(expr, mode="eval"))
            except ValueError as e:
                errors.append(f"objective {expr!r}: {e}")
            except SyntaxError:
                errors.append(f"objective {expr!r}: invalid syntax")
    return errors


def run_solver(ir: dict) -> dict:
    """Run Z3 on the IR. Returns {"status": ..., "solutions": [...], "objective": ...}.

    TODO(team): with limit=1 the solver stops after the first solution, so it reports "unique" even when
    several solutions exist. Think about what limit you need to tell "unique" from "multiple".
    """

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
    """Turn the solver result into the required answer format.

    Provided baseline: the model translates the assignment. This is a weak point - the model can
    introduce errors AFTER the solver. Status and objective are always taken from the solver, never
    from the model. Extension: write deterministic code for problem types where the mapping is obvious.
    """
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

    # TODO(team): if `errors` is not empty, make ONE more attempt:
    #   ir = formalize(problem, feedback="\n".join(errors)); errors = validate_ir(ir); record it in trace.
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
