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
import json
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
1. Make one variable per entity that must be placed or measured (a person, an item, a letter, an event).
   Its value is what the puzzle asks for: a position, a slot number, a digit, an amount.
2. If entities are matched with labels (for example people with drinks), the variable is the entity and its value
   is the number of the label. Number the labels 1, 2, 3, ... in the order they are mentioned in the puzzle.
   The same rule applies to ordered slots (days, time slots): number them 1, 2, 3, ... in time order.
3. If the puzzle says that entities take different values, add them to all_different.
4. Write every condition of the puzzle as a separate constraint, exactly as stated. Do not invent conditions.
5. Also write down facts that the text relies on but does not state (the order of days, what "last", "between"
   or "directly after" mean, that a leading digit is not 0, the total number of slots, and so on).
6. Encodings: "A is directly behind B" -> "A == B + 1"; "A is before B" -> "A < B"; "A or B" ->
   "(X == 1) or (Y == 1)"; a multi-digit number with digits a, b, c -> "100*a + 10*b + c"; a statement S made by a
   person P who is either a knight (always true) or a knave (always false) -> "P == (S)" with knight = 1, knave = 0.
7. Use only the syntax listed above: "==" for equality, no "=", no words, no strings, no function except abs().
8. A domain [lo, hi] is a range of exactly two numbers, for example [1, 5]. To list allowed values use {{"values": [1, 2, 4]}}. Never write a list of three or more numbers directly.

Examples:

Example 1.
Puzzle: Three trains leave from platforms 1, 2 and 3: an intercity, a regional and an express, one per platform. The express does not leave from platform 1. The regional leaves from a platform with a lower number than the intercity. The intercity does not leave from platform 3.
IR:
{{
  "variables": {{"Intercity": [1, 3], "Regional": [1, 3], "Express": [1, 3]}},
  "all_different": [["Intercity", "Regional", "Express"]],
  "constraints": ["Express != 1", "Regional < Intercity", "Intercity != 3"]
}}

Example 2.
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


def validate_ir(ir: dict) -> list:
    """Return a list of human-readable error messages; an empty list means the IR is valid.

    TODO(team). Things to check:
      * every domain is [lo, hi] with lo <= hi, or {"values": [...]} with integers;
      * every name used in all_different / constraints / objective is a declared variable;
      * every constraint parses and uses only allowed syntax.
    Hint: ir_solver.build(ir) raises ValueError with a readable message for unknown variables and
    unsupported syntax - you can call it inside try/except and turn the exception into an error message.
    """
    return []


def run_solver(ir: dict) -> dict:
    """Run Z3 on the IR. Returns {"status": ..., "solutions": [...], "objective": ...}.

    TODO(team): with limit=1 the solver stops after the first solution, so it reports "unique" even when
    several solutions exist. Think about what limit you need to tell "unique" from "multiple".
    """
    return ir_solver.solve(ir, limit=1)


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
        raise HybridError("invalid IR: " + "; ".join(errors))

    try:
        result = run_solver(ir)
    except ValueError as e:                      # the solver rejected the IR
        raise HybridError(f"solver rejected the IR: {e}") from e
    trace["solver"] = {"status": result["status"], "n_solutions": len(result["solutions"]),
                       "objective": result.get("objective"),
                       "first_solution": result["solutions"][0] if result["solutions"] else None}

    answer = present(problem, ir, result)
    return {"answer": answer, "trace": trace}
