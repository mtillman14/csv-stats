"""csvstats.lmm_formula: the lme4-style formula grammar.

Stage 1 of .claude/plan-statstudio-model-formula.md (scistack repo). Besides
the grammar's own cases, every sampled scenario's R-written formula must
parse into the random blocks and fixed terms its JSON records: 225 formulas
lme4 itself fitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from _lmm_oracle import as_list
from csvstats.lmm import lmm
from csvstats.lmm_fit import INTERCEPT, RandomTerm
from csvstats.lmm_formula import FormulaError, ModelFormula, parse, render

RESULTS = Path(__file__).parent / "reference" / "scenarios" / "results"
FIXTURES = sorted(RESULTS.glob("*.json")) if RESULTS.exists() else []


def test_a_full_formula():
    m = parse("y ~ group * time * x + (1 + x | subject) + (1 | item)")
    assert m.response == "y"
    assert m.fixed == ("group", "time", "x")
    assert m.random == (RandomTerm(("subject",), (INTERCEPT, "x")), RandomTerm(("item",)))
    assert m.names == ("group", "time", "x", "subject", "item")


@pytest.mark.parametrize("text, fixed, random", [
    ("~ 1 + (1 | subject)", (), [RandomTerm(("subject",))]),
    ("a + (x | g)", ("a",), [RandomTerm(("g",), (INTERCEPT, "x"))]),            # intercept implied
    ("a + (0 + x | g)", ("a",), [RandomTerm(("g",), ("x",))]),
    ("a + (1 + x || g)", ("a",), [RandomTerm(("g",), (INTERCEPT, "x"), False)]),
    ("a + (1 | site/subject)", ("a",), [RandomTerm(("site",)), RandomTerm(("site", "subject"))]),
    ("a + (1 | subject:time)", ("a",), [RandomTerm(("subject", "time"))]),
    ("`speed (m/s)` * b + (1 | `my subject`)", ("speed (m/s)", "b"), [RandomTerm(("my subject",))]),
    ("(1 | g) + a * b", ("a", "b"), [RandomTerm(("g",))]),                      # order is free
])
def test_grammar(text, fixed, random):
    m = parse(text)
    assert m.fixed == fixed and list(m.random) == random and m.response is None


@pytest.mark.parametrize("text", [
    "y ~ a * b * x + (1 + x | subject) + (1 | item)",
    "y ~ 1 + (1 | subject)",
    "y ~ a + (0 + x || g)",
    "`step length` ~ `speed (m/s)` * group + (1 + `speed (m/s)` | subject:session)",
])
def test_render_is_the_inverse_of_parse(text):
    assert render(parse(text)) == text


def test_render_canonicalises():
    assert render(parse("y~a*b+(x|g)")) == "y ~ a * b + (1 + x | g)"
    assert render(parse("y ~ a + (1 | s/t)")) == "y ~ a + (1 | s) + (1 | s:t)"
    assert render(ModelFormula(fixed=("a",)), response="dv") == "dv ~ a"


@pytest.mark.parametrize("text, reason, at", [
    ("y ~ a + b + (1 | g)", "ONE crossing", 8),
    ("y ~ a * b + x + (1 | g)", "write a * b * x", 12),
    ("y ~ a:b + (1 | g)", "interaction alone", 5),
    ("y ~ a * a + (1 | g)", "appears twice", 8),
    ("y ~ a + (1 + x | g", "not closed", 8),
    ("y ~ a + (1 + x g)", "expected | or ||", 15),
    ("y ~ a + (x * z | g)", "joined with +", 11),
    ("y ~ a + (2 | g)", "not a random-effect term", 9),
    ("y ~ a - 1 + (1 | g)", "removing terms", 6),
    ("y ~ a + (0 | g)", "no intercept and no slope", 8),
    ("y ~ `a + (1 | g)", "not closed", 4),
    ("y ~ a $ b", "unexpected character", 6),
    ("y ~ ", "right side is empty", 4),
    ("", "empty", 0),
    ("y z ~ a", "left side is one name", 0),
])
def test_errors_point_at_the_problem(text, reason, at):
    with pytest.raises(FormulaError) as err:
        parse(text)
    assert reason in err.value.reason
    assert err.value.position == at, (err.value.position, err.value.reason)


@pytest.mark.skipif(not FIXTURES, reason="no sampled scenarios (run make_lmm_scenarios.R)")
@pytest.mark.parametrize("path", FIXTURES, ids=[p.stem for p in FIXTURES])
def test_every_r_formula_parses_into_its_recorded_structure(path):
    """lme4 fitted these formulas; the JSON records their random blocks and
    fixed terms in a language-neutral form. A collinear scenario adds "+ z",
    a second fixed term, which csv-stats cannot fit and the grammar refuses."""
    ref = json.loads(path.read_text())
    if "formula" not in ref:
        pytest.skip("R could not fit it")
    if ref.get("collinear"):
        with pytest.raises(FormulaError, match="ONE crossing"):
            parse(ref["formula"])
        return
    m = parse(ref["formula"])
    assert m.response == "y"
    assert m.fixed == tuple(as_list(ref["factors"])) + tuple(as_list(ref["covariates"]))
    recorded = [RandomTerm(tuple(as_list(b["group"])), tuple(as_list(b["terms"])), bool(b["correlated"]))
                for b in ref["random"]]
    assert sorted(m.random, key=repr) == sorted(recorded, key=repr)
    assert parse(render(m)) == m


def test_too_few_observations_for_the_random_effects_is_refused():
    """lme4's check.nobs.vs.nRE: 2 observations per subject and a random
    intercept and slope (2 effects per subject) cannot be told apart from
    the residual (the RIS_014 shape)."""
    rng = np.random.default_rng(1)
    rows = [{"subject": f"S{s}", "x": x, "y": rng.normal()} for s in range(10) for x in (0.0, 1.0)]
    data = pd.DataFrame(rows)
    with pytest.raises(ValueError, match=r"number of observations \(20\) <= number of random effects \(20\)"):
        lmm(data, "y", subject="subject", covariates=["x"],
            random=[{"group": "subject", "terms": ["(Intercept)", "x"]}])
    lmm(data, "y", subject="subject", covariates=["x"])  # the intercept alone is fine
