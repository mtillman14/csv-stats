"""lme4-style model formulas for :mod:`csvstats.lmm`: the ONE owner of the
grammar (.claude/plan-statstudio-model-formula.md, scistack repo).

``parse`` turns ``"y ~ group * time * x + (1 + x | subject) + (1 | item)"``
into a :class:`ModelFormula`; ``render`` writes one back, so
``render(parse(s))`` is ``s`` in canonical spelling. Errors carry the
position of the offending text (:class:`FormulaError`), so an editor can
underline it.

What the grammar accepts is what :func:`csvstats.lmm.lmm` can fit:

- **Fixed part:** ONE crossing of factors and covariates,
  ``a * b * x``, or ``1`` for an intercept-only model. ``lmm`` fits the full
  factorial with the covariates crossed in, so ``a + b`` (main effects
  only), ``a * b + x`` and a bare ``a:b`` are refused with that reason
  instead of being fitted as something else.
- **Random terms:** ``(lhs | g)`` and ``(lhs || g)``, as lme4. ``lhs`` is
  ``1``, ``0`` and names joined by ``+``. The intercept is implied unless
  ``0`` is given (``(x | g)`` is ``(1 + x | g)``). ``g`` is a name, an
  interaction ``a:b``, or lme4's nesting shorthand ``a/b``, which expands
  to ``(… | a) + (… | a:b)``.
- **Names:** R identifiers, or anything in backticks (`` `speed (m/s)` ``).

Which name is a factor and which a covariate is NOT the grammar's
business: the caller knows the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lmm_fit import INTERCEPT, RandomTerm, quote_name, random_formula


class FormulaError(ValueError):
    """A formula the grammar refuses. ``position``/``length``: the offending
    span in the text given to :func:`parse` (0-based)."""

    def __init__(self, message: str, position: int, length: int = 1) -> None:
        super().__init__(f"{message} (at character {position + 1})")
        self.reason = message
        self.position = position
        self.length = max(1, length)


@dataclass(frozen=True)
class ModelFormula:
    """``fixed``: the crossed names, in order (empty = intercept only).
    ``random``: the random terms after ``/`` expansion. ``response``: the
    left side, or None when only the right side was given."""

    fixed: tuple[str, ...] = ()
    random: tuple[RandomTerm, ...] = ()
    response: str | None = None
    #: Every name the formula reads (fixed names, slopes, groups), in order.
    names: tuple[str, ...] = field(default=(), compare=False)


# --- tokens -------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(?P<name>[A-Za-z_.][A-Za-z0-9_.]*)|`(?P<quoted>[^`]*)`|(?P<num>\d+)"
                    r"|(?P<op>\|\||[~+*:|()/-]))")


@dataclass(frozen=True)
class _Tok:
    kind: str  # "name" | "num" | an operator | "end"
    text: str
    pos: int
    length: int


def _tokens(text: str) -> list[_Tok]:
    out, i = [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        m = _TOKEN.match(text, i)
        if m is None or m.end() == i:
            if text[i] == "`":
                raise FormulaError("a backtick-quoted name is not closed", i, len(text) - i)
            raise FormulaError(f"unexpected character {text[i]!r}", i)
        start = m.start(m.lastgroup)
        if m.group("name") is not None:
            out.append(_Tok("name", m.group("name"), start, m.end() - start))
        elif m.group("quoted") is not None:
            if not m.group("quoted").strip():
                raise FormulaError("an empty quoted name", start - 1, m.end() - start + 1)
            out.append(_Tok("name", m.group("quoted"), start - 1, m.end() - start + 1))
        elif m.group("num") is not None:
            out.append(_Tok("num", m.group("num"), start, m.end() - start))
        else:
            op = m.group("op")
            out.append(_Tok(op, op, start, len(op)))
        i = m.end()
    out.append(_Tok("end", "", len(text), 1))
    return out


# --- parser -------------------------------------------------------------------

class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.toks = _tokens(text)
        self.i = 0

    @property
    def tok(self) -> _Tok:
        return self.toks[self.i]

    def take(self, kind: str | None = None) -> _Tok:
        t = self.tok
        if kind is not None and t.kind != kind:
            want = "a name" if kind == "name" else repr(kind)
            got = "the end" if t.kind == "end" else repr(t.text)
            raise FormulaError(f"expected {want}, found {got}", t.pos, t.length)
        self.i += 1
        return t

    def formula(self) -> ModelFormula:
        response = None
        if any(t.kind == "~" for t in self.toks):
            if self.tok.kind == "name" and self.toks[self.i + 1].kind == "~":
                response = self.take("name").text
            elif self.tok.kind != "~":
                raise FormulaError("the left side is one name (the measure)", self.tok.pos, self.tok.length)
            self.take("~")
        if self.tok.kind == "end":
            raise FormulaError("the right side is empty", self.tok.pos)
        fixed: tuple[str, ...] | None = None
        random: list[RandomTerm] = []
        while True:
            start = self.tok
            if start.kind == "(":
                random.extend(self.random_term())
            elif start.kind == "-":
                raise FormulaError("removing terms (-) is not supported", start.pos)
            else:
                names = self.fixed_product()
                if fixed is not None:
                    raise FormulaError(
                        "the fixed part is ONE crossing of factors and covariates (csv-stats fits the full "
                        f"factorial): write {' * '.join(quote_name(n) for n in (*fixed, *names)) or '1'} "
                        "instead of joining fixed terms with +", start.pos, self.tok.pos - start.pos)
                fixed = names
            if self.tok.kind == "+":
                self.take("+")
                continue
            if self.tok.kind == "-":
                raise FormulaError("removing terms (-) is not supported", self.tok.pos)
            if self.tok.kind != "end":
                t = self.tok
                raise FormulaError(f"expected + or the end, found {t.text!r}", t.pos, t.length)
            break
        names: list[str] = []
        for n in (*(fixed or ()), *(c for r in random for c in (*r.terms, *r.group) if c != INTERCEPT)):
            if n not in names:
                names.append(n)
        return ModelFormula(fixed=fixed or (), random=tuple(random), response=response, names=tuple(names))

    def fixed_product(self) -> tuple[str, ...]:
        first = self.tok
        if first.kind == "num":
            self.take()
            if first.text != "1":
                raise FormulaError(f"{first.text} is not a fixed term (1 = intercept only)", first.pos, first.length)
            if self.tok.kind in ("*", ":"):
                raise FormulaError("1 cannot be crossed with anything", self.tok.pos)
            return ()
        names = [self.take("name")]
        while self.tok.kind in ("*", ":"):
            op = self.take()
            if op.kind == ":":
                raise FormulaError(
                    f"an interaction alone ({names[-1].text}:…) is not supported: csv-stats fits the full "
                    "factorial, so write a * b", op.pos)
            names.append(self.take("name"))
        seen: set[str] = set()
        for t in names:
            if t.text in seen:
                raise FormulaError(f"{t.text!r} appears twice in the fixed part", t.pos, t.length)
            seen.add(t.text)
        return tuple(t.text for t in names)

    def random_term(self) -> list[RandomTerm]:
        open_ = self.take("(")
        terms: list[str] = []
        intercept: bool | None = None
        while True:
            t = self.tok
            if t.kind == "num":
                self.take()
                if t.text not in ("0", "1"):
                    raise FormulaError(f"{t.text} is not a random-effect term (1 = intercept, 0 = none)",
                                       t.pos, t.length)
                intercept = t.text == "1" if intercept is None else intercept and t.text == "1"
            elif t.kind == "name":
                name = self.take().text
                if self.tok.kind in ("*", ":"):
                    raise FormulaError("random-effect terms are joined with +, one column each",
                                       self.tok.pos)
                if name in terms:
                    raise FormulaError(f"{name!r} appears twice in this random term", t.pos, t.length)
                terms.append(name)
            else:
                raise FormulaError("expected 1, 0 or a name inside ( … | group )", t.pos, t.length)
            if self.tok.kind == "+":
                self.take()
                continue
            break
        if self.tok.kind not in ("|", "||"):
            t = self.tok
            raise FormulaError(f"expected | or || before the grouping, found {t.text or 'the end'!r}",
                               t.pos, t.length)
        correlated = self.take().kind == "|"
        paths = [self.group_path()]
        while self.tok.kind == "/":
            self.take()
            paths.append(self.group_path())
        close = self.tok
        if close.kind != ")":
            raise FormulaError("this ( is not closed", open_.pos, close.pos - open_.pos)
        self.take()
        cols = ((INTERCEPT,) if intercept is not False else ()) + tuple(terms)
        if not cols:
            raise FormulaError("a random term with no intercept and no slope", open_.pos, close.pos - open_.pos + 1)
        out, prefix = [], ()
        for path in paths:  # a/b = (… | a) + (… | a:b)
            prefix = prefix + path
            out.append(RandomTerm(prefix, cols, correlated))
        return out

    def group_path(self) -> tuple[str, ...]:
        parts = [self.take("name").text]
        while self.tok.kind == ":":
            self.take()
            parts.append(self.take("name").text)
        return tuple(parts)


def parse(text: str) -> ModelFormula:
    """Parse a formula, with or without its left side (``"~ a * b + …"`` or
    ``"a * b + …"`` too). Raises :class:`FormulaError` with a position."""
    if not text or not text.strip():
        raise FormulaError("the formula is empty", 0)
    return _Parser(text).formula()


def render(model: ModelFormula, response: str | None = None) -> str:
    """The canonical spelling: ``y ~ a * b + (1 + x | g)``. ``response``
    overrides the model's own left side; with neither, the right side
    alone."""
    fixed = " * ".join(quote_name(n) for n in model.fixed) or "1"
    rhs = " + ".join(p for p in (fixed, random_formula(model.random)) if p)
    lhs = response if response is not None else model.response
    return f"{quote_name(lhs)} ~ {rhs}" if lhs else rhs
