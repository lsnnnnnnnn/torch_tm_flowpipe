"""Parser for Flow* ODE right-hand-side expression strings.

Produces the SAME abstract syntax tree Flow*'s bison grammar builds
(flowstar-toolbox/modelParser.y / modelLexer.l) — deliberately NOT sympy:
sympy auto-simplifies (x - x -> 0, constant folding, term reordering), which
would change the evaluation ORDER of Taylor-model operations and hence the
rounding/caching behavior we must mirror for parity.

Grammar subset (M2: polynomial ODEs; M5 adds / and the elementary functions):

    expression := expression '+' expression        %left
                | expression '-' expression        %left
                | expression '*' expression        %left, tighter
                | expression '/' expression        %left, tighter   (M5)
                | '-' expression                   %nonassoc, tighter still
                | expression '^' NUM               %right, tightest
                | '(' expression ')'
                | NUM | IDENT
                | 'sin(' e ')' | 'cos(' | 'exp(' | 'log(' | 'sqrt('  (M5)

Precedence is exactly modelParser.y:53-57: {+,-} < {*,/} < unary minus < ^.
So "-x^2" is -(x^2) and "-x*y" is (-x)*y, matching Flow*.

Number literals: decimal like Flow*'s lexer; parsed with Python float() —
both float() and MPFR-53/RNDN produce the correctly-rounded binary64, so
constants are BIT-IDENTICAL to Flow*'s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- AST node types (frozen: the compiler treats them as values) ------------


@dataclass(frozen=True)
class Num:
    """Numeric literal. [value] is the correctly-rounded binary64."""

    value: float


@dataclass(frozen=True)
class NumIv:
    """Interval literal "[lo, hi]" — Flow*'s uncertain constants (ODE<Interval>,
    e.g. higgins_selkov's [0.9995,1.0005]). Both endpoints correctly-rounded
    binary64 like Num."""

    lo: float
    hi: float


@dataclass(frozen=True)
class Var:
    """State-variable reference by index (Flow* declaration order)."""

    index: int


@dataclass(frozen=True)
class Un:
    """Unary op: 'neg' (M2) | 'sin' 'cos' 'exp' 'log' 'sqrt' (M5)."""

    op: str
    a: Node


@dataclass(frozen=True)
class Bin:
    """Binary op: '+' '-' '*' (M2) | '/' (M5) | '^' (b must be Num, integer)."""

    op: str
    a: Node
    b: Node


Node = Num | NumIv | Var | Un | Bin

_FUNCTIONS = ("sin", "cos", "exp", "log", "sqrt")

# Token regex: numbers (with optional exponent), identifiers, single-char ops.
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<op>[()+\-*/^\[\],]))"
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Split into (kind, lexeme) tokens; kind in {num, ident, op}."""
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if m is None:
            # Nothing matched at pos (skip pure trailing whitespace).
            if text[pos:].strip() == "":
                break
            raise ValueError(f"unexpected character {text[pos]!r} at position {pos} in {text!r}")
        pos = m.end()
        # Exactly one alternation group matched, so the break always fires.
        for kind in ("num", "ident", "op"):  # pragma: no cover  # loop exit unreachable: a successful match has exactly one non-None group
            lexeme = m.group(kind)
            if lexeme is not None:
                tokens.append((kind, lexeme))
                break
    return tokens


class _Parser:
    """Precedence-climbing parser over the token list (see module docstring)."""

    def __init__(self, tokens: list[tuple[str, str]], var_ids: dict[str, int], text: str):
        self.tokens = tokens
        self.pos = 0
        self.var_ids = var_ids
        self.text = text

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ValueError(f"unexpected end of expression: {self.text!r}")
        self.pos += 1
        return tok

    def expect_op(self, op: str) -> None:
        tok = self.take()
        if tok != ("op", op):
            raise ValueError(f"expected {op!r}, got {tok[1]!r} in {self.text!r}")

    def _signed_number(self) -> float:
        """A possibly-negated numeric literal (interval-endpoint grammar)."""
        neg = False
        while self.peek() == ("op", "-"):
            self.take()
            neg = not neg
        kind, lexeme = self.take()
        if kind != "num":
            raise ValueError(f"expected a number in interval literal in {self.text!r}")
        v = float(lexeme)
        return -v if neg else v

    # expression := term (('+'|'-') term)*
    def expression(self) -> Node:
        node = self.term()
        while (tok := self.peek()) is not None and tok[1] in ("+", "-") and tok[0] == "op":
            self.take()
            node = Bin(tok[1], node, self.term())
        return node

    # term := unary (('*'|'/') unary)*
    def term(self) -> Node:
        node = self.unary()
        while (tok := self.peek()) is not None and tok[1] in ("*", "/") and tok[0] == "op":
            self.take()
            node = Bin(tok[1], node, self.unary())
        return node

    # unary := '-' unary | power     (uminus binds tighter than * but looser than ^)
    def unary(self) -> Node:
        tok = self.peek()
        if tok == ("op", "-"):
            self.take()
            return Un("neg", self.unary())
        return self.power()

    # power := atom ('^' NUM)*      (right-assoc; exponent must be an integer literal,
    #                                as in Flow*'s grammar: expression '^' NUM)
    def power(self) -> Node:
        node = self.atom()
        while self.peek() == ("op", "^"):
            self.take()
            kind, lexeme = self.take()
            if kind != "num":
                raise ValueError(f"exponent must be a numeric literal in {self.text!r}")
            exponent = float(lexeme)
            if exponent != int(exponent) or exponent < 0:
                raise ValueError(f"exponent must be a nonnegative integer, got {lexeme!r}")
            node = Bin("^", node, Num(float(int(exponent))))
        return node

    # atom := '(' expression ')' | '[' NUM ',' NUM ']' | NUM | IDENT
    #        | func '(' expression ')'
    def atom(self) -> Node:
        kind, lexeme = self.take()
        if (kind, lexeme) == ("op", "["):
            lo = self._signed_number()
            self.expect_op(",")
            hi = self._signed_number()
            self.expect_op("]")
            if lo > hi:
                raise ValueError(f"interval literal [{lo}, {hi}] has lo > hi in {self.text!r}")
            return NumIv(lo, hi)
        if (kind, lexeme) == ("op", "("):
            node = self.expression()
            self.expect_op(")")
            return node
        if kind == "num":
            return Num(float(lexeme))
        if kind == "ident":
            if lexeme in _FUNCTIONS:
                self.expect_op("(")
                arg = self.expression()
                self.expect_op(")")
                return Un(lexeme, arg)
            if lexeme not in self.var_ids:
                raise ValueError(f"unknown variable {lexeme!r} in {self.text!r}")
            return Var(self.var_ids[lexeme])
        raise ValueError(f"unexpected token {lexeme!r} in {self.text!r}")


def parse(text: str, var_names: list[str]) -> Node:
    """Parse one RHS string against the declared state-variable names.

    var_names is the Flow* declaration order; Var.index refers into it.
    """
    var_ids = {name: i for i, name in enumerate(var_names)}
    parser = _Parser(_tokenize(text), var_ids, text)
    node = parser.expression()
    if parser.peek() is not None:
        raise ValueError(f"trailing tokens after expression: {text!r}")
    return node
