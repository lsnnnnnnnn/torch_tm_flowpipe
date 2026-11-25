"""ODE right-hand-side compiler: AST -> instruction tape -> three executors.

Flow* evaluates the RHS ASTs three ways during one integration step
(ALGORITHM.md; expression.h):

  P_point  (`evaluate_no_remainder`, expression.h)  — polynomial-only Picard:
           point coefficients, truncation DISCARDS (nctrunc + drop-cutoff).
  P_valid  (`evaluate` with intermediate_ranges, expression.h:1481) —
           validated pass: interval coefficients, every truncation/cutoff mass
           goes into remainders, and each MUL/CONST/POW-step CACHES interval
           ranges into a list.
  P_replay (`evaluate_remainder`, expression.h:1833) — remainder-only
           refinement: pure interval arithmetic over candidate remainders,
           consuming the cache IN THE SAME ORDER it was written.

We compile each AST once into a flat instruction tape; the three executors
below interpret the SAME tape, guaranteeing the cache slot layout matches
between the valid pass (writer) and the replay pass (reader) by construction.
Interpreter overhead is per-instruction (tapes are 5-40 instructions), while
every instruction is a large batched tensor op — the standard tape-VM tradeoff.

Cache layout (mirrors Flow*'s single global `intermediate_ranges` list):
components are evaluated in ODE order, each pushing slots in traversal order —
MUL pushes 3 (rangeP1, rangeP2, trunc+round), CONST pushes 1, each executed
multiply inside POW pushes 3, VAR pushes 0.

GOTCHAS #16 (soundness gap in Flow*, discovered here): the valid pass ctruncs
the polynomial iterate at each VAR leaf and folds the truncation tail into the
node's remainder, but the replay's VAR returns ONLY the candidate remainder —
the tail is silently missing from every refinement iteration, so Flow*'s
refined remainders can under-approximate the true Picard image by the tail
ranges. In parity mode we replicate this; in strict mode the executors store
the tails per leaf (`strict_tails`) and the replay adds them back (sound).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mpmath
import torch

from . import elementary as elem
from . import interval as iv
from . import polynomial as poly
from .expr_parser import Bin, Node, Num, NumIv, Un, Var, parse
from .monomials import MonomialTables
from .polynomial import StepTables

# Elementary function ops (M5): unary series dispatched to elementary.py.
_ELEM_FUNCS = ("sin", "cos", "exp", "log", "sqrt")
# Ops whose series/Lagrange formulas divide by DATA (rec inside div, log's /c
# and log(), sqrt's /2c and sqrt()) and can therefore flag domain violations.
_BAD_OPS = ("div", "log", "sqrt")

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instr:
    """One tape instruction; unused fields keep their defaults.

    op in {"var", "const", "neg", "add", "sub", "mul", "pow"} plus the M5 ops
    {"div", "sin", "cos", "exp", "log", "sqrt"} (dispatched to elementary.py).
    dst/a/b are slot indices; `cache_base` is the first interval-cache slot
    this instruction reads/writes and `cache_len` how many (order-dependent
    for the M5 ops — elementary.cache_len); `strict_slot` indexes the strict
    tail buffer for "var" instructions.
    """

    op: str
    dst: int
    a: int = -1
    b: int = -1
    var: int = -1
    const: float = 0.0
    # Interval-constant remainder after midpoint removal (ODE<Interval>, e.g.
    # higgins_selkov's [0.9995,1.0005]): the "const" carries mid = RN((lo+hi)/2)
    # and this outward-rounded [lo-mid, hi-mid] radius. Zero for point Nums.
    crem_lo: float = 0.0
    crem_hi: float = 0.0
    power: int = 0
    cache_base: int = -1
    cache_len: int = 0
    strict_slot: int = -1


def _pow_mul_count(degree: int) -> int:
    """Number of TM multiplies Flow*'s square-and-multiply POW executes.

    Mirrors the exact loop in expression.h (OPT_POW): for i = degree-1; i > 0:
    one multiply if (i & 1), then i >>= 1, then one multiply if i > 0.
    """
    count = 0
    i = degree - 1
    while i > 0:
        if i & 1:
            count += 1
        i >>= 1
        if i > 0:
            count += 1
    return count


@dataclass
class CompiledODE:
    """A compiled ODE right-hand side f: R^n -> R^n.

    Attributes:
        n: state dimension.
        var_names: declaration-order state variable names.
        instrs: the flat tape (all components concatenated, ODE order).
        out_slots [n]: slot holding each component's f_i result.
        n_slots: slot-pool size.
        n_cache: interval cache slots (= length of Flow*'s intermediate_ranges).
        n_strict: strict tail slots (one per "var" instruction).
        max_degree: highest ^ exponent in the system (table sizing aid).
        order: the series/truncation order the VALIDATED executors run at
            (compile_ode's `order` argument); None for pure-polynomial tapes.
            Elementary-function cache layouts are sized from it, so exec_valid
            must be called with exactly this k (it checks).
        has_elem: tape contains div/elementary instructions.
        needs_bad: tape contains ops that can flag domain violations
            (div/log/sqrt) — exec_valid/exec_replay then REQUIRE a `bad_out`
            lane mask so failures cannot be dropped silently.
    """

    n: int
    var_names: list[str]
    instrs: list[Instr] = field(default_factory=list)
    out_slots: list[int] = field(default_factory=list)
    n_slots: int = 0
    n_cache: int = 0
    n_strict: int = 0
    max_degree: int = 1
    order: int | None = None
    has_elem: bool = False
    needs_bad: bool = False


def _fold_const(node: Node) -> float | None:
    """Evaluate a pure-constant subtree to its float64 value, or None.

    WHY (GOTCHAS #20): Flow*'s series take a sentinel shortcut on constant
    arguments that makes cache length DATA-dependent — fatal for a static
    batched tape — so function-of-constant and constant/constant nodes are
    folded at COMPILE time instead (mirroring Flow*'s zero-remainder constant
    case, which also yields a bare constant TM).

    Arithmetic on doubles is IEEE RN = MPFR-53 RNDN, bit-identical to Flow*'s
    Real path; ^ mirrors the OPT_POW square-and-multiply chain. sin/cos/exp/
    log are evaluated CORRECTLY ROUNDED via mpmath (matching what MPFR RN
    would produce; Python's libm math.* has no such guarantee), sqrt via
    math.sqrt (IEEE-correctly-rounded by spec). The folded constant is a
    single double: its <= 0.5 ulp representation error is NOT tracked — same
    class as every Num literal in the tape (and as Flow*'s unbounded Real
    roundoff, GOTCHAS #15). Domain violations (log(c<=0), sqrt(c<0), x/0)
    raise ValueError at compile time — Flow* would exit(1) at runtime.
    """
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Un):
        v = _fold_const(node.a)
        if v is None:
            return None
        if node.op == "neg":
            return -v
        if node.op == "log" and v <= 0.0:
            raise ValueError(f"log of non-positive constant {v}")
        if node.op == "sqrt":
            if v < 0.0:
                raise ValueError(f"sqrt of negative constant {v}")
            return math.sqrt(v)
        with mpmath.workdps(40):
            return float(getattr(mpmath, node.op)(mpmath.mpf(v)))
    if isinstance(node, Bin):
        a = _fold_const(node.a)
        if a is None:
            return None
        if node.op == "^":
            degree = int(node.b.value)
            result = 1.0
            if degree > 0:
                result, temp = a, a
                i = degree - 1
                while i > 0:
                    if i & 1:
                        result *= temp
                    i >>= 1
                    if i > 0:
                        temp *= temp
            return result
        b = _fold_const(node.b)
        if b is None:
            return None
        if node.op == "/":
            if b == 0.0:
                raise ValueError("constant division by zero")
            return a / b
        return {"+": a + b, "-": a - b, "*": a * b}[node.op]
    return None  # Var


def compile_ode(
    rhs: list[str],
    var_names: list[str],
    order: int | None = None,
    require_square: bool = True,
) -> CompiledODE:
    """Parse and compile all RHS component strings into one tape.

    Components are compiled in ODE order into a single instruction list with a
    single cache-slot counter — exactly Flow*'s one global intermediate_ranges
    list written across components sequentially.

    `order` (M5): the series/truncation order k the VALIDATED pass will run at
    — Flow* passes k = TM order - 1 into `evaluate` (Picard_ctrunc_normal,
    TaylorModel.h:3711), and flowpipe.reach does the same. REQUIRED whenever
    the RHS contains division or elementary functions (their static cache
    layout depends on it, elementary.cache_len); ignored for polynomial tapes.
    """
    if require_square and len(rhs) != len(var_names):
        # ODE right-hand sides are square (one component per state variable);
        # constraint tapes (safety.py) legitimately compile single expressions
        # over the full variable set.
        raise ValueError(f"{len(rhs)} RHS strings for {len(var_names)} variables")
    if order is not None and order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    code = CompiledODE(n=len(var_names), var_names=list(var_names), order=order)

    def emit_const(value: float, crem_lo: float = 0.0, crem_hi: float = 0.0) -> int:
        dst = code.n_slots
        code.n_slots += 1
        code.instrs.append(
            Instr(op="const", dst=dst, const=value, crem_lo=crem_lo, crem_hi=crem_hi,
                  cache_base=code.n_cache, cache_len=1)
        )
        code.n_cache += 1
        return dst

    def emit_elem(op: str, dst_args: dict) -> int:
        """Emit a div/elementary instruction with its order-sized cache block."""
        if order is None:
            # NotImplementedError (not ValueError): compiling non-polynomial
            # RHS without an order is the pre-M5 contract — the cache layout
            # is order-sized, so an order-less tape cannot implement these ops.
            name = "division" if op == "div" else op
            raise NotImplementedError(
                f"{name} needs compile_ode(..., order=): the elementary cache "
                "layout is order-sized (M5)"
            )
        dst = code.n_slots
        code.n_slots += 1
        length = elem.cache_len(op, order)
        code.instrs.append(
            Instr(op=op, dst=dst, cache_base=code.n_cache, cache_len=length, **dst_args)
        )
        code.n_cache += length
        code.has_elem = True
        if op in _BAD_OPS:
            code.needs_bad = True
        return dst

    def emit(node: Node) -> int:
        """Post-order emit; returns the slot holding the node's value."""
        if isinstance(node, Num):
            return emit_const(node.value)
        if isinstance(node, NumIv):
            # Flow* NODE_CONST with an interval constant (expression.h:1637):
            # I.remove_midpoint(c) -> polynomial part c = RN midpoint,
            # remainder = [lo - c, hi - c] with directed rounding. We take one
            # nextafter step outward on each remainder endpoint (sound,
            # <= 1 ulp wider than MPFR's directed subtraction).
            import math as _math

            mid = (node.lo + node.hi) * 0.5  # RN, like Interval::remove_midpoint
            rl = node.lo - mid
            rh = node.hi - mid
            rl = _math.nextafter(rl, -_math.inf) if rl != 0.0 else 0.0
            rh = _math.nextafter(rh, _math.inf) if rh != 0.0 else 0.0
            return emit_const(mid, crem_lo=rl, crem_hi=rh)
        if isinstance(node, Var):
            dst = code.n_slots
            code.n_slots += 1
            code.instrs.append(Instr(op="var", dst=dst, var=node.index, strict_slot=code.n_strict))
            code.n_strict += 1
            return dst
        if isinstance(node, Un):
            if node.op != "neg":
                folded = _fold_const(node)  # GOTCHAS #20: fold f(constant)
                if folded is not None:
                    return emit_const(folded)
                return emit_elem(node.op, {"a": emit(node.a)})
            a = emit(node.a)
            dst = code.n_slots
            code.n_slots += 1
            code.instrs.append(Instr(op="neg", dst=dst, a=a))
            return dst
        if isinstance(node, Bin):
            if node.op == "^":
                a = emit(node.a)
                degree = int(node.b.value)  # parser guarantees integer Num
                code.max_degree = max(code.max_degree, degree)
                dst = code.n_slots
                code.n_slots += 2  # dst (result) and dst+1 (the `temp` register)
                n_mults = _pow_mul_count(degree)
                code.instrs.append(
                    Instr(
                        op="pow",
                        dst=dst,
                        a=a,
                        power=degree,
                        cache_base=code.n_cache,
                        cache_len=3 * n_mults,
                    )
                )
                code.n_cache += 3 * n_mults
                return dst
            if node.op == "/":
                folded = _fold_const(node)  # GOTCHAS #20: fold const/const
                if folded is not None:
                    return emit_const(folded)
                return emit_elem("div", {"a": emit(node.a), "b": emit(node.b)})
            a = emit(node.a)
            b = emit(node.b)
            dst = code.n_slots
            code.n_slots += 1
            if node.op == "*":
                code.instrs.append(
                    Instr(op="mul", dst=dst, a=a, b=b, cache_base=code.n_cache, cache_len=3)
                )
                code.n_cache += 3
            elif node.op == "+":
                code.instrs.append(Instr(op="add", dst=dst, a=a, b=b))
            elif node.op == "-":
                code.instrs.append(Instr(op="sub", dst=dst, a=a, b=b))
            else:  # pragma: no cover  # unreachable: parser emits only + - * / ^
                raise AssertionError(f"unknown binary op {node.op!r}")
            return dst
        raise AssertionError(f"unknown AST node {node!r}")  # pragma: no cover  # unreachable: parser output is total

    for component in rhs:
        code.out_slots.append(emit(parse(component, var_names)))

    return code


# ---------------------------------------------------------------------------
# P_point — polynomial-only evaluation (Flow* evaluate_no_remainder)
# ---------------------------------------------------------------------------


def exec_point(
    code: CompiledODE,
    x: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
) -> torch.Tensor:
    """Evaluate f(x) with point coefficients, truncating at degree k.

    x [B, n, T] (the polynomial Picard iterate over the working basis) ->
    f(x) [B, n, T] (each component's result, zero beyond its actual degree).

    Truncation semantics mirror evaluate_no_remainder exactly: VAR leaves
    nctrunc (discard beyond degree k); MUL = full product, nctrunc, drop-cutoff
    (mul_no_remainder). All arithmetic tier-P RN — identical to Flow* on Real.
    """
    B, n, T = x.shape
    keep = tables.prefix_len_cpu[k + 1]  # K_k: monomials of degree <= k

    slots = torch.zeros(B, code.n_slots, T, dtype=x.dtype, device=x.device)

    def mul_no_remainder(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """[B, T] x [B, T] -> [B, T]: product, nctrunc(k), cutoff-drop."""
        return elem.mul_no_remainder(a, b, k, tables, cutoff_threshold)

    _point_series = {
        "sin": elem.sin_series_point,
        "cos": elem.cos_series_point,
        "exp": elem.exp_series_point,
        "log": elem.log_series_point,
        "sqrt": elem.sqrt_series_point,
    }

    for ins in code.instrs:
        if ins.op == "var":
            v = torch.zeros(B, T, dtype=x.dtype, device=x.device)
            v[..., :keep] = x[:, ins.var, :keep]  # nctrunc at k
            slots[:, ins.dst] = v
        elif ins.op == "const":
            slots[:, ins.dst, 0] = ins.const
        elif ins.op == "neg":
            slots[:, ins.dst] = -slots[:, ins.a]
        elif ins.op == "add":
            slots[:, ins.dst] = slots[:, ins.a] + slots[:, ins.b]
        elif ins.op == "sub":
            slots[:, ins.dst] = slots[:, ins.a] - slots[:, ins.b]
        elif ins.op == "mul":
            slots[:, ins.dst] = mul_no_remainder(slots[:, ins.a], slots[:, ins.b])
        elif ins.op == "div":
            # OPT_DIV (expression.h:1251): rec series on the denominator's
            # POLYNOMIAL, then numerator * that with nctrunc + cutoff.
            rec_b = elem.rec_series_point(slots[:, ins.b], k, tables, cutoff_threshold)
            slots[:, ins.dst] = mul_no_remainder(slots[:, ins.a], rec_b)
        elif ins.op in _point_series:
            # UNA_OPT branch of evaluate_no_remainder (expression.h:1205-1223).
            slots[:, ins.dst] = _point_series[ins.op](
                slots[:, ins.a], k, tables, cutoff_threshold
            )
        elif ins.op == "pow":
            # Mirrors expression.h OPT_POW (no-remainder variant) bit by bit.
            degree = ins.power
            if degree == 0:
                slots[:, ins.dst] = 0.0
                slots[:, ins.dst, 0] = 1.0
            else:
                result = slots[:, ins.a].clone()  # [B, T]
                temp = slots[:, ins.a].clone()
                i = degree - 1
                while i > 0:
                    if i & 1:
                        result = mul_no_remainder(result, temp)
                    i >>= 1
                    if i > 0:
                        temp = mul_no_remainder(temp, temp)
                slots[:, ins.dst] = result
        else:  # pragma: no cover  # unreachable: compile_ode emits only known ops
            raise AssertionError(f"bad op {ins.op}")

    return torch.stack([slots[:, s] for s in code.out_slots], dim=1)  # [B, n, T]


# ---------------------------------------------------------------------------
# P_valid — validated evaluation with range caching (Flow* evaluate + caches)
# ---------------------------------------------------------------------------


def _elem_prologue(
    code: CompiledODE, k: int, bad_out: torch.Tensor | None, device: str
) -> elem.ElemTables | None:
    """Shared M5 entry checks for exec_valid/exec_replay.

    Returns the elementary constant tables (None for polynomial tapes).
    Enforces the two M5 contracts: the runtime series order must equal the
    compile-time order the cache layout was sized for, and tapes that can flag
    domain violations must be given a `bad_out` mask to report into.
    """
    if not code.has_elem:
        return None
    if k != code.order:
        raise ValueError(
            f"tape compiled for order {code.order} but executed at k={k}: "
            "the elementary cache layout is order-sized"
        )
    if code.needs_bad and bad_out is None:
        raise ValueError(
            "bad_out lane mask required: the tape contains div/log/sqrt, whose "
            "domain violations must not be dropped silently"
        )
    return elem.elem_tables(code.order, device)


def exec_valid(
    code: CompiledODE,
    x: torch.Tensor,
    x_rem: torch.Tensor,
    k: int,
    tables: MonomialTables,
    step: StepTables,
    cutoff_threshold: float,
    bad_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate f over the TM (x, x_rem) with interval coefficients + caching.

    x [B, n, T] point coefficients of the Picard iterate (Flow* passes the
    Real-coefficient TMs; conversion to interval coefficients at the leaves is
    exact); x_rem [B, n, 2] the current remainder guess.

    bad_out [B] bool (M5): lane mask that div/log/sqrt domain violations are
    OR-ed into IN PLACE; the flagged lanes' outputs are finite garbage and the
    caller MUST fail them (flowpipe: FAILED_DIV). REQUIRED when the tape can
    flag (code.needs_bad) so failures cannot be dropped silently; the return
    shape stays a 4-tuple so polynomial-tape callers are unaffected.

    Returns:
        coeffs_iv [B, n, T, 2]: interval coefficients of each f_i result.
        rem [B, n, 2]: result remainders.
        cache [B, n_cache, 2]: the intermediate-ranges list, Flow* order.
        strict_tails [B, n_strict, 2]: VAR-leaf truncation tails (GOTCHAS #16;
            consumed by exec_replay only in strict mode).
    """
    B, n, T = x.shape
    dev, dt = x.device, x.dtype

    tabs = _elem_prologue(code, k, bad_out, str(dev))

    coeff_slots = torch.zeros(B, code.n_slots, T, 2, dtype=dt, device=dev)
    rem_slots = torch.zeros(B, code.n_slots, 2, dtype=dt, device=dev)
    cache = torch.zeros(B, code.n_cache, 2, dtype=dt, device=dev)
    strict_tails = torch.zeros(B, max(code.n_strict, 1), 2, dtype=dt, device=dev)

    def write_poly(dst: int, kept_iv: torch.Tensor) -> None:
        """Store a [B, K', 2] prefix into a zeroed full-width slot."""
        coeff_slots[:, dst] = 0.0
        coeff_slots[:, dst, : kept_iv.shape[-2]] = kept_iv

    def mul_valid(
        a_coeffs: torch.Tensor,
        a_rem: torch.Tensor,
        b_coeffs: torch.Tensor,
        b_rem: torch.Tensor,
        cache_at: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TaylorModel<Interval>::mul_insert_ctrunc_normal, batched — see
        elementary.tm_mul_valid (shared with the series loops)."""
        return elem.tm_mul_valid(
            a_coeffs, a_rem, b_coeffs, b_rem, k, tables, step, cutoff_threshold, cache, cache_at
        )

    _valid_series = {
        "sin": elem.sin_series_valid,
        "cos": elem.cos_series_valid,
        "exp": elem.exp_series_valid,
    }
    _valid_series_bad = {"log": elem.log_series_valid, "sqrt": elem.sqrt_series_valid}

    for ins in code.instrs:
        if ins.op == "var":
            v_iv = iv.from_point(x[:, ins.var])  # [B, T, 2] exact conversion
            kept, tail = poly.ctrunc_normal_iv(v_iv, tables, step, k)
            write_poly(ins.dst, kept)
            rem_slots[:, ins.dst] = iv.add(x_rem[:, ins.var], tail)
            strict_tails[:, ins.strict_slot] = tail
        elif ins.op == "const":
            # Flow*: I = [lo, hi]; remove_midpoint -> poly const = midpoint,
            # remainder = I - mid (exactly [0, 0] for point constants, the
            # compile-time radius for interval constants); the remainder is
            # ALSO the cached slot the replay pass re-reads (expression.h
            # NODE_CONST: intermediate_ranges.push_back(I)).
            coeff_slots[:, ins.dst] = 0.0
            coeff_slots[:, ins.dst, 0, 0] = ins.const
            coeff_slots[:, ins.dst, 0, 1] = ins.const
            rem_slots[:, ins.dst, 0] = ins.crem_lo
            rem_slots[:, ins.dst, 1] = ins.crem_hi
            cache[:, ins.cache_base, 0] = ins.crem_lo
            cache[:, ins.cache_base, 1] = ins.crem_hi
        elif ins.op == "neg":
            coeff_slots[:, ins.dst] = iv.neg(coeff_slots[:, ins.a])
            rem_slots[:, ins.dst] = iv.neg(rem_slots[:, ins.a])
        elif ins.op == "add":
            coeff_slots[:, ins.dst] = iv.add(coeff_slots[:, ins.a], coeff_slots[:, ins.b])
            rem_slots[:, ins.dst] = iv.add(rem_slots[:, ins.a], rem_slots[:, ins.b])
        elif ins.op == "sub":
            coeff_slots[:, ins.dst] = iv.sub(coeff_slots[:, ins.a], coeff_slots[:, ins.b])
            rem_slots[:, ins.dst] = iv.sub(rem_slots[:, ins.a], rem_slots[:, ins.b])
        elif ins.op == "mul":
            kept, rem = mul_valid(
                coeff_slots[:, ins.a],
                rem_slots[:, ins.a],
                coeff_slots[:, ins.b],
                rem_slots[:, ins.b],
                ins.cache_base,
            )
            write_poly(ins.dst, kept)
            rem_slots[:, ins.dst] = rem
        elif ins.op == "div":
            # OPT_DIV (expression.h:1554): rec_taylor on the denominator (its
            # own cache block), then mul_insert_ctrunc_normal with 3 pushes.
            rec_c, rec_r, bad = elem.rec_series_valid(
                coeff_slots[:, ins.b], rem_slots[:, ins.b], k, tables, step,
                cutoff_threshold, cache, ins.cache_base, tabs,
            )
            bad_out |= bad
            kept, rem = mul_valid(
                coeff_slots[:, ins.a], rem_slots[:, ins.a], rec_c, rec_r,
                ins.cache_base + 3 * k + 3,
            )
            write_poly(ins.dst, kept)
            rem_slots[:, ins.dst] = rem
        elif ins.op in _valid_series:
            out_c, out_r = _valid_series[ins.op](
                coeff_slots[:, ins.a], rem_slots[:, ins.a], k, tables, step,
                cutoff_threshold, cache, ins.cache_base, tabs,
            )
            coeff_slots[:, ins.dst] = out_c
            rem_slots[:, ins.dst] = out_r
        elif ins.op in _valid_series_bad:
            out_c, out_r, bad = _valid_series_bad[ins.op](
                coeff_slots[:, ins.a], rem_slots[:, ins.a], k, tables, step,
                cutoff_threshold, cache, ins.cache_base, tabs,
            )
            bad_out |= bad
            coeff_slots[:, ins.dst] = out_c
            rem_slots[:, ins.dst] = out_r
        elif ins.op == "pow":
            degree = ins.power
            if degree == 0:
                coeff_slots[:, ins.dst] = 0.0
                coeff_slots[:, ins.dst, 0, 0] = 1.0
                coeff_slots[:, ins.dst, 0, 1] = 1.0
                rem_slots[:, ins.dst] = 0.0
            else:
                res_c = coeff_slots[:, ins.a].clone()
                res_r = rem_slots[:, ins.a].clone()
                tmp_c = res_c.clone()
                tmp_r = res_r.clone()
                cursor = ins.cache_base
                i = degree - 1
                while i > 0:
                    if i & 1:
                        kept, res_r = mul_valid(res_c, res_r, tmp_c, tmp_r, cursor)
                        cursor += 3
                        res_c = torch.zeros_like(res_c)
                        res_c[..., : kept.shape[-2], :] = kept
                    i >>= 1
                    if i > 0:
                        kept, tmp_r = mul_valid(tmp_c, tmp_r, tmp_c, tmp_r, cursor)
                        cursor += 3
                        tmp_c = torch.zeros_like(tmp_c)
                        tmp_c[..., : kept.shape[-2], :] = kept
                coeff_slots[:, ins.dst] = res_c
                rem_slots[:, ins.dst] = res_r
        else:  # pragma: no cover  # unreachable: compile_ode emits only known ops
            raise AssertionError(f"bad op {ins.op}")

    out_c = torch.stack([coeff_slots[:, s] for s in code.out_slots], dim=1)  # [B, n, T, 2]
    out_r = torch.stack([rem_slots[:, s] for s in code.out_slots], dim=1)  # [B, n, 2]
    return out_c, out_r, cache, strict_tails


# ---------------------------------------------------------------------------
# P_replay — remainder-only refinement (Flow* evaluate_remainder)
# ---------------------------------------------------------------------------


def exec_replay(
    code: CompiledODE,
    cand_rem: torch.Tensor,
    cache: torch.Tensor,
    strict_tails: torch.Tensor,
    strict: bool,
    bad_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Recompute result remainders from candidate remainders + cached ranges.

    cand_rem [B, n, 2] (the current refinement candidates); cache
    [B, n_cache, 2] from exec_valid. Returns rem [B, n, 2].

    Pure [B, 2]-sized interval arithmetic — no polynomial work — replicating
    evaluate_remainder's formulas verbatim (MUL: c1*R2 + c2*R1 + R1*R2 + c3,
    in that add order). In strict mode the VAR leaves add back their
    truncation tails (repairing GOTCHAS #16); parity mode reproduces Flow*.

    bad_out [B] bool (M5): as in exec_valid — the Lagrange tails of div/log/
    sqrt divide by data and can flag lanes even here (their domains only ever
    SHRINK during refinement, so a lane that passed exec_valid stays clean in
    practice; the mask is a required defensive channel, not a hot path).
    """
    B = cand_rem.shape[0]
    tabs = _elem_prologue(code, code.order, bad_out, str(cand_rem.device))
    rem_slots = torch.zeros(B, code.n_slots, 2, dtype=cand_rem.dtype, device=cand_rem.device)

    def replay_mul(r1: torch.Tensor, r2: torch.Tensor, at: int) -> torch.Tensor:
        """result = c1*R2 + c2*R1 + R1*R2 + c3 — Flow*'s exact order."""
        out = iv.mul(cache[:, at], r2)
        out = iv.add(out, iv.mul(cache[:, at + 1], r1))
        out = iv.add(out, iv.mul(r1, r2))
        out = iv.add(out, cache[:, at + 2])
        return out

    _replay_series = {
        "sin": elem.sin_series_replay,
        "cos": elem.cos_series_replay,
        "exp": elem.exp_series_replay,
    }
    _replay_series_bad = {"log": elem.log_series_replay, "sqrt": elem.sqrt_series_replay}

    for ins in code.instrs:
        if ins.op == "var":
            r = cand_rem[:, ins.var]
            if strict:
                r = iv.add(r, strict_tails[:, ins.strict_slot])
            rem_slots[:, ins.dst] = r
        elif ins.op == "const":
            rem_slots[:, ins.dst] = cache[:, ins.cache_base]
        elif ins.op == "neg":
            rem_slots[:, ins.dst] = iv.neg(rem_slots[:, ins.a])
        elif ins.op == "add":
            rem_slots[:, ins.dst] = iv.add(rem_slots[:, ins.a], rem_slots[:, ins.b])
        elif ins.op == "sub":
            rem_slots[:, ins.dst] = iv.sub(rem_slots[:, ins.a], rem_slots[:, ins.b])
        elif ins.op == "mul":
            rem_slots[:, ins.dst] = replay_mul(
                rem_slots[:, ins.a], rem_slots[:, ins.b], ins.cache_base
            )
        elif ins.op == "div":
            # OPT_DIV (expression.h:1900): rec twin on the denominator's
            # candidate, then the generic MUL replay on its cache triple.
            rec_r, bad = elem.rec_series_replay(
                rem_slots[:, ins.b], cache, ins.cache_base, code.order, tabs
            )
            bad_out |= bad
            rem_slots[:, ins.dst] = replay_mul(
                rem_slots[:, ins.a], rec_r, ins.cache_base + 3 * code.order + 3
            )
        elif ins.op in _replay_series:
            rem_slots[:, ins.dst] = _replay_series[ins.op](
                rem_slots[:, ins.a], cache, ins.cache_base, code.order, tabs
            )
        elif ins.op in _replay_series_bad:
            r, bad = _replay_series_bad[ins.op](
                rem_slots[:, ins.a], cache, ins.cache_base, code.order, tabs
            )
            bad_out |= bad
            rem_slots[:, ins.dst] = r
        elif ins.op == "pow":
            degree = ins.power
            if degree == 0:
                rem_slots[:, ins.dst] = 0.0
            else:
                result = rem_slots[:, ins.a].clone()
                temp = result.clone()
                cursor = ins.cache_base
                i = degree - 1
                while i > 0:
                    if i & 1:
                        result = replay_mul(result, temp, cursor)
                        cursor += 3
                    i >>= 1
                    if i > 0:
                        temp = replay_mul(temp, temp, cursor)
                        cursor += 3
                rem_slots[:, ins.dst] = result
        else:  # pragma: no cover  # unreachable: compile_ode emits only known ops
            raise AssertionError(f"bad op {ins.op}")

    return torch.stack([rem_slots[:, s] for s in code.out_slots], dim=1)  # [B, n, 2]
