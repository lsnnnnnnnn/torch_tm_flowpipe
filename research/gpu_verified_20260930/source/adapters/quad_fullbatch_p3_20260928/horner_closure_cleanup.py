"""Isolated lifetime-only repair of the frozen recursive Horner evaluator.

Only the final two original statements are wrapped in try/finally, clearing the
recursive function cell after the return expression or on exception. No tensor
arithmetic, graph, pool, keepalive or original source file is modified.
"""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,inspect,textwrap

SOURCE_SHA='d090e314f710d1d4763ea299336e0dba4dbada642b43700f09fec7c238977f4f'
POLICY='clear_recursive_horner_evaluate_cell_after_original_return_expression'
OLD='    coeff, rem, support = evaluate(_tree(sup_a, eng))\n    return coeff, iv.add(rem, a_rem), support\n'
NEW='    try:\n        coeff, rem, support = evaluate(_tree(sup_a, eng))\n        return coeff, iv.add(rem, a_rem), support\n    finally:\n        evaluate = None\n'

def install(horner):
    path=Path(horner.__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE_SHA
    original=horner.compose_horner
    assert original.__qualname__=='compose_horner' and original.__globals__ is horner.__dict__
    assert Path(inspect.getfile(original)).resolve()==path.resolve()
    source=textwrap.dedent(inspect.getsource(original))
    assert source.count(OLD)==1 and source.endswith(OLD)
    changed=source.replace(OLD,NEW)
    before=ast.parse(source);after=ast.parse(changed)
    function=after.body[0];tail=function.body[-1]
    assert isinstance(tail,ast.Try) and not tail.handlers and not tail.orelse
    assert ast.dump(tail.finalbody[0])==ast.dump(ast.parse('evaluate = None').body[0]) and len(tail.finalbody)==1
    function.body[-1:]=tail.body
    assert ast.dump(before)==ast.dump(after),'The inverse cleanup transform must recover the exact original AST'
    namespace=dict(horner.__dict__)
    exec(compile(changed,str(Path(__file__).resolve())+'::isolated_compose_horner','exec'),namespace)
    candidate=namespace['compose_horner'];horner.compose_horner=candidate
    def restore():
        assert horner.compose_horner is candidate
        horner.compose_horner=original
    return SimpleNamespace(policy=POLICY,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_source_sha256=SOURCE_SHA,original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        candidate_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        inverse_AST_equal=True,original=original,candidate=candidate,restore=restore)
