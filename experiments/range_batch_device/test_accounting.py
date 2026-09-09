"""Count test identities once; preserve and explicitly resolve initial failures."""
from collections import Counter
import xml.etree.ElementTree as ET
from .common import RUN,read,numerical_sources


def account(root=RUN, *, include_evidence=True):
    names=["root"]
    if (root/"tests/root_path_retry.command.json").exists():names.append("root_path_retry")
    if include_evidence and (root/"tests/evidence.command.json").exists():names.append("evidence")
    current={};initial_failures=[];groups={};commands=[]
    for name in names:
        command=read(root/f"tests/{name}.command.json")
        assert "exit_code" in command,"check still running"
        assert command["numerical_sources"]==numerical_sources()
        assert int((root/command["exit"]).read_text())==command["exit_code"]
        cases=list(ET.parse(root/command["xml"]).iter("testcase"))
        counts=Counter()
        for case in cases:
            identity=case.get("classname"),case.get("name")
            status="failed" if case.find("failure") is not None or case.find("error") is not None else "skipped" if case.find("skipped") is not None else "passed"
            if status=="failed":initial_failures.append(identity)
            counts[status]+=1;current[identity]=status
        groups[name]=dict(counts);commands.append(command)
    assert current and all(v!="failed" for v in current.values()),"unresolved current test failure"
    assert all(current[i]=="passed" for i in initial_failures)
    return dict(unique_totals=dict(Counter(current.values())),groups=groups,
        initial_failures_resolved=[list(i) for i in initial_failures],repeated_checks_added_to_total=False),commands
