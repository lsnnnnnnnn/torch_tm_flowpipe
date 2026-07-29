from __future__ import annotations

from generate_report import (
    _at_requested_horizon,
    _carry_loss,
    _winner_text,
)


def _row(
    *,
    tool: str,
    protocol: str,
    time: float,
    width: float,
) -> dict[str, object]:
    return {
        "tool": tool,
        "variant": "candidate",
        "protocol": protocol,
        "system": "van_der_pol",
        "h": "0.01",
        "horizon": "1.0",
        "time": str(time),
        "width": str(width),
    }


def test_requested_horizon_filter_rejects_failed_prefix() -> None:
    assert _at_requested_horizon(
        _row(
            tool="torch",
            protocol="common_affine_carry",
            time=1.0,
            width=1.0,
        )
    )
    assert not _at_requested_horizon(
        _row(
            tool="flowstar",
            protocol="common_affine_carry",
            time=0.13,
            width=0.2,
        )
    )


def test_winner_text_never_groups_different_absolute_times() -> None:
    text = _winner_text(
        [
            _row(
                tool="short_prefix",
                protocol="common_affine_carry",
                time=0.13,
                width=0.2,
            ),
            _row(
                tool="full_horizon",
                protocol="common_affine_carry",
                time=1.0,
                width=1.0,
            ),
        ]
    )
    assert "t=0.13" in text
    assert "t=1" in text


def test_affine_box_ratio_requires_identical_absolute_time() -> None:
    affine = [
        _row(
            tool="torch",
            protocol="common_affine_carry",
            time=1.0,
            width=1.0,
        )
    ]
    mismatched_box = [
        _row(
            tool="torch",
            protocol="common_box_carry",
            time=0.5,
            width=2.0,
        )
    ]
    assert _carry_loss(affine, mismatched_box) == []

    matched_box = [
        _row(
            tool="torch",
            protocol="common_box_carry",
            time=1.0,
            width=2.0,
        )
    ]
    loss = _carry_loss(affine, matched_box)
    assert len(loss) == 1
    assert loss[0][-1] == 2.0
