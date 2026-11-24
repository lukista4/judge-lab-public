"""Pure-torch tests of judgelab.logits (no vllm).
(1) penalize == the reference penalty step (logits[seen] -= p), duplicates penalised once;
(2) processor via a fake base + fake BatchUpdate: prompt scope penalises prompt | output ids, output scope only output ids, rows without
    the key untouched; the LIVE output list growing between steps is seen; (3) removed rows stop; SWAP exchanges rows, UNIDIRECTIONAL
    moves one; (4) NEGATIVES: a wrong scope string is refused; output scope differs from the prompt rule.
"""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from judgelab import logits as L  # noqa: E402


class FakeBase:
    pass


def bu(added=(), removed=(), moved=()):
    return SimpleNamespace(added=list(added), removed=list(removed), moved=list(moved))


def params(p=None, scope="prompt"):
    return SimpleNamespace(extra_args=({"hfj3_presence": p, "hfj3_scope": scope} if p is not None else None))


def test_penalize_matches_reference_step():
    x = torch.randn(50, generator=torch.Generator().manual_seed(0))
    seen = [3, 7, 7, 49, 0]
    ref = x.float().clone()
    ref[torch.tensor(seen)] -= 1.5   # advanced indexing with a repeated id subtracts ONCE
    assert torch.allclose(L.penalize(x, seen, 1.5), ref)


def _want():
    want = torch.zeros(3, 12)
    want[0, [1, 2, 5]] = -1.5
    want[1, [9]] = -1.5
    return want


def test_processor_scopes_rows_and_moves():
    P = L.make_processor_class(FakeBase)()
    out0 = [5]
    P.update_state(bu(added=[(0, params(1.5, "prompt"), [1, 2, 2], out0), (1, params(1.5, "output"), [1, 2], [9]), (2, params(None), [1], [])]))
    assert torch.equal(P.apply(torch.zeros(3, 12)), _want())
    out0.append(6)   # live list grows
    r2 = P.apply(torch.zeros(3, 12))
    assert r2[0, 6].item() == -1.5 and r2[0, 1].item() == -1.5

    P.update_state(bu(removed=[1]))
    r3 = P.apply(torch.zeros(3, 12))
    P.update_state(bu(moved=[(0, 2, "MoveDirectionality.SWAP")]))
    r4 = P.apply(torch.zeros(3, 12))
    P.update_state(bu(moved=[(2, 1, "MoveDirectionality.UNIDIRECTIONAL")]))
    r5 = P.apply(torch.zeros(3, 12))
    assert r3[1].abs().sum().item() == 0
    assert r4[2, 5].item() == -1.5 and r4[0].abs().sum().item() == 0
    assert r5[1, 5].item() == -1.5 and r5[2].abs().sum().item() == 0


def test_negatives():
    Q = L.make_processor_class(FakeBase)()
    with pytest.raises(ValueError):
        Q.update_state(bu(added=[(0, params(1.5, "both"), [1], [])]))
    O = L.make_processor_class(FakeBase)()
    O.update_state(bu(added=[(0, params(1.5, "output"), [1, 2], [5])]))
    assert not torch.equal(O.apply(torch.zeros(1, 12)), _want()[:1])
