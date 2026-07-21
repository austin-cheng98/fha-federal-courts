import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fha import config
from fha.classify import rule_is_fha

CLAIMS = ["disparate_treatment", "disparate_impact", "refusal_rent_sell",
          "reasonable_accommodation", "zoning_exclusionary"]

# The LLM-baseline helpers may live in the package or beside the other
# validation entry points; import from whichever module supplies them.
CANDIDATES = ["fha.llm_baseline", "fha.llm_eval", "score_llm_baseline",
              "llm_baseline"]
NEEDED = ["majority_vote", "self_consistency", "wilson", "precision_recall_correction",
          "mcnemar_exact"]


def _load_module():
    for name in CANDIDATES:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if all(hasattr(mod, fn) for fn in NEEDED):
            return mod
    raise ImportError(
        "no LLM-baseline module exposing %s; looked for %s"
        % (", ".join(NEEDED), ", ".join(CANDIDATES)))


llm = _load_module()


def _rec(cluster_id, values, framework="none", pass_no=1, set_name="gold"):
    row = {"set": set_name, "cluster_id": cluster_id, "pass": pass_no,
           "framework": framework}
    row.update(dict(zip(CLAIMS, values)))
    return row


def _load_json(name):
    return json.load((config.VALIDATION / name).open())


@pytest.fixture(scope="module")
def raw_labels():
    return _load_json("llm_labels_3pass.json")


@pytest.fixture(scope="module")
def committed_votes():
    return _load_json("llm_majority_votes.json")


@pytest.fixture(scope="module")
def substantive_ids():
    index = _load_json("random_sample_index.json")
    corpus = {r["cluster_id"]: r for r in
              (json.loads(l) for l in
               (config.PROCESSED / "paper_corpus.jsonl").open() if l.strip())}
    return [c for c in index["cluster_ids"]
            if c in corpus and rule_is_fha(corpus[c])]


def test_majority_vote_breaks_two_one():
    records = [_rec(1, [1, 0, 0, 1, 0], framework="mcdonnell", pass_no=1),
               _rec(1, [1, 0, 1, 0, 0], framework="mcdonnell", pass_no=2),
               _rec(1, [0, 0, 1, 1, 0], framework="hud", pass_no=3)]
    vote = llm.majority_vote(records)
    assert vote["disparate_treatment"] == 1
    assert vote["refusal_rent_sell"] == 1
    assert vote["reasonable_accommodation"] == 1
    assert vote["disparate_impact"] == 0
    assert vote["zoning_exclusionary"] == 0
    assert vote["framework"] == "mcdonnell"
    assert vote["n"] == 3


def test_majority_vote_unanimous():
    records = [_rec(2, [0, 1, 0, 0, 1], framework="hud", pass_no=p)
               for p in (1, 2, 3)]
    vote = llm.majority_vote(records)
    assert [vote[c] for c in CLAIMS] == [0, 1, 0, 0, 1]
    assert vote["framework"] == "hud"
    assert vote["n"] == 3


def test_majority_vote_two_passes():
    records = [_rec(3, [1, 1, 0, 0, 0], framework="none", pass_no=1),
               _rec(3, [1, 1, 0, 0, 0], framework="none", pass_no=3)]
    vote = llm.majority_vote(records)
    assert [vote[c] for c in CLAIMS] == [1, 1, 0, 0, 0]
    assert vote["framework"] == "none"
    assert vote["n"] == 2

    # A 1-1 split has no majority. The tie convention is not pinned by the
    # corpus (its only 2-pass case is unanimous), so require just that the
    # call yields a binary label and does not depend on pass ordering.
    split = [_rec(4, [1, 0, 0, 0, 0], framework="hud", pass_no=1),
             _rec(4, [0, 0, 0, 0, 0], framework="none", pass_no=2)]
    tie = llm.majority_vote(split)
    assert tie["disparate_treatment"] in (0, 1)
    assert tie["n"] == 2
    assert tie == llm.majority_vote(split)
    assert tie == llm.majority_vote(list(reversed(split)))


def test_self_consistency_bounds():
    agree = [_rec(1, [1, 0, 1, 0, 0], pass_no=p) for p in (1, 2, 3)]
    agree += [_rec(2, [0, 0, 0, 1, 1], pass_no=p) for p in (1, 2, 3)]
    assert llm.self_consistency(agree) == pytest.approx(1.0)

    split = [_rec(1, [1, 1, 1, 1, 1], pass_no=1),
             _rec(1, [0, 0, 0, 0, 0], pass_no=2),
             _rec(2, [0, 0, 0, 0, 0], pass_no=1),
             _rec(2, [1, 1, 1, 1, 1], pass_no=2)]
    assert llm.self_consistency(split) == pytest.approx(0.0)


def test_self_consistency_is_a_fraction():
    labels = [_rec(1, [1, 0, 0, 0, 0], pass_no=1),
              _rec(1, [0, 0, 0, 0, 0], pass_no=2),
              _rec(1, [0, 0, 0, 0, 0], pass_no=3)]
    # one of five constructs splits
    assert llm.self_consistency(labels) == pytest.approx(0.8)


def test_wilson_contains_point_estimate():
    # bounds are analytically in [0, 1]; tolerate float noise at k = 0 or k = n
    for k, n in [(1, 10), (5, 10), (9, 10), (76, 150), (0, 20), (20, 20)]:
        lo, hi = llm.wilson(k, n)
        assert -1e-9 <= lo <= hi <= 1 + 1e-9, (k, n)
        # tolerance also admits implementations that round bounds to 3 decimals
        assert lo - 1e-3 <= k / n <= hi + 1e-3, (k, n)


def test_wilson_widens_as_n_shrinks():
    wide = llm.wilson(3, 10)
    narrow = llm.wilson(30, 100)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    narrower = llm.wilson(300, 1000)
    assert (narrow[1] - narrow[0]) > (narrower[1] - narrower[0])


def test_precision_recall_correction():
    # precision == recall leaves the observed share unchanged
    assert llm.precision_recall_correction(0.376, 0.8, 0.8) == pytest.approx(0.376, abs=1e-9)
    # zero recall is undefined, not infinite
    assert llm.precision_recall_correction(0.3, 0.9, 0.0) != llm.precision_recall_correction(0.3, 0.9, 0.0)
    # correction cannot exceed a share of one
    assert llm.precision_recall_correction(0.9, 0.95, 0.1) == pytest.approx(1.0)


def test_precision_recall_correction_matches_paper_corrections():
    # observed regex shares on the 417 substantive clusters, the regex
    # precision/recall from the 93-case overlap, and the corrected shares
    cases = [
        ("disparate_treatment", 0.3765, 0.907, 0.722, 0.472),
        ("reasonable_accommodation", 0.3549, 0.829, 0.944, 0.312),
        ("disparate_impact", 0.2806, 0.407, 0.917, 0.125),
        ("refusal_rent_sell", 0.2374, 0.731, 0.613, 0.283),
        ("zoning_exclusionary", 0.0935, 0.800, 0.800, 0.094),
    ]
    for name, observed, precision, recall, corrected in cases:
        got = llm.precision_recall_correction(observed, precision, recall)
        assert 0.0 <= got <= 1.0, name
        assert got == pytest.approx(corrected, abs=1e-3), name


def test_mcnemar_exact_symmetric_cases():
    assert llm.mcnemar_exact(7, 7) == pytest.approx(1.0)
    assert llm.mcnemar_exact(0, 0) == pytest.approx(1.0)
    assert llm.mcnemar_exact(1, 1) == pytest.approx(1.0)


def test_mcnemar_exact_known_value():
    assert llm.mcnemar_exact(15, 8) == pytest.approx(0.2100, abs=1e-3)
    assert llm.mcnemar_exact(8, 15) == pytest.approx(0.2100, abs=1e-3)
    assert llm.mcnemar_exact(31, 13) == pytest.approx(0.0096, abs=1e-3)
    assert llm.mcnemar_exact(28, 3) == pytest.approx(0.0, abs=1e-3)


def test_mcnemar_exact_in_unit_interval():
    for n10 in range(0, 12):
        for n01 in range(0, 12):
            p = llm.mcnemar_exact(n10, n01)
            assert 0.0 <= p <= 1.0


def test_committed_majority_votes_rederive(raw_labels, committed_votes):
    assert raw_labels["n_passes"] == 3
    groups = {}
    for rec in raw_labels["labels"]:
        groups.setdefault((rec["set"], rec["cluster_id"]), []).append(rec)
    assert len(groups) == 243

    for set_name, key in (("gold", "goldVote"), ("rand", "randVote")):
        committed = committed_votes[key]
        for cid, expected in committed.items():
            vote = llm.majority_vote(groups[(set_name, int(cid))])
            for claim in CLAIMS:
                assert vote[claim] == expected[claim], (key, cid, claim)
            assert vote["framework"] == expected["framework"], (key, cid)
            assert vote["n"] == expected["n"], (key, cid)

    assert len(committed_votes["goldVote"]) == 93
    assert len(committed_votes["randVote"]) == 150
    assert llm.self_consistency(raw_labels["labels"]) == pytest.approx(
        committed_votes["selfConsistency"], abs=5e-5)
    assert llm.self_consistency(raw_labels["labels"]) == pytest.approx(
        0.9877, abs=5e-5)


def test_random_sample_substantive_subset(substantive_ids):
    index = _load_json("random_sample_index.json")
    assert index["seed"] == 20260720
    assert index["n"] == 150
    assert len(index["cluster_ids"]) == 150
    assert len(set(index["cluster_ids"])) == 150
    assert len(substantive_ids) == 76
    assert len(substantive_ids) / index["n"] == pytest.approx(0.507, abs=5e-4)


def test_llm_prevalence_on_substantive_subset(substantive_ids, committed_votes):
    votes = committed_votes["randVote"]
    expected = {"disparate_treatment": 0.461, "reasonable_accommodation": 0.224,
                "disparate_impact": 0.132, "refusal_rent_sell": 0.171,
                "zoning_exclusionary": 0.079}
    for claim, target in expected.items():
        share = sum(votes[str(c)][claim] for c in substantive_ids) / len(substantive_ids)
        assert share == pytest.approx(target, abs=5e-4), claim


def test_paired_prevalence_contrasts(substantive_ids, committed_votes):
    votes = committed_votes["randVote"]

    def discordant(a, b):
        n10 = sum(1 for c in substantive_ids
                  if votes[str(c)][a] and not votes[str(c)][b])
        n01 = sum(1 for c in substantive_ids
                  if not votes[str(c)][a] and votes[str(c)][b])
        return n10, n01

    cases = [("reasonable_accommodation", "disparate_impact", 15, 8, 0.2100),
             ("disparate_treatment", "reasonable_accommodation", 31, 13, 0.0096),
             ("disparate_treatment", "disparate_impact", 28, 3, 0.0000)]
    for a, b, exp10, exp01, exp_p in cases:
        n10, n01 = discordant(a, b)
        assert (n10, n01) == (exp10, exp01), (a, b)
        assert llm.mcnemar_exact(n10, n01) == pytest.approx(exp_p, abs=1e-3), (a, b)


# --- exact-power sample sizing (src/fha/prevalence.py) ---
from fha import prevalence as pv  # noqa: E402


def test_mcnemar_power_matches_reference():
    # observed RA-vs-DI effect: 15/8 discordant of 76 substantive cases
    assert pv.mcnemar_power(15, 8, 76, 76) == pytest.approx(0.236, abs=2e-3)
    assert pv.mcnemar_power(15, 8, 76, 295) == pytest.approx(0.80, abs=1e-2)


def test_mcnemar_power_is_monotone_in_n():
    ns = [76, 150, 250, 400, 600]
    powers = [pv.mcnemar_power(15, 8, 76, n) for n in ns]
    assert powers == sorted(powers)
    assert all(0.0 <= p <= 1.0 for p in powers)


def test_n_for_power_converges_and_orders():
    r80 = pv.n_for_power(15, 8, 76, 0.80, substantive_rate=76 / 150)
    r90 = pv.n_for_power(15, 8, 76, 0.90, substantive_rate=76 / 150)
    assert r80["converged"] and r90["converged"]
    assert 250 <= r80["n_substantive"] <= 350          # ~295
    assert r90["n_substantive"] > r80["n_substantive"]  # more power costs more n
    assert r80["n_draws"] > r80["n_substantive"]        # draws inflate by 1/rate


def test_n_for_power_noninformative_split_does_not_converge():
    # a 50/50 discordant split carries no directional signal; no n suffices
    out = pv.n_for_power(12, 12, 76, 0.80)
    assert out["converged"] is False
    assert out["n_substantive"] is None
