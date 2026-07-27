"""MANE Select fallback for caches without the flag (GRCh37).

VEP only emits ``mane_select`` in the GRCh38 cache. On GRCh37 every transcript
comes back unflagged, so the MANE-first ordering in ``_parse_vep_record`` would
otherwise pick a non-MANE isoform (e.g. PTEN NM_001304717 vs MANE NM_000314),
shifting HGVS numbering and breaking VCEP codon-range criteria. The fallback
recovers the flag by matching the gene's MANE base accession.
"""
from acmg_classifier.local_db.vep_runner import _apply_mane_fallback, _parse_vep_record
from acmg_classifier.models.annotation import ConsequenceInfo
from acmg_classifier.models.enums import ConsequenceType

# PTEN: MANE Select NM_000314 / ENST00000371953; NM_001304717 is the longer isoform.
_PTEN_MANE = {"PTEN": ("NM_000314", "ENST00000371953")}


def _cons(tx: str, mane: bool = False, canonical: bool = False) -> ConsequenceInfo:
    return ConsequenceInfo(
        transcript_id=tx, gene_id="g", gene_symbol="PTEN",
        consequence=ConsequenceType.STOP_GAINED, biotype="protein_coding",
        is_mane_select=mane, is_canonical=canonical,
    )


def test_fallback_flags_mane_transcript_by_refseq_base():
    cons = [_cons("NM_001304717.5", canonical=True), _cons("NM_000314.8")]
    _apply_mane_fallback(cons, _PTEN_MANE)
    flagged = [c.transcript_id for c in cons if c.is_mane_select]
    assert flagged == ["NM_000314.8"]


def test_fallback_matches_ensembl_base():
    cons = [_cons("ENST00000371953.8"), _cons("NM_001304717.5", canonical=True)]
    _apply_mane_fallback(cons, _PTEN_MANE)
    assert [c.transcript_id for c in cons if c.is_mane_select] == ["ENST00000371953.8"]


def test_fallback_noop_when_real_flag_present():
    # A genuine GRCh38 flag must never be overridden by the accession map.
    cons = [_cons("NM_001304717.5", mane=True), _cons("NM_000314.8")]
    _apply_mane_fallback(cons, _PTEN_MANE)
    assert [c.transcript_id for c in cons if c.is_mane_select] == ["NM_001304717.5"]


def test_fallback_noop_without_map():
    cons = [_cons("NM_001304717.5"), _cons("NM_000314.8")]
    _apply_mane_fallback(cons, None)
    assert not any(c.is_mane_select for c in cons)


def test_parse_record_selects_mane_first_on_grch37():
    """End-to-end: a GRCh37-style record (no mane_select) sorts MANE to front."""
    record = {
        "id": "chr10:89717708:C:T",
        "transcript_consequences": [
            {"transcript_id": "NM_001304717.5", "gene_symbol": "PTEN",
             "consequence_terms": ["stop_gained"], "canonical": 1, "biotype": "protein_coding"},
            {"transcript_id": "NM_000314.8", "gene_symbol": "PTEN",
             "consequence_terms": ["stop_gained"], "biotype": "protein_coding"},
        ],
    }
    _key, cons = _parse_vep_record(record, _PTEN_MANE)
    assert cons[0].transcript_id == "NM_000314.8"
    assert cons[0].is_mane_select


# --- neighbour-gene mis-selection (gene-best-severity top sort key) ------------
# Boundary variants sit in the target gene's body but within VEP's default 5 kb of
# a neighbour, so VEP returns both genes. Before the fix, the neighbour's
# MANE+canonical transcript (a mere up/downstream call) outranked the target
# gene's coding transcript. These pairs are all real GRCh37 adjacencies.
_NEIGHBOUR_MANE = {
    "PMS2": ("NM_000535", "ENST00000265849"),
    "RSPH10B": ("NM_173565", "ENST00000389039"),
    "AIMP2": ("NM_006303", "ENST00000221265"),
    "MUTYH": ("NM_001048174", "ENST00000456914"),
    "HPDL": ("NM_032756", "ENST00000334815"),
}


def _tc(tx: str, gene: str, term: str, canonical: bool) -> dict:
    d = {"transcript_id": tx, "gene_symbol": gene,
         "consequence_terms": [term], "biotype": "protein_coding"}
    if canonical:
        d["canonical"] = 1
    return d


def test_coding_gene_beats_neighbour_downstream():
    """PMS2 coding transcript must win over neighbouring RSPH10B downstream call,
    even though RSPH10B's transcript is canonical and MANE-flagged."""
    record = {
        "id": "chr7:6013062:T:TG",
        "transcript_consequences": [
            # Neighbour: canonical MANE transcript, but only a downstream call.
            _tc("NM_173565.5", "RSPH10B", "downstream_gene_variant", canonical=True),
            # Target: PMS2 MANE coding transcript, not flagged canonical by VEP.
            _tc("NM_000535.7", "PMS2", "missense_variant", canonical=False),
        ],
    }
    _key, cons = _parse_vep_record(record, _NEIGHBOUR_MANE)
    assert cons[0].gene_symbol == "PMS2"
    assert cons[0].transcript_id == "NM_000535.7"


def test_coding_gene_beats_neighbour_upstream():
    """PMS2 coding must win over an AIMP2 upstream call at the 5' boundary."""
    record = {
        "id": "chr7:6045570:A:C",
        "transcript_consequences": [
            _tc("NM_006303.4", "AIMP2", "upstream_gene_variant", canonical=True),
            _tc("NM_000535.7", "PMS2", "missense_variant", canonical=False),
        ],
    }
    _key, cons = _parse_vep_record(record, _NEIGHBOUR_MANE)
    assert cons[0].gene_symbol == "PMS2"


def test_mutyh_beats_hpdl_neighbour():
    record = {
        "id": "chr1:45795019:T:TG",
        "transcript_consequences": [
            _tc("NM_032756.4", "HPDL", "downstream_gene_variant", canonical=True),
            _tc("NM_001048174.2", "MUTYH", "missense_variant", canonical=False),
        ],
    }
    _key, cons = _parse_vep_record(record, _NEIGHBOUR_MANE)
    assert cons[0].gene_symbol == "MUTYH"
