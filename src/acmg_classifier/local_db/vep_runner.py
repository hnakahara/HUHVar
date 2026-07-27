"""
Local VEP subprocess runner.

Runs Ensembl VEP as a child process with --offline --json and parses
the JSON output into VEPConsequence / AnnotationData structures.
Supports batched execution (default 500 variants per VEP invocation).
"""
from __future__ import annotations
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import structlog

from acmg_classifier.exceptions import VEPRunError
from acmg_classifier.models.annotation import ConsequenceInfo
from acmg_classifier.models.enums import ConsequenceType
from acmg_classifier.models.variant import VariantRecord

log = structlog.get_logger()

# Ensembl VEP consequence term -> internal ConsequenceType.
# Order matters: _SEVERITY_ORDER (derived below) uses dict insertion order
# to define a severity ranking, which transcript-level sorting and
# _most_severe() both rely on. Keep most-severe entries at the top.
_CONSEQUENCE_MAP: dict[str, ConsequenceType] = {
    "frameshift_variant": ConsequenceType.FRAMESHIFT,
    "stop_gained": ConsequenceType.STOP_GAINED,
    "stop_lost": ConsequenceType.STOP_LOST,
    "start_lost": ConsequenceType.START_LOST,
    "splice_acceptor_variant": ConsequenceType.SPLICE_ACCEPTOR,
    "splice_donor_variant": ConsequenceType.SPLICE_DONOR,
    "missense_variant": ConsequenceType.MISSENSE,
    "synonymous_variant": ConsequenceType.SYNONYMOUS,
    "inframe_insertion": ConsequenceType.INFRAME_INSERTION,
    "inframe_deletion": ConsequenceType.INFRAME_DELETION,
    "splice_region_variant": ConsequenceType.SPLICE_REGION,
    "intron_variant": ConsequenceType.INTRON,
    "5_prime_UTR_variant": ConsequenceType.FIVE_PRIME_UTR,
    "3_prime_UTR_variant": ConsequenceType.THREE_PRIME_UTR,
    "upstream_gene_variant": ConsequenceType.UPSTREAM,
    "downstream_gene_variant": ConsequenceType.DOWNSTREAM,
    "intergenic_variant": ConsequenceType.INTERGENIC,
    "transcript_ablation": ConsequenceType.TRANSCRIPT_ABLATION,
}

_SEVERITY_ORDER = list(_CONSEQUENCE_MAP.keys())
# Map ConsequenceType -> rank (lower = more severe). Used to pick the most impactful
# transcript when multiple MANE Select transcripts overlap the variant.
_SEVERITY_INDEX: dict[ConsequenceType, int] = {
    ct: i for i, ct in enumerate(_CONSEQUENCE_MAP.values())
}


def _most_severe(terms: list[str]) -> ConsequenceType:
    for term in _SEVERITY_ORDER:
        if term in terms:
            return _CONSEQUENCE_MAP[term]
    return ConsequenceType.OTHER


def _parse_intron_distance(hgvs_c: str | None) -> int | None:
    """Extract the +/- intron distance from an HGVS coding-DNA string.

    HGVS encodes intronic positions as "<exonic_pos>+<n>" or
    "<exonic_pos>-<n>" (e.g. c.123+5, c.123-21). BP7 needs this number to
    decide whether an intronic variant is "deep" enough to be benign by
    distance alone. Returns None when the variant is purely exonic (no
    +/- suffix) so callers can distinguish "no intronic position" from
    "intronic but near the splice site"."""
    import re
    if hgvs_c is None:
        return None
    m = re.search(r"[*]?[\d]+([+-]\d+)", hgvs_c)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return None


def _parse_transcript(tc: dict[str, Any]) -> ConsequenceInfo | None:
    terms = tc.get("consequence_terms", [])
    gene_id = tc.get("gene_id", "")
    gene_symbol = tc.get("gene_symbol", "")
    transcript_id = tc.get("transcript_id", "")
    biotype = tc.get("biotype", "")
    if not transcript_id:
        return None

    consequence = _most_severe(terms)
    is_canonical = bool(tc.get("canonical"))
    is_mane = bool(tc.get("mane_select"))

    hgvs_c = tc.get("hgvsc")
    hgvs_p = tc.get("hgvsp")
    exon = tc.get("exon")
    intron = tc.get("intron")
    amino_acids = tc.get("amino_acids")

    protein_pos = None
    pp = tc.get("protein_start")
    if pp is not None:
        try:
            protein_pos = int(pp)
        except (TypeError, ValueError):
            pass

    domains_raw = tc.get("domains", [])
    domains = []
    for d in domains_raw:
        if isinstance(d, dict):
            domains.append(str(d.get("db", "")) + ": " + str(d.get("name", "")))

    # NOTE: aa_change is computed but the ConsequenceInfo model has no
    # `amino_acid_change` field (that field lives on ClinVarRecord). With
    # Pydantic v2 default extra="ignore" this assignment is silently
    # dropped. See docs/cleanup-candidates.md.
    aa_change = None
    if amino_acids and protein_pos:
        parts = amino_acids.split("/")
        if len(parts) == 2:
            aa_change = parts[0] + str(protein_pos) + parts[1]

    # VEP --uniprot emits "swissprot" (preferred) and "trembl" fields. Both can
    # be a single string or a list; the version suffix (e.g. "P38398.4") is
    # stripped to match Brandes 2023 ESM1b filenames ("P38398_LLR.csv").
    # SwissProt is preferred because it is the manually-curated subset that
    # the Brandes archive is built against — TrEMBL is the unreviewed
    # fallback only.
    uniprot_id = None
    sp = tc.get("swissprot") or tc.get("trembl")
    if sp:
        first = sp[0] if isinstance(sp, list) and sp else sp
        if isinstance(first, str) and first:
            uniprot_id = first.split(".", 1)[0]

    return ConsequenceInfo(
        transcript_id=transcript_id,
        gene_id=gene_id,
        gene_symbol=gene_symbol,
        consequence=consequence,
        biotype=biotype,
        is_canonical=is_canonical,
        is_mane_select=is_mane,
        hgvs_c=hgvs_c,
        hgvs_p=hgvs_p,
        exon=exon,
        intron=intron,
        domains=domains,
        amino_acids=amino_acids,
        protein_position=protein_pos,
        amino_acid_change=aa_change,
        uniprot_id=uniprot_id,
        intron_distance_from_splice=_parse_intron_distance(hgvs_c),
    )


def _apply_mane_fallback(
    consequences: list[ConsequenceInfo],
    mane_map: dict[str, tuple[str, str]] | None,
) -> None:
    """Recover the MANE Select flag when VEP provides none (GRCh37 cache).

    VEP only sets ``mane_select`` in the GRCh38 cache. When no consequence is
    flagged, mark the one whose transcript base accession matches the gene's
    MANE Select RefSeq/Ensembl accession, so the existing MANE-first ordering
    and ``primary_consequence`` pick the MANE-equivalent transcript on GRCh37.
    Mutates ``consequences`` in place; no-op when a real flag already exists."""
    if not mane_map or any(c.is_mane_select for c in consequences):
        return
    for c in consequences:
        mane = mane_map.get(c.gene_symbol)
        if not mane:
            continue
        refseq_base, ensembl_base = mane
        tx_base = c.transcript_id.split(".", 1)[0]
        if tx_base and (tx_base == refseq_base or tx_base == ensembl_base):
            c.is_mane_select = True


def _parse_vep_record(
    record: dict[str, Any],
    mane_map: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, list[ConsequenceInfo]]:
    """Translate one VEP JSON line into (variant_key, [consequences]).

    The 3-step key resolution is defensive: VEP can drop or rename the
    ID column depending on input format, and we MUST recover the original
    CHROM:POS:REF:ALT key because that is the join key the pipeline uses
    to attach VEP results to the input VariantRecord."""
    # 1. VEP echoes back the VCF ID column value as `id` when present —
    #    this is the fastest path and works for variants written by us.
    raw_id = record.get("id", "")
    key = raw_id if raw_id and raw_id != "." else ""

    # 2. If the JSON `id` is missing, parse the original input line that
    #    VEP preserves in `input` (tab-separated VCF fields).
    if not key:
        input_line = record.get("input", "")
        fields = input_line.split("\t") if input_line else []
        if len(fields) >= 3 and fields[2] not in (".", ""):
            key = fields[2]

    # 3. Last resort — reconstruct from VEP's parsed coordinates. This is
    #    SNV-safe but may not exactly match indel keys because VEP can
    #    left-align differently than the input. Used only when both
    #    preferred paths fail.
    if not key:
        chrom = record.get("seq_region_name", "")
        if not chrom.startswith("chr"):
            chrom = "chr" + chrom
        pos = record.get("start", 0)
        alleles = record.get("allele_string", "/").split("/")
        ref = alleles[0] if alleles else ""
        alt = alleles[1] if len(alleles) > 1 else ""
        key = chrom + ":" + str(pos) + ":" + ref + ":" + alt

    consequences: list[ConsequenceInfo] = []
    for tc in record.get("transcript_consequences", []):
        c = _parse_transcript(tc)
        if c:
            consequences.append(c)

    # GRCh37 has no VEP mane_select flag — recover it by accession before sorting.
    _apply_mane_fallback(consequences, mane_map)

    # Per-gene most-severe consequence rank. This is the TOP sort key so the gene
    # whose transcript actually overlaps the variant wins across genes, BEFORE the
    # MANE/canonical transcript attributes. Without it, a variant sitting in one
    # gene's coding region but within VEP's default 5 kb of a *neighbouring* gene
    # (e.g. MUTYH/HPDL, PTEN/KLLN, PMS2/RSPH10B/AIMP2) could pick the neighbour's
    # MANE+canonical transcript — a mere upstream/downstream call — over the target
    # gene's coding transcript, because is_mane_select/canonical outranked severity.
    gene_best: dict[str, int] = {}
    for c in consequences:
        idx = _SEVERITY_INDEX.get(c.consequence, 999)
        if idx < gene_best.get(c.gene_symbol, 999):
            gene_best[c.gene_symbol] = idx

    # Pre-sort so AnnotationData.primary_consequence (which picks the first
    # match) sees the clinically-preferred transcript first:
    #   0. Gene with the most-severe consequence (keeps the overlapping gene)
    #   1. MANE Select before non-MANE          } transcript choice WITHIN that gene
    #   2. RefSeq (NM_) before Ensembl (ENST)    }
    #   3. Canonical before non-canonical        }
    #   4. Most-severe consequence within the same rank
    # Using `not` on the booleans inverts True→False so True (preferred)
    # sorts ahead under Python's ascending sort.
    consequences.sort(key=lambda c: (
        gene_best.get(c.gene_symbol, 999),
        not c.is_mane_select,
        not c.transcript_id.startswith("NM_"),
        not c.is_canonical,
        _SEVERITY_INDEX.get(c.consequence, 999),
    ))
    return key, consequences


class LocalVEPRunner:
    def __init__(
        self,
        vep_cmd: str,
        cache_dir: Path,
        fasta: Path,
        assembly: str,
        workers: int = 4,
        mane_map: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._vep_cmd = vep_cmd
        self._cache_dir = cache_dir
        self._fasta = fasta
        self._assembly = assembly
        self._workers = workers
        # gene -> (refseq_base, ensembl_base) for recovering the MANE flag on
        # caches that don't provide it (GRCh37). None disables the fallback.
        self._mane_map = mane_map

    def annotate_batch(
        self,
        variants: list[VariantRecord],
        batch_size: int = 500,
    ) -> dict[str, list[ConsequenceInfo]]:
        """Annotate a list of variants by chunking them into VEP subprocess calls.

        VEP startup cost (cache load, FASTA index, plugin init) is large —
        typically several seconds — so we amortise it across a batch of
        ~500 variants per invocation. Smaller batches lose throughput;
        larger batches risk hitting OS argv/stdin limits and make failure
        recovery coarser (one bad variant kills the whole chunk)."""
        results: dict[str, list[ConsequenceInfo]] = {}
        for i in range(0, len(variants), batch_size):
            chunk = variants[i : i + batch_size]
            chunk_results = self._run_vep(chunk)
            results.update(chunk_results)
        return results

    def _run_vep(self, variants: list[VariantRecord]) -> dict[str, list[ConsequenceInfo]]:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.vcf"
            output_path = Path(tmpdir) / "output.json"
            self._write_input_vcf(variants, input_path)
            self._exec_vep(input_path, output_path)
            parsed = self._parse_output(output_path)
            annotatable = [v for v in variants if self._is_annotatable(v)]
            # Compare against the DEDUPLICATED key set, not len(annotatable):
            # duplicate CHROM:POS:REF:ALT rows in the batch (repeated VCF lines,
            # multi-allelic splits collapsing to the same key, multi-sample
            # duplicates) collapse to a single VEP output record. That is
            # correct behaviour, so a genuine drop is only a *unique* input key
            # that never appears in `parsed`.
            input_keys = {v.key for v in annotatable}
            missing = [k for k in input_keys if k not in parsed]
            if missing:
                log.warning(
                    "vep_batch_undercount",
                    unique_input=len(input_keys),
                    output=len(parsed),
                    missing=len(missing),
                    missing_sample=missing[:5],
                )
            elif len(annotatable) != len(input_keys):
                # Pure duplicates — no variant was lost. Debug-only so it is
                # suppressed at the default INFO log level.
                log.debug(
                    "vep_batch_duplicate_keys",
                    rows=len(annotatable),
                    unique=len(input_keys),
                )
            return parsed

    @staticmethod
    def _is_annotatable(v: VariantRecord) -> bool:
        """Skip records without a real ALT allele (e.g. ALT='.' no-variant sites)."""
        return bool(v.alt) and v.alt != "."

    def _write_input_vcf(self, variants: list[VariantRecord], path: Path) -> None:
        with path.open("w") as fh:
            fh.write("##fileformat=VCFv4.2\n")
            fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            for v in variants:
                if not self._is_annotatable(v):
                    continue
                chrom = v.chrom[3:] if v.chrom.startswith("chr") else v.chrom
                fh.write(chrom + "\t" + str(v.pos) + "\t" + v.key + "\t" + v.ref + "\t" + v.alt + "\t.\t.\t.\n")

    def _exec_vep(self, input_path: Path, output_path: Path) -> None:
        cmd = [
            self._vep_cmd,
            "--input_file", str(input_path),
            "--output_file", str(output_path),
            "--format", "vcf", "--json",
            "--cache", "--offline",
            "--cache_version", "111",
            "--dir_cache", str(self._cache_dir),
            "--assembly", self._assembly,
            "--merged",
            "--fasta", str(self._fasta),
            "--hgvs", "--hgvsg", "--canonical", "--mane_select",
            "--numbers", "--domains", "--symbol", "--biotype",
            "--uniprot",
            "--no_progress",
            "--fork", str(self._workers),
            "--force_overwrite",
        ]
        log.debug("vep_exec", cmd=" ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise VEPRunError(
                "VEP exited " + str(result.returncode) + ":\n" + result.stderr[:2000]
            )
        # VEP often emits useful warnings on stderr (skipped variants, missing contigs, etc.)
        if result.stderr.strip():
            log.info("vep_stderr", message=result.stderr.strip()[:2000])

    def _parse_output(self, output_path: Path) -> dict[str, list[ConsequenceInfo]]:
        if not output_path.exists():
            return {}
        results: dict[str, list[ConsequenceInfo]] = {}
        with output_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key, consequences = _parse_vep_record(record, self._mane_map)
                results[key] = consequences
        return results
