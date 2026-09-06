"""A stable colour book for bacterial / archaeal phyla.

The BiotaWiz pipeline (via Rhea's ``tax.summary.all.tab``) reports phyla under
the LPSN / SeqCode nomenclature -- ``Bacillota`` rather than ``Firmicutes``,
``Pseudomonadota`` rather than ``Proteobacteria``, and so on -- with row labels
like ``"p__Bacteroidota"``, ``"p__Bacillota--other"`` or
``"p__Candidatus Zhuqueibacterota"``.

:func:`phylum_colors` maps those onto a fixed palette so the same phylum keeps
its colour across panels and studies. Names are normalised first
(:func:`resolve_phylum`): the ``p__`` prefix, a trailing ``--other``, a
``Candidatus`` prefix and GTDB ``_A``/``_B`` suffixes are stripped, legacy names
are translated, and anything unclassified collapses to ``"Other"``. Phyla the
book does not know get a deterministic fallback colour (hashed into
:data:`FALLBACK_PALETTE`), so every possible label still resolves to something.

Pure Python; no matplotlib import here. The matching legend helper lives in
:func:`cbrmb.plotting.phylum_handles`.
"""

from __future__ import annotations

import hashlib
import re

__all__ = ["PHYLUM_COLORS", "FALLBACK_PALETTE", "OTHER_COLOR", "resolve_phylum", "phylum_colors"]

OTHER_COLOR = "#D9D5D1"

#: Canonical phylum name (LPSN ``-ota``) -> hex colour.
PHYLUM_COLORS: dict[str, str] = {
    # -- dominant gut / oral phyla: kept maximally distinct, they stack together
    "Bacteroidota": "#47855F",
    "Bacillota": "#3B6FA0",
    "Clostridiota": "#C65B7C",
    "Pseudomonadota": "#A98B45",
    "Actinomycetota": "#8E6CAF",
    "Fusobacteriota": "#6E7DA0",
    "Verrucomicrobiota": "#E08A3C",
    "Desulfotomaculota": "#9C6B3F",
    "Mycoplasmatota": "#D3A0C8",
    "Cyanobacteriota": "#4BA3A0",
    "Spirochaetota": "#C0453F",
    "Desulfovibrionota": "#6D5643",
    # -- less common, still seen in 16S surveys
    "Synergistota": "#9A9A3C",
    "Deinococcota": "#E093A8",
    "Minisyncoccota": "#7E6BA8",
    "Elusimicrobiota": "#8FAF63",
    "Lentisphaerota": "#5EA8D4",
    "Planctomycetota": "#3E9B87",
    "Gemmatimonadota": "#8C8CC4",
    "Bdellovibrionota": "#B76E9A",
    "Chloroflexota": "#B98A54",
    "Campylobacterota": "#C77CA0",
    "Patescibacteria": "#9AA0A6",
    "Acidobacteriota": "#6B8E4E",
    "Myxococcota": "#9C6E3E",
    "Nitrospirota": "#5C8A72",
    "Nitrospinota": "#4F7E86",
    "Thermotogota": "#BE6A5A",
    "Aquificota": "#5B92CF",
    "Chlamydiota": "#D98CB3",
    "Deferribacterota": "#96754F",
    "Fibrobacterota": "#74A05B",
    "Armatimonadota": "#6FA8A0",
    "Cloacimonadota": "#A97C52",
    "Fermentibacterota": "#8484B6",
    "Coprothermobacterota": "#B5651D",
    "Thermodesulfobacteriota": "#6E5B7B",
    "Caldisericota": "#977A5B",
    "Dictyoglomota": "#C08552",
    "Rhodothermota": "#C1666B",
    "Ignavibacteriota": "#5F8575",
    "Kiritimatiellota": "#4F8FA6",
    "Calditrichota": "#A88C6B",
    "Balneolota": "#7EA6B2",
    "Abditibacteriota": "#8A9BA0",
    "Sumerlaeota": "#A7935E",
    "Omnitrophota": "#7B9C8C",
    "Dependentiae": "#B58FB0",
    # -- Archaea
    "Methanobacteriota": "#8A9A5B",
    "Thermoplasmatota": "#B58A57",
    "Halobacteriota": "#C97A57",
    "Thermoproteota": "#9E7BB5",
    "Nitrososphaerota": "#7FA98F",
    "Nanoarchaeota": "#A9A9A9",
    "Euryarchaeota": "#8C7B5C",
}

#: Deterministic pool for phyla missing from :data:`PHYLUM_COLORS`.
FALLBACK_PALETTE: list[str] = [
    "#6C8EBF", "#B85450", "#82B366", "#9673A6", "#D6B656", "#5B9BD5",
    "#C7754F", "#7EA97E", "#A57FB2", "#C0A16B", "#6FA8A8", "#B57EA5",
    "#8C9B5A", "#C98A6A", "#7B7BA8", "#9BA86F",
]

_OTHER_KEY = "Other"

# legacy / alternate spellings -> canonical (lookup is case-insensitive)
_SYNONYMS: dict[str, str] = {
    "firmicutes": "Bacillota",
    "proteobacteria": "Pseudomonadota",
    "alphaproteobacteria": "Pseudomonadota",
    "betaproteobacteria": "Pseudomonadota",
    "gammaproteobacteria": "Pseudomonadota",
    "deltaproteobacteria": "Pseudomonadota",
    "epsilonproteobacteria": "Campylobacterota",
    "epsilonbacteraeota": "Campylobacterota",
    "actinobacteria": "Actinomycetota",
    "actinobacteriota": "Actinomycetota",
    "bacteroidetes": "Bacteroidota",
    "bacteroidetes/chlorobi group": "Bacteroidota",
    "fusobacteria": "Fusobacteriota",
    "verrucomicrobia": "Verrucomicrobiota",
    "chlamydiae/verrucomicrobia group": "Verrucomicrobiota",
    "tenericutes": "Mycoplasmatota",
    "mollicutes": "Mycoplasmatota",
    "cyanobacteria": "Cyanobacteriota",
    "cyanobacteria/melainabacteria group": "Cyanobacteriota",
    "melainabacteria": "Cyanobacteriota",
    "spirochaetes": "Spirochaetota",
    "spirochaetae": "Spirochaetota",
    "deinococcus-thermus": "Deinococcota",
    "planctomycetes": "Planctomycetota",
    "chloroflexi": "Chloroflexota",
    "synergistetes": "Synergistota",
    "lentisphaerae": "Lentisphaerota",
    "elusimicrobia": "Elusimicrobiota",
    "gemmatimonadetes": "Gemmatimonadota",
    "acidobacteria": "Acidobacteriota",
    "nitrospirae": "Nitrospirota",
    "chlamydiae": "Chlamydiota",
    "thermotogae": "Thermotogota",
    "aquificae": "Aquificota",
    "fibrobacteres": "Fibrobacterota",
    "deferribacteres": "Deferribacterota",
    "armatimonadetes": "Armatimonadota",
    "saccharibacteria": "Patescibacteria",
    "candidatus saccharibacteria": "Patescibacteria",
    "candidate division tm7": "Patescibacteria",
    "tm7": "Patescibacteria",
    "thaumarchaeota": "Nitrososphaerota",
    "crenarchaeota": "Thermoproteota",
}

_UNCLASSIFIED_RE = re.compile(r"^(unknown|unclassified|unassigned|uncultured|na_|no_)", re.I)
_GTDB_SUFFIX_RE = re.compile(r"_[A-Z]{1,2}$")


def resolve_phylum(name: str) -> str:
    """Normalise a phylum label to its canonical name (or ``"Other"``).

    Strips a ``p__`` prefix, a trailing ``--other``, a ``Candidatus`` prefix and
    a GTDB ``_A``/``_B`` suffix; folds legacy names (``Firmicutes`` ->
    ``Bacillota`` etc.); maps blanks and any ``unknown``/``unclassified``/
    ``uncultured`` token to ``"Other"``.
    """

    s = str(name).strip()
    if s[:3].lower() == "p__":
        s = s[3:].strip()
    if s.lower().endswith("--other"):
        s = s[:-7].strip()
    if s.lower().startswith("candidatus "):
        s = s[11:].strip()

    low = s.lower()
    if not s or low in {"na", "nan", "none", "bacteria", "archaea", "root", "other"} \
            or _UNCLASSIFIED_RE.match(low):
        return _OTHER_KEY

    if s in PHYLUM_COLORS:
        return s
    s = _GTDB_SUFFIX_RE.sub("", s)
    return _SYNONYMS.get(s.lower(), s)


def _fallback_color(canonical: str) -> str:
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()
    return FALLBACK_PALETTE[int(digest, 16) % len(FALLBACK_PALETTE)]


def phylum_colors(names=None, *, other: str = OTHER_COLOR, extra: dict = None,
                  fallback: bool = True):
    """Phylum colours from the book.

    With ``names`` -> a **list** of hex colours, one per label and in that order,
    ready for ``cbrviz.CompBars(colors=...)``. Without ``names`` -> the full
    **dict** ``{canonical name: hex}`` (plus ``"Other"``) for lookups and
    legends.

    Parameters
    ----------
    names : iterable of str, optional
        Phylum labels (raw ``"p__..."`` is fine); normalised via
        :func:`resolve_phylum`.
    other : str
        Colour for ``"Other"`` / unclassified.
    extra : dict, optional
        Overrides merged on top; keys are resolved through
        :func:`resolve_phylum`, so ``{"Firmicutes": "#333"}`` recolours
        ``Bacillota``.
    fallback : bool
        If False, unknown phyla get ``other`` instead of a hashed colour.
    """

    book = {**PHYLUM_COLORS, _OTHER_KEY: other}
    for k, v in (extra or {}).items():
        book[resolve_phylum(k)] = v

    if names is None:
        return book

    out = []
    for name in names:
        canonical = resolve_phylum(name)
        if canonical in book:
            out.append(book[canonical])
        else:
            out.append(_fallback_color(canonical) if fallback else other)
    return out
