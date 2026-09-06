import pytest

from cbrmb import PHYLUM_COLORS, phylum_colors, resolve_phylum
from cbrmb.palettes import OTHER_COLOR

# phylum-level row labels exactly as they come out of the BiotaWiz / Rhea
# tax.summary.all.tab (obsm["tax_binning"])
BIOTAWIZ_PHYLA = [
    "p__Actinomycetota", "p__Bacillota", "p__Bacillota--other", "p__Bacteroidota",
    "p__Bdellovibrionota", "p__Candidatus Zhuqueibacterota", "p__Chloroflexota--other",
    "p__Clostridiota", "p__Cyanobacteriota", "p__Deinococcota", "p__Desulfotomaculota",
    "p__Desulfovibrionota", "p__Elusimicrobiota", "p__Fusobacteriota", "p__Gemmatimonadota",
    "p__Lentisphaerota", "p__Methanobacteriota", "p__Minisyncoccota", "p__Mycoplasmatota",
    "p__Planctomycetota", "p__Pseudomonadota", "p__Spirochaetota", "p__Synergistota",
    "p__Thermoplasmatota", "p__unknown_ Bacteria", "p__Verrucomicrobiota",
]


def test_every_biotawiz_phylum_gets_a_hex_colour():
    seq = phylum_colors(BIOTAWIZ_PHYLA)
    assert isinstance(seq, list) and len(seq) == len(BIOTAWIZ_PHYLA)
    assert all(isinstance(v, str) and v.startswith("#") for v in seq)


@pytest.mark.parametrize(
    "label,canonical",
    [
        ("p__Bacteroidota", "Bacteroidota"),
        ("Bacteroidota", "Bacteroidota"),
        ("p__Bacillota--other", "Bacillota"),
        ("p__Candidatus Zhuqueibacterota", "Zhuqueibacterota"),
        ("Firmicutes", "Bacillota"),
        ("Proteobacteria", "Pseudomonadota"),
        ("Actinobacteriota", "Actinomycetota"),
        ("Bacteroidota_A", "Bacteroidota"),
        ("p__unknown_ Bacteria", "Other"),
        ("Unassigned", "Other"),
        ("", "Other"),
    ],
)
def test_resolve_phylum(label, canonical):
    assert resolve_phylum(label) == canonical


def test_known_phylum_matches_book_and_synonym_agrees():
    assert phylum_colors(["Bacteroidota"]) == [PHYLUM_COLORS["Bacteroidota"]]
    # legacy spelling resolves to the same colour as the canonical name
    assert phylum_colors(["Firmicutes"]) == [PHYLUM_COLORS["Bacillota"]]


def test_other_and_fallback_behaviour():
    assert phylum_colors(["p__unknown_ Bacteria"]) == [OTHER_COLOR]
    # unknown phylum -> deterministic fallback, stable across calls
    (a,) = phylum_colors(["p__Weirdota"])
    (b,) = phylum_colors(["p__Weirdota"])
    assert a == b and a.startswith("#") and a != OTHER_COLOR
    # ... unless fallback is turned off
    assert phylum_colors(["p__Weirdota"], fallback=False) == [OTHER_COLOR]


def test_full_book_is_a_dict_and_extra_override_applies():
    book = phylum_colors()
    assert isinstance(book, dict)
    assert "Other" in book and book["Bacteroidota"] == PHYLUM_COLORS["Bacteroidota"]
    # extra key resolved via synonyms -> recolours Bacillota in both call styles
    assert phylum_colors(["Firmicutes"], extra={"Firmicutes": "#000000"}) == ["#000000"]
    assert phylum_colors(extra={"Bacteroidetes": "#111111"})["Bacteroidota"] == "#111111"
