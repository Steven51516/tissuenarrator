import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Tuple, Optional


def _parse_meta_block(s: str) -> Dict[str, str]:
    """
    Parse metadata from a comma-separated string like:
    'class: Astro, region: Cortex, spatial_domain: OLF'

    Returns:
        dict[str, str]: Parsed key-value pairs (lowercased keys).
    """
    d: Dict[str, str] = {}
    for chunk in (c.strip() for c in s.split(",") if c.strip()):
        if ":" in chunk:
            k, v = chunk.split(":", 1)
            d[k.strip().lower()] = v.strip()
        else:
            d[chunk.strip().lower()] = "true"
    return d


def _format_meta_block(d: Dict[str, str]) -> str:
    """Format metadata dict into 'key: value' comma-separated string."""
    return ", ".join(f"{k}: {v}" for k, v in d.items())



@dataclass
class CellSentence:
    """
    Represents a single cell's spatial and transcriptomic description.

    Attributes:
        x (float): X-coordinate (supports negative values).
        y (float): Y-coordinate (supports negative values).
        meta (dict): Metadata fields (e.g., class, region).
        cs (list[str]): List of gene symbols or cell tokens.
    """
    x: float
    y: float
    meta: Dict[str, str] = field(default_factory=dict)
    cs: List[str] = field(default_factory=list)

    @classmethod
    def from_string(cls, s: str) -> "CellSentence":
        """
        Parse a single cell string block such as:
        '<pos> X: -12.3, Y: 45.6 <meta> class: OEC, region: OLF <cs> GFAP AQP4 </cs>'
        """
        # Allow signed floats for X and Y
        m_pos = re.search(
            r"<pos>\s*X:\s*([+-]?\d+(?:\.\d+)?)\s*,\s*Y:\s*([+-]?\d+(?:\.\d+)?)",
            s, flags=re.IGNORECASE
        )
        if not m_pos:
            raise ValueError("Cannot find <pos> X: ..., Y: ... in cell string.")
        x, y = float(m_pos.group(1)), float(m_pos.group(2))

        # Optional metadata section
        m_meta = re.search(r"<meta>\s*(.*?)(?=\s*<cs>|$)", s, flags=re.IGNORECASE | re.DOTALL)
        meta_dict = _parse_meta_block(m_meta.group(1).strip()) if m_meta else {}

        # Required gene/cell content section
        m_cs = re.search(r"<cs>\s*(.*?)(?:</cs>|$)", s, flags=re.IGNORECASE | re.DOTALL)
        if not m_cs:
            raise ValueError("Cannot find <cs> ... in cell string.")
        cs_tokens = m_cs.group(1).split()
        cs_list = [tok.strip() for tok in cs_tokens if tok.strip()]

        return cls(x=x, y=y, meta=meta_dict, cs=cs_list)

    def to_string(self, include: Iterable[str] = ("pos", "meta", "cs")) -> str:
        """Serialize the cell back into the standardized text format."""
        inc = {s.lower() for s in include}
        parts: List[str] = []

        if "pos" in inc:
            parts.append(f"<pos> X: {self.x:g}, Y: {self.y:g}")
        if "meta" in inc and self.meta:
            parts.append(f"<meta> {_format_meta_block(self.meta)}")
        if "cs" in inc and self.cs:
            parts.append(f"<cs> {' '.join(self.cs)} </cs>")

        return " ".join(parts)

    def xy(self) -> Tuple[float, float]:
        """Return (x, y) coordinates as floats for spatial calculations."""
        return float(self.x), float(self.y)

    def class_lower(self) -> Optional[str]:
        """Return lowercase cell class, if defined."""
        v = self.meta.get("class")
        return v.strip().lower() if isinstance(v, str) and v.strip() else None

    def __repr__(self) -> str:
        cls_name = self.class_lower() or "unknown"
        return f"<CellSentence x={self.x:.2f} y={self.y:.2f} class={cls_name}>"


@dataclass
class SpatialSentence:
    """
    Represents a full spatial sentence containing multiple cell sentences.
    """
    meta: Dict[str, str] = field(default_factory=dict)
    cells: List[CellSentence] = field(default_factory=list)

    @classmethod
    def from_string(cls, text: str, *, strict: bool = False) -> "SpatialSentence":
        """
        Parse a spatial sentence with optional metadata and multiple <pos> blocks.

        Args:
            text: Full text containing optional meta and multiple cell definitions.
            strict: If True, raises an error on malformed blocks; else skips them.
        """
        text = text.strip()
        first_pos = re.search(r"(?i)<pos>", text)

        # Extract head meta (before first <pos>)
        sent_meta: Dict[str, str] = {}
        rest = text
        if first_pos:
            prefix = text[:first_pos.start()].strip().strip(",")
            rest = text[first_pos.start():]
            if prefix:
                sent_meta = _parse_meta_block(prefix)

        # Split by <pos> tags
        parts = re.split(r"(?i)(?=<pos>)", rest)
        cell_blocks = [p.strip() for p in parts if p.strip().lower().startswith("<pos>")]

        cells: List[CellSentence] = []
        for block in cell_blocks:
            try:
                cells.append(CellSentence.from_string(block))
            except ValueError:
                if strict:
                    raise
                continue

        return cls(meta=sent_meta, cells=cells)

    def to_string(
        self,
        include_sentence_meta: bool = True,
        include_cell: Iterable[str] = ("pos", "meta", "cs"),
    ) -> str:
        """Serialize the full spatial sentence."""
        head = _format_meta_block(self.meta).strip() if (include_sentence_meta and self.meta) else ""
        body = " ".join(cell.to_string(include=include_cell) for cell in self.cells)
        return f"{head} {body}".strip()

    def __repr__(self) -> str:
        return f"<SpatialSentence n_cells={len(self.cells)} meta={list(self.meta.keys())}>"


# ==============================
# Quick Check
# ==============================
if __name__ == "__main__":
    raw_text = """
    meta_key1: foo, metakey2: bar
    <pos> X: 10, Y: -20 <meta> class: Vascular, spatial_domain: OLF <cs> AQP1 GJA1 EGFR </cs>
    <pos> X: -30.5, Y: 40.2 <meta> class: OEC, spatial_domain: OLF <cs> GFAP AQP4 IGF2 </cs>
    """

    print("\n=== Parsing SpatialSentence ===")
    sp = SpatialSentence.from_string(raw_text)
    print(sp)
    for c in sp.cells:
        print("  ", c, "xy:", c.xy())

    print("\n=== Serialized Back ===")
    out_str = sp.to_string()
    print(out_str)

    print("\n=== Round-trip Consistency Check ===")
    sp2 = SpatialSentence.from_string(out_str)
    print("Cells preserved:", len(sp.cells) == len(sp2.cells))
    print("Meta preserved:", sp.meta == sp2.meta)
