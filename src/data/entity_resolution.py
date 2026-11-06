"""Cross-source player entity resolution and identity matching engine.

Resolves player identities across historical editions, modern datasets, and
third-party market sources using deterministic composite keys and fuzzy fallbacks.
"""

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

SPECIAL_CHAR_MAP = {
    "ø": "o",
    "Ø": "o",
    "æ": "ae",
    "Æ": "ae",
    "œ": "oe",
    "Œ": "oe",
    "ß": "ss",
    "ł": "l",
    "Ł": "l",
    "đ": "d",
    "Đ": "d",
    "ð": "d",
    "Ð": "d",
    "þ": "th",
    "Þ": "th",
}


def normalize_string(val: str | None) -> str:
    """Normalize text by unescaping legacy byte tags, stripping accents, and standardizing."""
    if not val or not isinstance(val, str):
        return ""

    text = val
    # Unescape legacy hex markers like <e9> to actual characters
    text = re.sub(
        r"<([0-9a-fA-F]{2})>",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )

    # Map special latin letters not decomposed by NFKD
    for src, target in SPECIAL_CHAR_MAP.items():
        text = text.replace(src, target)

    # Strip diacritics
    nfkd_form = unicodedata.normalize("NFKD", text)
    cleaned = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    # Lowercase and strip punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned).lower()
    return " ".join(cleaned.split())


def build_composite_key(name: str, dob: str | None = None, nationality: str | None = None) -> str:
    """Build deterministic composite key from name, date of birth, and nationality."""
    norm_name = normalize_string(name)
    norm_dob = dob.strip() if dob and isinstance(dob, str) else ""
    norm_nat = normalize_string(nationality)
    return f"{norm_name}|{norm_dob}|{norm_nat}"


@dataclass
class CatalogEntry:
    """Typed entry representing an indexed player profile."""

    player_id: int
    name: str
    dob: str = ""
    nationality: str = ""
    edition: int = 0


@dataclass
class ResolutionMatch:
    """Result of an entity resolution lookup."""

    player_id: int
    matched_name: str
    confidence: float
    match_type: str
    matched_edition: int | None = None


@dataclass
class ResolutionSummary:
    """Batch summary of resolved candidate profiles."""

    total_candidates: int = 0
    resolved_count: int = 0
    unresolved_count: int = 0
    match_types: dict[str, int] = field(default_factory=dict)


class PlayerEntityResolver:
    """In-memory entity resolver for matching player records across data sources."""

    def __init__(self) -> None:
        self.composite_index: dict[str, list[CatalogEntry]] = {}
        self.name_index: dict[str, list[CatalogEntry]] = {}
        # Blocked index by nationality for high-speed bounded fuzzy matching
        self.nation_blocked_index: dict[str, list[str]] = {}
        self.total_indexed_profiles: int = 0

    def index_catalog(self, df: pd.DataFrame) -> int:
        """Index a reference dataframe of player profiles supporting multi-alias lookups."""
        required = ["player_id"]
        if not all(col in df.columns for col in required):
            raise ValueError("Catalog must contain at least 'player_id'")

        has_long = "long_name" in df.columns
        has_short = "short_name" in df.columns
        if not has_long and not has_short:
            raise ValueError("Catalog must contain 'short_name' or 'long_name'")

        has_dob = "dob" in df.columns
        has_nat = "nationality_name" in df.columns
        has_ver = "fifa_version" in df.columns

        indexed = 0
        for _, row in df.iterrows():
            pid = int(row["player_id"])
            p_short = str(row["short_name"]) if has_short and pd.notnull(row["short_name"]) else ""
            p_long = str(row["long_name"]) if has_long and pd.notnull(row["long_name"]) else ""
            primary_name = p_long if p_long else p_short
            if not primary_name:
                continue

            pdob = str(row["dob"]) if has_dob and pd.notnull(row["dob"]) else ""
            pnat = (
                str(row["nationality_name"])
                if has_nat and pd.notnull(row["nationality_name"])
                else ""
            )
            pver = int(row["fifa_version"]) if has_ver and pd.notnull(row["fifa_version"]) else 0

            entry = CatalogEntry(
                player_id=pid,
                name=primary_name,
                dob=pdob,
                nationality=pnat,
                edition=pver,
            )

            # Index multi-aliases (both short_name and long_name)
            aliases = set()
            if p_long:
                aliases.add(p_long)
            if p_short:
                aliases.add(p_short)

            for alias in aliases:
                ckey = build_composite_key(alias, pdob, pnat)
                if ckey not in self.composite_index:
                    self.composite_index[ckey] = []
                self.composite_index[ckey].append(entry)

                norm_alias = normalize_string(alias)
                if norm_alias not in self.name_index:
                    self.name_index[norm_alias] = []
                self.name_index[norm_alias].append(entry)

                # Add to nation-blocked index
                norm_nat = normalize_string(pnat)
                if norm_nat not in self.nation_blocked_index:
                    self.nation_blocked_index[norm_nat] = []
                if norm_alias not in self.nation_blocked_index[norm_nat]:
                    self.nation_blocked_index[norm_nat].append(norm_alias)

            indexed += 1

        self.total_indexed_profiles += indexed
        return indexed

    def resolve_player(
        self,
        name: str,
        dob: str | None = None,
        nationality: str | None = None,
        min_confidence: float = 0.85,
    ) -> ResolutionMatch | None:
        """Resolve a candidate player against the catalog with blocking fallbacks."""
        if not name:
            return None

        # 1. Exact Composite Key Match
        ckey = build_composite_key(name, dob, nationality)
        if ckey in self.composite_index and self.composite_index[ckey]:
            best = self.composite_index[ckey][0]
            return ResolutionMatch(
                player_id=best.player_id,
                matched_name=best.name,
                confidence=1.00,
                match_type="EXACT_COMPOSITE",
                matched_edition=best.edition,
            )

        # 2. Exact Normalized Name Match
        norm_name = normalize_string(name)
        if norm_name in self.name_index and self.name_index[norm_name]:
            candidates = self.name_index[norm_name]
            if nationality:
                norm_query_nat = normalize_string(nationality)
                for cand in candidates:
                    if normalize_string(cand.nationality) == norm_query_nat:
                        return ResolutionMatch(
                            player_id=cand.player_id,
                            matched_name=cand.name,
                            confidence=0.95,
                            match_type="NAME_AND_NATION_MATCH",
                            matched_edition=cand.edition,
                        )

            best = candidates[0]
            return ResolutionMatch(
                player_id=best.player_id,
                matched_name=best.name,
                confidence=0.90,
                match_type="EXACT_NAME",
                matched_edition=best.edition,
            )

        # 3. Blocked Fuzzy Match (by nationality if available, else first-letter partitioned)
        norm_nat = normalize_string(nationality) if nationality else ""
        candidate_pool = (
            self.nation_blocked_index.get(norm_nat, [])
            if norm_nat and norm_nat in self.nation_blocked_index
            else list(self.name_index.keys())
        )

        close_matches = difflib.get_close_matches(
            norm_name, candidate_pool, n=1, cutoff=min_confidence
        )
        if close_matches:
            matched_norm = close_matches[0]
            sim = difflib.SequenceMatcher(None, norm_name, matched_norm).ratio()
            best = self.name_index[matched_norm][0]
            return ResolutionMatch(
                player_id=best.player_id,
                matched_name=best.name,
                confidence=round(sim, 3),
                match_type="FUZZY_NAME_SIMILARITY",
                matched_edition=best.edition,
            )

        return None
