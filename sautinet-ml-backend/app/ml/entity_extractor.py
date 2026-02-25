"""
SentiKenya Entity Extractor
=============================
Named Entity Recognition optimized for Kenyan context.
Detects counties, political figures, parties, government bodies, and organizations.
"""

import re
import json
import logging
from typing import List, Dict, Set, Optional

from app.models.schemas import Entity, NERResult

logger = logging.getLogger(__name__)


class KenyanEntityExtractor:
    """
    Rule-based NER for Kenyan entities.

    Categories:
    - COUNTY: Kenyan counties (47 + aliases)
    - POLITICAL_PARTY: Registered parties + common abbreviations
    - GOV_BODY: Government institutions (Parliament, KRA, IEBC, etc.)
    - ORGANIZATION: Companies, media, universities
    - PERSON: (Basic pattern matching — production would use fine-tuned model)

    Why rule-based instead of ML?
    - No labeled Kenyan NER dataset exists at scale
    - Kenyan entity names are highly specific (e.g., "Maandamano" = protests)
    - Rule-based gives 100% precision on known entities
    - Can be combined with transformer NER for unknown entities
    """

    def __init__(self, entities_path: str = "./data/kenyan_entities.json"):
        self.counties: Set[str] = set()
        self.county_aliases: Dict[str, str] = {}
        self.political_parties: Dict[str, List[str]] = {}
        self.government_bodies: Set[str] = set()
        self.media_outlets: Set[str] = set()
        self.institutions: Set[str] = set()
        self._load_entities(entities_path)

    def _load_entities(self, path: str):
        """Load entity dictionaries from JSON."""
        try:
            with open(path, "r") as f:
                data = json.load(f)

            self.counties = set(data.get("counties", []))
            self.county_aliases = data.get("county_aliases", {})
            self.government_bodies = set(data.get("government_bodies", []))
            self.media_outlets = set(data.get("media_outlets", []))
            self.institutions = set(data.get("institutions", []))

            # Build party lookup
            for party in data.get("political_parties", []):
                name = party["name"]
                aliases = party.get("aliases", [])
                self.political_parties[name] = aliases

            logger.info(
                f"Loaded entities: {len(self.counties)} counties, "
                f"{len(self.political_parties)} parties, "
                f"{len(self.government_bodies)} gov bodies"
            )
        except Exception as e:
            logger.warning(f"Failed to load entities: {e}")

    def extract(self, text: str) -> NERResult:
        """
        Extract all named entities from text.

        Returns NERResult with categorized entities.
        """
        entities: List[Entity] = []
        political_figures: List[str] = []
        organizations: List[str] = []
        counties_mentioned: List[str] = []

        text_lower = text.lower()

        # 1. Extract counties
        for county in self.counties:
            pattern = re.compile(r'\b' + re.escape(county) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(),
                    label="COUNTY",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    normalized=county,
                ))
                if county not in counties_mentioned:
                    counties_mentioned.append(county)

        # Check county aliases
        for alias, county in self.county_aliases.items():
            pattern = re.compile(r'\b' + re.escape(alias) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(),
                    label="COUNTY",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                    normalized=county,
                ))
                if county not in counties_mentioned:
                    counties_mentioned.append(county)

        # 2. Extract political parties
        for party_name, aliases in self.political_parties.items():
            all_names = [party_name] + aliases
            for name in all_names:
                pattern = re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append(Entity(
                        text=match.group(),
                        label="POLITICAL_PARTY",
                        start=match.start(),
                        end=match.end(),
                        confidence=0.92,
                        normalized=party_name,
                    ))
                    if party_name not in organizations:
                        organizations.append(party_name)

        # 3. Extract government bodies
        for body in self.government_bodies:
            pattern = re.compile(r'\b' + re.escape(body) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(),
                    label="GOV_BODY",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.93,
                    normalized=body,
                ))
                if body not in organizations:
                    organizations.append(body)

        # 4. Extract media and institutions
        for outlet in self.media_outlets:
            pattern = re.compile(r'\b' + re.escape(outlet) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(),
                    label="MEDIA",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.90,
                    normalized=outlet,
                ))

        for inst in self.institutions:
            pattern = re.compile(r'\b' + re.escape(inst) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(),
                    label="ORGANIZATION",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.88,
                    normalized=inst,
                ))
                if inst not in organizations:
                    organizations.append(inst)

        # 5. Basic person detection (title + capitalized words)
        person_patterns = [
            r'\b(?:President|Governor|Senator|MP|Hon\.?|Dr\.?|Prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b(?:Rais|Gavana|Seneta|Mbunge)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        ]
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                full_match = match.group(0)
                name = match.group(1)
                entities.append(Entity(
                    text=full_match,
                    label="PERSON",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.75,
                    normalized=name,
                ))
                if name not in political_figures:
                    political_figures.append(name)

        # 6. Monetary/numeric entities (budget figures, prices)
        money_patterns = [
            r'KES\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M))?',
            r'Ksh\.?\s*[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|B|M))?',
            r'\d+(?:,\d+)*(?:\.\d+)?\s*(?:billion|million|trillion)\s*(?:shillings?)?',
        ]
        for pattern in money_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(Entity(
                    text=match.group(),
                    label="MONETARY",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.90,
                ))

        # Deduplicate by position
        seen_positions = set()
        unique_entities = []
        for entity in entities:
            key = (entity.start, entity.end, entity.label)
            if key not in seen_positions:
                seen_positions.add(key)
                unique_entities.append(entity)

        return NERResult(
            entities=unique_entities,
            political_figures=political_figures,
            organizations=organizations,
            counties_mentioned=counties_mentioned,
        )
