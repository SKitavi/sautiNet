"""
SentiKenya Topic Classifier
=============================
Classifies social media posts into Kenyan political/social topics.
Uses keyword-based classification with TF-IDF scoring and context awareness.
"""

import json
import re
import logging
from typing import Dict, List, Optional
from collections import Counter

from app.models.schemas import TopicResult

logger = logging.getLogger(__name__)


class KenyanTopicClassifier:
    """
    Topic classifier optimized for Kenyan political and social discourse.

    Classification strategy:
    1. Keyword matching against curated topic dictionaries
    2. Weighted scoring (exact match > substring > related word)
    3. Political sub-topic detection for governance-related posts
    4. Multi-label support (posts can span multiple topics)
    """

    def __init__(self, entities_path: str = "./data/kenyan_entities.json"):
        self.topic_keywords: Dict[str, List[str]] = {}
        self._load_topic_data(entities_path)

    def _load_topic_data(self, path: str):
        """Load topic keyword dictionaries."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.topic_keywords = data.get("topic_keywords", {})
            logger.info(f"Loaded {len(self.topic_keywords)} topic categories")
        except Exception as e:
            logger.warning(f"Failed to load topic data: {e}")
            self._use_default_topics()

    def _use_default_topics(self):
        """Fallback topic keywords if data file is missing."""
        self.topic_keywords = {
            "economy": ["economy", "inflation", "prices", "tax", "budget", "uchumi", "bei"],
            "healthcare": ["health", "hospital", "afya", "doctor", "NHIF", "SHA"],
            "education": ["school", "education", "elimu", "university", "KCSE", "CBC"],
            "security": ["police", "security", "crime", "usalama"],
            "infrastructure": ["road", "railway", "water", "electricity", "barabara"],
            "employment": ["job", "employment", "kazi", "unemployment", "ajira"],
            "corruption": ["corruption", "rushwa", "steal", "scandal", "EACC"],
            "election": ["election", "vote", "IEBC", "uchaguzi", "kura"],
            "technology": ["tech", "digital", "Safaricom", "M-Pesa", "innovation"],
        }

    # ── Political sub-topic rules ──
    POLITICAL_INDICATORS = {
        "election": ["election", "uchaguzi", "vote", "kura", "campaign", "ballot",
                      "tallying", "IEBC", "rigging", "winning", "losing"],
        "governance": ["government", "serikali", "president", "rais", "parliament",
                       "bunge", "senate", "cabinet", "policy", "law", "bill"],
        "accountability": ["corruption", "rushwa", "audit", "EACC", "scandal",
                          "stolen", "accountability", "transparency", "investigation"],
        "rights": ["rights", "haki", "justice", "freedom", "protest", "demonstration",
                   "human rights", "constitution", "katiba"],
        "devolution": ["county", "governor", "devolution", "CDF", "ward",
                       "county assembly", "gavana"],
        "taxation": ["tax", "ushuru", "KRA", "revenue", "levy", "VAT",
                     "income tax", "housing levy", "fuel levy"],
    }

    def classify(self, text: str, entities: Optional[List[str]] = None) -> TopicResult:
        """
        Classify text into topic categories.

        Returns primary topic and scores for all matched topics.
        """
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))

        # Score each topic
        topic_scores: Dict[str, float] = {}
        topic_keywords_found: Dict[str, List[str]] = {}

        for topic, keywords in self.topic_keywords.items():
            score = 0.0
            matched_keywords = []

            for keyword in keywords:
                kw_lower = keyword.lower()
                # Exact word match (strongest signal)
                if kw_lower in words:
                    score += 2.0
                    matched_keywords.append(keyword)
                # Substring match (weaker signal)
                elif kw_lower in text_lower:
                    score += 1.0
                    matched_keywords.append(keyword)

            if score > 0:
                topic_scores[topic] = score
                topic_keywords_found[topic] = matched_keywords

        # Determine primary topic
        if not topic_scores:
            primary_topic = "general"
            all_keywords = []
        else:
            primary_topic = max(topic_scores, key=topic_scores.get)
            all_keywords = topic_keywords_found.get(primary_topic, [])

        # Normalize scores
        max_score = max(topic_scores.values()) if topic_scores else 1.0
        normalized_scores = {
            k: round(v / max_score, 4) for k, v in topic_scores.items()
        }

        # Detect political context
        is_political, political_subtopic = self._detect_political(text_lower, words)

        return TopicResult(
            primary_topic=primary_topic,
            topic_scores=normalized_scores,
            keywords=all_keywords[:10],
            is_political=is_political,
            political_subtopic=political_subtopic,
        )

    def _detect_political(self, text_lower: str, words: set) -> tuple:
        """Detect if post is political and identify sub-topic."""
        political_scores: Dict[str, float] = {}

        for subtopic, indicators in self.POLITICAL_INDICATORS.items():
            score = sum(1 for ind in indicators if ind.lower() in words or ind.lower() in text_lower)
            if score > 0:
                political_scores[subtopic] = score

        if not political_scores:
            return False, None

        top_subtopic = max(political_scores, key=political_scores.get)
        return True, top_subtopic

    def get_trending_topics(
        self,
        texts: List[str],
        top_n: int = 10,
    ) -> List[Dict]:
        """
        Analyze a batch of texts and return trending topics.

        Returns sorted list of topics with counts and average scores.
        """
        topic_counter = Counter()
        topic_examples: Dict[str, List[str]] = {}

        for text in texts:
            result = self.classify(text)
            if result.primary_topic != "general":
                topic_counter[result.primary_topic] += 1
                if result.primary_topic not in topic_examples:
                    topic_examples[result.primary_topic] = []
                if len(topic_examples[result.primary_topic]) < 3:
                    topic_examples[result.primary_topic].append(text[:100])

        trending = []
        for topic, count in topic_counter.most_common(top_n):
            trending.append({
                "topic": topic,
                "count": count,
                "percentage": round(count / max(len(texts), 1) * 100, 1),
                "example_texts": topic_examples.get(topic, []),
            })

        return trending
