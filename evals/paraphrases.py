"""Hand-written follow-up queries that do not share wording with the synthetic templates.

The templated queries in ``synthetic.py`` are built from the same person/topic strings as the
agent names, so lexical retrieval is almost guaranteed to find them. These paraphrases use
abbreviations, synonyms, punctuation and typos the way people actually type, to measure how
brittle the roster retrieval is. Each entry resolves to a ground-truth agent by rule:
person + topic -> the most recently used agent for that pair; person only -> that person's
most recent agent; topic only -> that topic's most recent agent.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .synthetic import Workload

PARAPHRASES: List[Dict[str, str]] = [
    {"text": "any word from Priya?", "person": "Priya"},
    {"text": "has Alice written back", "person": "Alice"},
    {"text": "chase Bob about the dinner thing once more", "person": "Bob", "topic": "dinner"},
    {"text": "the dentist stuff for alice — done?", "person": "Alice", "topic": "dentist"},
    {"text": "ping tara re the invoice", "person": "Tara", "topic": "invoice"},
    {"text": "pls follow up w/ Omar", "person": "Omar"},
    {"text": "how's the visa situation", "topic": "visa"},
    {"text": "did we hear from Sofia", "person": "Sofia"},
    {"text": "lunch w Elena, status?", "person": "Elena", "topic": "lunch"},
    {"text": "bump the contract w priya", "person": "Priya", "topic": "contract"},
    {"text": "that gift for Sofia, any progress", "person": "Sofia", "topic": "birthday gift"},
    {"text": "rosa's flight — confirmed?", "person": "Rosa", "topic": "flight"},
    {"text": "hotel: anything back from priya", "person": "Priya", "topic": "hotel"},
    {"text": "noor rent, all good now?", "person": "Noor", "topic": "rent"},
    {"text": "liam's flight, where did we land", "person": "Liam", "topic": "flight"},
    {"text": "greta warranty??", "person": "Greta", "topic": "warranty"},
    {"text": "re amara's flight, news", "person": "Amara", "topic": "flight"},
    {"text": "jonah dinner reminder — still on?", "person": "Jonah", "topic": "dinner"},
    {"text": "keith wedding rsvp, did he answer", "person": "Keith", "topic": "wedding"},
    {"text": "hugo apt viewing, booked?", "person": "Hugo", "topic": "apartment"},
    {"text": "yuki car thing", "person": "Yuki", "topic": "car service"},
    {"text": "fatima payroll — resolved", "person": "Fatima", "topic": "payroll"},
    {"text": "marcus gym cancel, done?", "person": "Marcus", "topic": "gym"},
    {"text": "zoe lunch, did she say yes", "person": "Zoe", "topic": "lunch"},
    {"text": "bob's laptop warranty, news?", "person": "Bob", "topic": "warranty"},
    {"text": "tomasz tuition $, any movement", "person": "Tomasz", "topic": "tuition"},
    {"text": "sharanjeet wedding — reply?", "person": "Sharanjeet", "topic": "wedding"},
    {"text": "tara talk proposal, submitted?", "person": "Tara", "topic": "conference talk"},
    {"text": "priya car service, when", "person": "Priya", "topic": "car service"},
    {"text": "alice tuition $, back yet?", "person": "Alice", "topic": "tuition"},
    {"text": "Priyaa invoice?", "person": "Priya", "topic": "invoice"},  # typo
    {"text": "anythng from omar", "person": "Omar"},  # typo
]


def resolve(workload: Workload, item: Dict[str, str]) -> Optional[str]:
    """Ground-truth agent name for a paraphrase, or None if the workload has no such agent."""
    person, topic = item.get("person"), item.get("topic")
    candidates = [
        a for a in workload.agents
        if (person is None or a.person == person) and (topic is None or a.topic == topic)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.last_used_at).name


__all__ = ["PARAPHRASES", "resolve"]
