"""Synthetic "weeks of use" workload with ground truth.

Generates a population of execution agents the way OpenPoke's Interaction Agent would name
them, spread over several weeks, with realistic collisions (the same person across several
topics; the same topic across several people). Then generates follow-up user messages whose
correct target agent is known, in four flavours:

- ``keyword``   - names the person and topic ("what did Priya say about the invoice")
- ``name_only`` - names only the person; several agents share that person, the most
                  recently used one is correct ("did priya reply?")
- ``topic_only``- names only the topic ("any update on the visa?")
- ``recency``   - no retrievable words at all ("did she reply?", "ok send it"); the most
                  recently used agent overall is correct

Everything is seeded, so runs are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

HOUR = 3600.0
DAY = 24 * HOUR

PEOPLE = [
    "Alice", "Bob", "Priya", "Sharanjeet", "Keith", "Marcus", "Elena", "Tomasz", "Yuki", "Noor",
    "Diego", "Fatima", "Liam", "Ingrid", "Omar", "Hannah", "Rafael", "Mei", "Sven", "Aisha",
    "Carlos", "Zoe", "Dmitri", "Leila", "Jonah", "Sofia", "Arjun", "Greta", "Mateo", "Nia",
    "Felix", "Rosa", "Kenji", "Amara", "Luca", "Ivy", "Hugo", "Tara", "Ravi", "Beatrix",
]

TOPICS = [
    ("lunch", "lunch invite", "schedule lunch"),
    ("dinner", "dinner plans", "book dinner"),
    ("invoice", "October invoice", "chase the invoice"),
    ("contract", "consulting contract", "countersign the contract"),
    ("job offer", "job offer details", "negotiate the offer"),
    ("flight", "flight confirmation", "find the flight booking"),
    ("hotel", "hotel reservation", "confirm the hotel"),
    ("dentist", "dentist appointment", "reschedule the dentist"),
    ("car service", "car service booking", "book the car service"),
    ("rent", "rent payment", "confirm rent transfer"),
    ("birthday gift", "birthday gift ideas", "pick a birthday gift"),
    ("conference talk", "conference talk proposal", "submit the talk"),
    ("reference letter", "reference letter request", "ask for a reference"),
    ("tax documents", "tax documents", "collect tax forms"),
    ("visa", "visa paperwork", "check visa status"),
    ("wedding", "wedding RSVP", "reply to the wedding invite"),
    ("gym", "gym membership", "cancel the gym membership"),
    ("insurance", "insurance claim", "follow up on the claim"),
    ("book club", "book club pick", "vote on the book club pick"),
    ("apartment", "apartment viewing", "schedule the viewing"),
    ("payroll", "payroll discrepancy", "query the payroll error"),
    ("warranty", "laptop warranty", "file the warranty claim"),
    ("vet", "vet appointment", "book the vet"),
    ("tuition", "tuition refund", "request the tuition refund"),
]

NAME_TEMPLATES = [
    "Email to {person} about {topic}",
    "{person} {topic_title}",
    "{topic_title} with {person}",
    "Follow up with {person} - {topic}",
    "Reminder: {topic} ({person})",
]

INSTRUCTION_TEMPLATES = [
    "Draft an email to {person} to {verb}. Keep it short and friendly.",
    "Find the latest email from {person} about the {desc} and summarise where things stand.",
    "Reply to {person}'s last message about the {desc} and ask for a confirmation.",
    "Set a reminder to {verb} with {person} next week.",
    "Check whether {person} has replied about the {desc}; if not, draft a gentle nudge.",
]

SUMMARY_TEMPLATES = [
    "Sent draft to {person} about the {desc}; awaiting reply",
    "{person} confirmed the {desc}; nothing pending",
    "Waiting on {person} for the {desc} since {days}d ago",
    "Reminder scheduled for the {desc}",
]

KEYWORD_QUERIES = [
    "what did {person} say about the {desc}?",
    "any update on the {topic} with {person}",
    "did {person} get back to me about the {desc}",
    "cancel the {topic} thing with {person}",
    "remind me what we told {person} about the {topic}",
]
NAME_ONLY_QUERIES = ["did {person} reply?", "anything from {person}?", "nudge {person} again", "what's the latest with {person}"]
TOPIC_ONLY_QUERIES = ["any update on the {topic}?", "where are we on the {desc}", "is the {topic} sorted yet"]
RECENCY_QUERIES = ["did she reply?", "did he reply?", "any news?", "ok send it", "go ahead", "what did they say", "yes do that", "and then?"]


@dataclass
class SyntheticAgent:
    name: str
    person: str
    topic: str  # short key
    desc: str
    instructions: List[str]
    summary: str
    created_at: float
    last_used_at: float
    use_count: int
    has_trigger: bool = False


@dataclass
class Query:
    text: str
    target: str  # agent name
    kind: str  # keyword | name_only | topic_only | recency
    target_age_days: float


@dataclass
class Workload:
    now: float
    agents: List[SyntheticAgent]
    queries: List[Query]
    seed: int
    span_days: int

    def by_name(self) -> Dict[str, SyntheticAgent]:
        return {a.name: a for a in self.agents}


def generate(n_agents: int, *, seed: int = 7, span_days: int = 21, n_queries: int = 200,
             now: float = 1_800_000_000.0, stratified: bool = False) -> Workload:
    """Build a workload. ``stratified`` draws equal numbers of each query kind (so a 40-query
    live run has 10 keyword-free follow-ups, not 1) instead of sampling the 4:3:2:1 mix."""
    rng = random.Random(seed)
    agents: List[SyntheticAgent] = []
    names_taken = set()

    # Bias person selection so collisions happen (Zipf-ish): a handful of people recur a lot.
    people_weights = [1.0 / (i + 1) ** 0.7 for i in range(len(PEOPLE))]
    topic_weights = [1.0 / (i + 1) ** 0.4 for i in range(len(TOPICS))]

    attempts = 0
    while len(agents) < n_agents and attempts < n_agents * 20:
        attempts += 1
        person = rng.choices(PEOPLE, people_weights)[0]
        topic, desc, verb = rng.choices(TOPICS, topic_weights)[0]
        template = rng.choice(NAME_TEMPLATES)
        name = template.format(person=person, topic=topic, topic_title=desc.title())
        if name in names_taken:
            continue
        names_taken.add(name)
        created = now - rng.uniform(0, span_days * DAY)
        # Agents are used a few times after creation; the last use is somewhere between
        # creation and now, skewed towards creation (most agents are short-lived).
        use_count = rng.choice([1, 1, 1, 2, 2, 3, 4, 6])
        last_used = created + (now - created) * (rng.random() ** 2.5)
        instructions = [
            rng.choice(INSTRUCTION_TEMPLATES).format(person=person, desc=desc, verb=verb)
            for _ in range(min(use_count, 3))
        ]
        summary = rng.choice(SUMMARY_TEMPLATES).format(
            person=person, desc=desc, days=int((now - last_used) / DAY)
        )
        agents.append(
            SyntheticAgent(
                name=name, person=person, topic=topic, desc=desc, instructions=instructions,
                summary=summary, created_at=created, last_used_at=last_used, use_count=use_count,
                has_trigger=(template.startswith("Reminder") or rng.random() < 0.08),
            )
        )

    agents.sort(key=lambda a: a.last_used_at)
    most_recent = agents[-1]
    by_person: Dict[str, List[SyntheticAgent]] = {}
    by_topic: Dict[str, List[SyntheticAgent]] = {}
    by_pair: Dict[Tuple[str, str], List[SyntheticAgent]] = {}
    for a in agents:
        by_person.setdefault(a.person, []).append(a)
        by_topic.setdefault(a.topic, []).append(a)
        by_pair.setdefault((a.person, a.topic), []).append(a)

    queries: List[Query] = []
    kinds = ["keyword"] * 4 + ["name_only"] * 3 + ["topic_only"] * 2 + ["recency"] * 1
    all_kinds = ["keyword", "name_only", "topic_only", "recency"]
    for i in range(n_queries):
        kind = all_kinds[i % 4] if stratified else rng.choice(kinds)
        if kind == "recency":
            target = most_recent
            text = rng.choice(RECENCY_QUERIES)
        elif kind == "name_only":
            person = rng.choice(list(by_person))
            target = max(by_person[person], key=lambda a: a.last_used_at)
            text = rng.choice(NAME_ONLY_QUERIES).format(person=person)
        elif kind == "topic_only":
            topic = rng.choice(list(by_topic))
            target = max(by_topic[topic], key=lambda a: a.last_used_at)
            text = rng.choice(TOPIC_ONLY_QUERIES).format(topic=topic, desc=target.desc)
        else:
            # Several agents may exist for the same person+topic (duplicates the Interaction
            # Agent spawned); the user means the one most recently worked on.
            seed_agent = rng.choice(agents)
            target = max(by_pair[(seed_agent.person, seed_agent.topic)], key=lambda a: a.last_used_at)
            text = rng.choice(KEYWORD_QUERIES).format(
                person=target.person, topic=target.topic, desc=target.desc
            )
        if rng.random() < 0.5:
            text = text.lower()
        queries.append(Query(text=text, target=target.name, kind=kind,
                             target_age_days=(now - target.last_used_at) / DAY))

    return Workload(now=now, agents=agents, queries=queries, seed=seed, span_days=span_days)


def transcript_for(workload: Workload, k: int = 6) -> str:
    """A conversation tail both eval arms see: the last ``k`` interactions, oldest first.

    Each interaction is the user's request to that agent followed by the agent's result in the
    exact format the runtime logs ("[SUCCESS] Agent Name: summary"). It is query-independent,
    so it cannot leak the answer; it simply restores the signal upstream's Interaction Agent
    always had and that a roster-only eval strips.
    """
    recent = sorted(workload.agents, key=lambda a: a.last_used_at)[-k:]
    lines: List[str] = []
    for agent in recent:
        stamp = f"{int(agent.last_used_at)}"
        request = agent.instructions[-1] if agent.instructions else f"Handle {agent.name}"
        lines.append(f'<user_message timestamp="{stamp}">{request}</user_message>')
        lines.append(f'<agent_message timestamp="{stamp}">[SUCCESS] {agent.name}: {agent.summary}</agent_message>')
    return "\n".join(lines)


__all__ = ["DAY", "HOUR", "Query", "SyntheticAgent", "Workload", "generate", "transcript_for"]
