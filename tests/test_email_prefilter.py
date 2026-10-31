"""The pre-filter must never dismiss an email the labelled set marks important."""

from __future__ import annotations

import pytest

from evals.email_eval import eval_prefilter, load_fixtures, to_processed
from server.services.gmail.prefilter import prefilter

ROWS = load_fixtures()


@pytest.mark.parametrize("row", [r for r in ROWS if r["important"]], ids=lambda r: r["id"])
def test_important_emails_always_reach_the_model(row):
    assert prefilter(to_processed(row)).skip is False, row["note"]


def test_prefilter_dismisses_a_useful_share_of_noise():
    metrics = eval_prefilter(ROWS)
    assert metrics["skip_precision"] == 1.0
    assert metrics["noise_caught_by_rules"] >= 0.6, metrics


def test_otp_from_noreply_is_kept():
    row = next(r for r in ROWS if r["id"] == "i03")
    assert prefilter(to_processed(row)).reason == "keep-signal"


def test_fraud_alert_from_alerts_sender_reaches_model():
    row = {"id": "x", "sender": "Bank <alerts@bank.com>", "subject": "Unusual activity on your card",
           "body": "We noticed a charge of $900. If this wasn't you call us. Opt-out of alerts.", "labels": ["INBOX"]}
    assert prefilter(to_processed(row)).skip is False
