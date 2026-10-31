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
    assert metrics["noise_caught_by_rules"] >= 0.3, metrics


def test_otp_from_noreply_is_kept():
    row = next(r for r in ROWS if r["id"] == "i03")
    assert prefilter(to_processed(row)).reason == "keep-signal"


def test_fraud_alert_from_alerts_sender_reaches_model():
    row = {"id": "x", "sender": "Bank <alerts@bank.com>", "subject": "Unusual activity on your card",
           "body": "We noticed a charge of $900. If this wasn't you call us. Opt-out of alerts.", "labels": ["INBOX"]}
    assert prefilter(to_processed(row)).skip is False


@pytest.mark.parametrize("row_id", ["a01", "a02", "a03", "a04", "a05"])
def test_alert_style_mail_with_footers_reaches_model(row_id):
    row = next(r for r in ROWS if r["id"] == row_id)
    assert prefilter(to_processed(row)).skip is False, row["note"]


def test_only_spam_and_promotions_are_hard_skips():
    reasons = {prefilter(to_processed(r)).reason for r in ROWS if prefilter(to_processed(r)).skip}
    assert reasons <= {"label:spam", "label:promotions+unsubscribe"}, reasons


def test_marketing_digest_with_here_colon_is_skipped():
    row = next(r for r in ROWS if r["id"] == "a06")
    assert prefilter(to_processed(row)).skip is True


def test_missing_body_never_raises():
    row = {"id": "x", "sender": "news@shop.com", "subject": "Sale", "body": None, "labels": ["CATEGORY_PROMOTIONS"]}
    assert prefilter(to_processed(row)).skip is False


# ---- round three: held-out set (written before the classifier prompt changed) ------------

HELDOUT = load_fixtures("emails_heldout")


def test_heldout_set_shape():
    assert len(HELDOUT) == 14
    assert {r["id"][0] for r in HELDOUT} == {"h"}
    assert 5 <= sum(r["important"] for r in HELDOUT) <= 9


@pytest.mark.parametrize("row", [r for r in HELDOUT if r["important"]], ids=lambda r: r["id"])
def test_heldout_important_reach_model(row):
    assert prefilter(to_processed(row)).skip is False, row["note"]
