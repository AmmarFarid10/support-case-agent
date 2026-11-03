import pytest
from conftest import auth
from test_workflow import SAMPLE, decide, seed, start

from app.schemas import Draft, Plan


def test_draft_uses_verified_order_and_excludes_reviewer_notes(client):
    seed(client)
    result = start(client).json()
    proposal = result["proposal"]
    assert proposal["category"] == "returns"
    assert "O-100" in proposal["body"] and "delivered" in proposal["body"]
    assert "fraud" not in proposal["body"].lower()
    assert all(c["text"] in c["quote"] for c in proposal["claims"])
    assert proposal["delivery"] == "local-reviewed-reply-only"


def test_nonexistent_or_other_customers_order_is_rejected(client):
    seed(client)
    result = start(client, body={**SAMPLE["case"], "order_id": "other-order"})
    assert result.status_code == 404


def test_customer_update_invalidates_pending_reply(client):
    seed(client)
    result = start(client).json()
    changed = {**SAMPLE["customer"], "tier": "standard"}
    client.post("/customers", json=changed)
    auth(client, "b")
    assert decide(client, result).json()["status"] == "CONFLICT"
    assert client.get("/cases/" + result["id"] + "/reply").status_code == 404


def test_new_policy_invalidates_pending_reply(client):
    seed(client)
    result = start(client).json()
    client.post(
        "/documents",
        json={
            "source": "new-policy",
            "title": "New policy",
            "entity": "returns",
            "text": "Return rules changed and require manual review.",
        },
    )
    auth(client, "b")
    assert decide(client, result).json()["status"] == "CONFLICT"


def test_unsupported_question_escalates_without_inventing_answer(client):
    seed(client)
    result = start(
        client, body={**SAMPLE["case"], "message": "Can you help me with planetary astronomy?"}
    ).json()
    assert result["proposal"]["escalate"] and not result["proposal"]["claims"]


def test_fabricated_model_claim_is_removed_and_escalated(client, monkeypatch):
    seed(client)
    fake = Draft.model_validate(
        {
            "claims": [
                {
                    "text": "You have a guaranteed refund",
                    "quote": "Guaranteed refund",
                    "chunk_id": "invented",
                }
            ],
            "escalate": False,
        }
    )
    monkeypatch.setattr(client.app.state.service.model, "structured", lambda *_: fake)
    result = start(client, body={**SAMPLE["case"], "draft": "ollama"}).json()
    assert result["proposal"]["escalate"]
    assert "guaranteed refund" not in result["proposal"]["body"].lower()


def test_model_cannot_select_write_tools_or_skip_read_dependencies(client, monkeypatch):
    seed(client)
    fake = Plan.model_validate(
        {"category": "returns", "tools": ["search_policy", "lookup_customer"]}
    )
    monkeypatch.setattr(client.app.state.service.model, "structured", lambda *_: fake)
    result = start(client, body={**SAMPLE["case"], "planner": "ollama"}).json()
    assert result["status"] == "FAILED" and result["proposal"] is None


def test_approval_never_sends_a_real_message(client):
    seed(client)
    result = start(client).json()
    auth(client, "b")
    approved = decide(client, result).json()
    assert approved["result"]["sent_externally"] is False
    assert client.get("/cases/" + result["id"] + "/reply").json()["sent_externally"] is False


@pytest.mark.parametrize(
    "order",
    [
        {"order_id": "x", "status": "delivered"},
        {"order_id": "x", "status": "shipped", "delivered_days_ago": 3},
    ],
)
def test_inconsistent_delivery_records_are_rejected(client, order):
    assert (
        client.post("/customers", json={"customer_id": "x", "orders": [order]}).status_code == 422
    )
