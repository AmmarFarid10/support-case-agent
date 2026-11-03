"""Exercise the real FastAPI API against temporary storage; never send external messages."""

import argparse
import json
import os
import secrets
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient

from app.api import app

ROOT = Path(__file__).resolve().parents[1]


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    args = parser.parse_args()
    operator, reviewer = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    os.environ["AUTH_KEYS_JSON"] = json.dumps(
        [
            {"tenant": "demo", "subject": "alice", "role": "operator", "key": operator},
            {"tenant": "demo", "subject": "bob", "role": "reviewer", "key": reviewer},
        ]
    )
    os.environ["EMBEDDER"] = "fastembed" if args.semantic else "hash"
    with tempfile.TemporaryDirectory() as temp:
        os.environ["DATA_PATH"] = str(Path(temp) / "demo.sqlite3")
        with TestClient(app) as client:
            client.headers["Authorization"] = "Bearer " + operator
            summary = exercise(client, operator, reviewer)
    print(json.dumps(summary, indent=2))
    return summary


def exercise(client, operator, reviewer):
    sample = json.loads((ROOT / "examples/scenario.json").read_text())
    client.headers["Authorization"] = "Bearer " + reviewer
    for doc in sample["documents"]:
        assert client.post("/documents", json=doc).status_code == 200
    client.headers["Authorization"] = "Bearer " + operator
    assert client.post("/customers", json=sample["customer"]).status_code == 201
    response = client.post("/cases", json=sample["case"], headers={"Idempotency-Key": "demo-case"})
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["status"] == "AWAITING_APPROVAL", result
    assert "fraud" not in result["proposal"]["body"].lower()
    endpoint = "/cases/" + result["id"]
    decision = {"approved": True, "proposal_hash": result["proposal_hash"]}
    assert client.post(endpoint + "/decision", json=decision).status_code == 403
    client.headers["Authorization"] = "Bearer " + reviewer
    approved = client.post(endpoint + "/decision", json=decision).json()
    assert approved["status"] == "APPROVED", approved
    reply = client.get(endpoint + "/reply").json()
    assert not reply["sent_externally"]
    return {
        "scenario": "Damaged industrial switch support case",
        "category": result["proposal"]["category"],
        "status": approved["status"],
        "policy_claims": len(result["proposal"]["claims"]),
        "reply_ready": approved["result"]["reply_ready"],
        "sent_externally": False,
    }


if __name__ == "__main__":
    main()
