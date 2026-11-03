import os
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Request

from app.core.database import Database
from app.core.http import Decision, Document, ingest, instrument
from app.core.retrieval import Knowledge, configured_encoder
from app.core.security import Principal, authenticate
from app.schemas import Case, Customer
from app.service import Support


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = Database(os.getenv("DATA_PATH", "data/support.sqlite3"))
    app.state.knowledge = Knowledge(db, configured_encoder())
    app.state.service = Support(db, app.state.knowledge)
    yield
    app.state.service.workflow.close()


app = FastAPI(title="Support Case Agent", lifespan=lifespan)
instrument(app)
Identity = Annotated[Principal, Depends(authenticate)]
Key = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=100)]


@app.get("/health")
def health():
    return {"status": "ok", "service": "support-case-agent"}


@app.post("/documents")
def documents(body: Document, request: Request, principal: Identity):
    return ingest(request.app, principal, body)


@app.post("/customers", status_code=201)
def customer(body: Customer, request: Request, principal: Identity):
    return request.app.state.service.put_customer(principal, body)


@app.post("/cases", status_code=201)
def case(body: Case, request: Request, principal: Identity, key: Key):
    try:
        record, _ = request.app.state.service.customer(principal.tenant, body.customer_id)
    except ValueError:
        raise HTTPException(404, "Customer not found") from None
    if body.order_id and body.order_id not in {o["order_id"] for o in record["orders"]}:
        raise HTTPException(404, "Order not found for customer")
    return request.app.state.service.workflow.start(principal, body.model_dump(mode="json"), key)


@app.get("/cases/{run_id}")
def get(run_id: UUID, request: Request, principal: Identity):
    return request.app.state.service.workflow.get(principal, str(run_id))


@app.post("/cases/{run_id}/decision")
def decide(run_id: UUID, body: Decision, request: Request, principal: Identity):
    return request.app.state.service.workflow.decide(
        principal, str(run_id), body.approved, body.proposal_hash
    )


@app.get("/cases/{run_id}/reply")
def reply(run_id: UUID, request: Request, principal: Identity):
    service = request.app.state.service
    service.workflow.get(principal, str(run_id))
    with service.db.connect() as db:
        row = db.execute(
            "SELECT body,hash FROM replies WHERE tenant=? AND run_id=?",
            (principal.tenant, str(run_id)),
        ).fetchone()
    if not row:
        raise HTTPException(404, "No approved reply")
    return {"body": row["body"], "hash": row["hash"], "sent_externally": False}
