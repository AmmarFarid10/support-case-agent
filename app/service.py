import json
import re
from typing import TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.core.database import Database, digest
from app.core.model import LocalModel, ModelError
from app.core.retrieval import Knowledge, validate_quotes
from app.core.security import Principal
from app.core.workflow import Conflict, Workflow
from app.schemas import Customer, Draft, Plan


class State(TypedDict, total=False):
    tenant: str
    run_id: str
    creator: str
    role: str
    input: dict
    plan: dict
    customer_revision: str
    facts: dict
    evidence: list[dict]
    policy_snapshot: list[dict]
    proposal: dict
    proposal_hash: str
    expires_at: float
    decision: dict
    events: list[dict]
    status: str


def classify(message: str) -> str:
    words = set(re.findall(r"[a-z]+", message.lower()))
    if words & {"return", "returns", "refund", "damaged", "defective"}:
        return "returns"
    if words & {"delivery", "shipping", "shipment", "shipped", "tracking", "arrive"}:
        return "shipping"
    if words & {"invoice", "billing", "charged", "payment", "charge"}:
        return "billing"
    return "other"


class Support:
    def __init__(self, db: Database, knowledge: Knowledge, clock=None, model=None):
        self.db, self.knowledge, self.model = db, knowledge, model or LocalModel()
        with db.connect() as connection:
            connection.executescript("""
            CREATE TABLE IF NOT EXISTS customers(tenant TEXT,id TEXT,revision TEXT,body TEXT,
                PRIMARY KEY(tenant,id));
            CREATE TABLE IF NOT EXISTS replies(tenant TEXT,run_id TEXT,body TEXT,hash TEXT,
                PRIMARY KEY(tenant,run_id));
            """)
        options = {"clock": clock} if clock else {}
        self.workflow = Workflow(db, self.build, self.effect, **options)

    def put_customer(self, principal: Principal, body: Customer) -> dict:
        document = body.model_dump(mode="json")
        revision = digest(document)
        with self.db.connect() as db:
            db.execute(
                "INSERT INTO customers VALUES(?,?,?,?) ON CONFLICT(tenant,id) DO UPDATE SET "
                "revision=excluded.revision,body=excluded.body",
                (principal.tenant, body.customer_id, revision, json.dumps(document)),
            )
            self.db.audit(
                db,
                principal.tenant,
                principal.subject,
                "customer_updated",
                body.customer_id,
                {"revision": revision, "orders": len(body.orders)},
            )
        return {"customer_id": body.customer_id, "revision": revision}

    def customer(self, tenant: str, customer_id: str) -> tuple[dict, str]:
        with self.db.connect() as db:
            row = db.execute(
                "SELECT body,revision FROM customers WHERE tenant=? AND id=?", (tenant, customer_id)
            ).fetchone()
        if not row:
            raise ValueError("Customer not found")
        return json.loads(row["body"]), row["revision"]

    def build(self, workflow: Workflow, saver):
        def plan(state: State) -> dict:
            payload = state["input"]
            expected = (
                ["lookup_customer"]
                + (["lookup_order"] if payload.get("order_id") else [])
                + ["search_policy"]
            )
            category = classify(payload["message"])
            if payload.get("planner") == "ollama":
                result = self.model.structured(
                    Plan,
                    "Classify the support issue. Select lookup_customer, lookup_order only when an order ID "
                    "is present, then search_policy, in that order. Never send a message or change an order.",
                    {
                        "message": payload["message"],
                        "has_order_id": bool(payload.get("order_id")),
                        "available_tools": ["lookup_customer", "lookup_order", "search_policy"],
                    },
                )
                if result.tools != expected:
                    raise ModelError("Plan attempted invalid tools or order")
                category = result.category
            return {
                "plan": {"category": category, "tools": expected},
                "events": [{"step": "plan", "tool_count": len(expected)}],
            }

        def tools(state: State) -> dict:
            customer, revision = self.customer(state["tenant"], state["input"]["customer_id"])
            # Customer-visible drafts never retrieve reviewer-only internal notes.
            scope = Principal(state["tenant"], state["creator"], "operator")
            policy_snapshot = self.knowledge.snapshot(scope, [state["plan"]["category"]])

            @tool
            def lookup_customer() -> dict:
                """Read the selected customer's account tier from the tenant-scoped repository."""
                return {"customer_id": customer["customer_id"], "tier": customer["tier"]}

            @tool
            def lookup_order() -> dict:
                """Read only an order belonging to the selected customer."""
                order = next(
                    (o for o in customer["orders"] if o["order_id"] == state["input"]["order_id"]),
                    None,
                )
                if order is None:
                    raise ValueError("Order does not belong to customer")
                return order

            @tool
            def search_policy() -> dict:
                """Retrieve current team-visible policy for the selected support category."""
                category = state["plan"]["category"]
                if category == "other":
                    return {"evidence": []}
                question = {
                    "returns": "return refund damaged eligibility window",
                    "shipping": "delivery shipping tracking lost parcel",
                    "billing": "billing invoice payment charge",
                }[category]
                return {"evidence": self.knowledge.search(scope, question, [category], limit=4)}

            registry = {t.name: t for t in [lookup_customer, lookup_order, search_policy]}
            facts = {}
            evidence = []
            for name in state["plan"]["tools"]:
                observation = registry[name].invoke({})
                if name == "search_policy":
                    evidence = observation["evidence"]
                else:
                    facts[name] = observation
            return {
                "facts": facts,
                "evidence": evidence,
                "customer_revision": revision,
                "policy_snapshot": policy_snapshot,
                "events": state["events"] + [{"step": name} for name in state["plan"]["tools"]],
            }

        def propose(state: State) -> dict:
            evidence = state["evidence"]
            claims = [
                {"text": c["text"], "quote": c["text"], "chunk_id": c["id"]} for c in evidence
            ]
            escalate = not evidence or state["plan"]["category"] == "other"
            if state["input"].get("draft") == "ollama" and evidence:
                result = self.model.structured(
                    Draft,
                    "Select exact policy quotations relevant to the support question. All claim text must "
                    "be copied from its quote, with the correct chunk ID. Do not invent return eligibility, "
                    "refund promises, prices or shipping dates. Escalate when evidence is insufficient.",
                    {
                        "message": state["input"]["message"],
                        "facts": state["facts"],
                        "evidence": evidence,
                        "policy_snapshot": state["policy_snapshot"],
                    },
                )
                claims = [c.model_dump() for c in result.claims]
                escalate = result.escalate
            if not validate_quotes(claims, evidence):
                claims, escalate = [], True
            # Facts are rendered from verified records, not from model text.
            paragraphs = ["Thank you for contacting support."]
            order = state["facts"].get("lookup_order")
            if order:
                paragraphs.append(
                    f"Order {order['order_id']} currently has status: {order['status']}."
                )
            paragraphs.extend(c["text"] for c in claims)
            if escalate:
                paragraphs.append(
                    "This case needs further review before we can confirm the next step."
                )
            else:
                paragraphs.append("A support reviewer will confirm the applicable next step.")
            proposal = {
                "customer_id": state["input"]["customer_id"],
                "customer_revision": state["customer_revision"],
                "category": state["plan"]["category"],
                "escalate": escalate,
                "body": "\n\n".join(paragraphs),
                "claims": claims,
                "facts": state["facts"],
                "evidence": evidence,
                "policy_snapshot": state["policy_snapshot"],
                "events": state["events"],
                "delivery": "local-reviewed-reply-only",
            }
            return workflow.offer(dict(state), proposal)

        def review(state: State) -> dict:
            return {
                "decision": interrupt(
                    {
                        "proposal": state["proposal"],
                        "proposal_hash": state["proposal_hash"],
                        "expires_at": state["expires_at"],
                    }
                )
            }

        def commit(state: State) -> dict:
            return workflow.finish(dict(state), state["decision"])

        graph = StateGraph(State)
        for name, fn in [
            ("plan", plan),
            ("tools", tools),
            ("propose", propose),
            ("review", review),
            ("commit", commit),
        ]:
            graph.add_node(name, fn)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "tools")
        graph.add_edge("tools", "propose")
        graph.add_edge("propose", "review")
        graph.add_edge("review", "commit")
        graph.add_edge("commit", END)
        return graph.compile(checkpointer=saver)

    def effect(self, db, principal: Principal, state: dict, proposal: dict) -> dict:
        row = db.execute(
            "SELECT revision FROM customers WHERE tenant=? AND id=?",
            (principal.tenant, proposal["customer_id"]),
        ).fetchone()
        if not row or row["revision"] != proposal["customer_revision"]:
            raise Conflict("Customer facts changed after draft")
        scope = Principal(principal.tenant, principal.subject, "operator")
        if (
            self.knowledge.snapshot(scope, [proposal["category"]], db)
            != proposal["policy_snapshot"]
        ):
            raise Conflict("Policy changed after draft")
        if proposal["claims"] and not validate_quotes(proposal["claims"], proposal["evidence"]):
            raise Conflict("Reply claims are unsupported")
        db.execute(
            "INSERT INTO replies VALUES(?,?,?,?)",
            (principal.tenant, state["run_id"], proposal["body"], digest(proposal["body"])),
        )
        return {
            "reply_ready": True,
            "reply_hash": digest(proposal["body"]),
            "escalated": proposal["escalate"],
            "sent_externally": False,
        }
