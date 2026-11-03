from typing import Literal

from pydantic import Field, model_validator

from app.core.http import StrictModel


class Order(StrictModel):
    order_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    status: Literal["processing", "shipped", "delivered"]
    delivered_days_ago: int | None = Field(default=None, ge=0, le=3650)

    @model_validator(mode="after")
    def delivery(self):
        if self.status == "delivered" and self.delivered_days_ago is None:
            raise ValueError("Delivered orders need verified delivery age")
        if self.status != "delivered" and self.delivered_days_ago is not None:
            raise ValueError("Undelivered orders cannot have delivery age")
        return self


class Customer(StrictModel):
    customer_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    tier: Literal["standard", "business"] = "standard"
    orders: list[Order] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def ids(self):
        if len({o.order_id for o in self.orders}) != len(self.orders):
            raise ValueError("Duplicate order IDs")
        return self


class Case(StrictModel):
    customer_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    order_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_.-]{1,80}$")
    message: str = Field(min_length=1, max_length=4000)
    planner: Literal["rules", "ollama"] = "rules"
    draft: Literal["extractive", "ollama"] = "extractive"


class Plan(StrictModel):
    category: Literal["returns", "shipping", "billing", "other"]
    tools: list[Literal["lookup_customer", "lookup_order", "search_policy"]] = Field(
        min_length=2, max_length=3
    )


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    quote: str = Field(min_length=1, max_length=3000)
    chunk_id: str


class Draft(StrictModel):
    claims: list[Claim] = Field(max_length=5)
    escalate: bool
