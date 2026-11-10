"""What the widget integration tests need of a core: thirty customers over a route, and a task whose fields say where their options are."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter
from orchestrator.core.forms import FormPage
from orchestrator.core.targets import Target
from orchestrator.core.workflow import StepList, done, init, step, workflow
from orchestrator.core.workflows import LazyWorkflowInstance
from pydantic import Field
from pydantic_forms.types import FormGenerator

P1 = UUID("d8a3f9b0-1111-4000-8000-000000000001")
P2 = UUID("d8a3f9b0-1111-4000-8000-000000000002")

# Written as plain schema keys so the image's pydantic-forms need not know ``OptionsSource``: the agent reads the keyword.
CUSTOMER_SOURCE = {"url": "/api/demo/customers", "value": "customer_id", "label": "{fullname} ({shortcode})"}
PRODUCT_SOURCE = {"url": "/api/products/", "value": "product_id", "label": "name"}

Customer = Annotated[str, Field(json_schema_extra={"format": "customerId", "optionsSource": CUSTOMER_SOURCE})]
Product = Annotated[UUID, Field(json_schema_extra={"format": "productId", "optionsSource": PRODUCT_SOURCE})]


def _customer(n: int) -> dict[str, str]:
    if n == 7:
        return {"customer_id": "cust-07", "fullname": "Testaccount", "shortcode": "TA"}
    return {"customer_id": f"cust-{n:02d}", "fullname": f"Customer {n:02d}", "shortcode": f"C{n:02d}"}


CUSTOMERS = [_customer(n) for n in range(30)]

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.get("/customers")
def list_customers() -> list[dict[str, str]]:
    """Thirty customers, as a deployment's CRM-backed route would return them."""
    return CUSTOMERS


def widget_demo_form() -> FormGenerator:
    class WidgetDemoPage(FormPage):
        customer_id: Customer
        product_id: Product
        customers: list[Customer] = []
        note: str = ""

    user_input = yield WidgetDemoPage
    return user_input.model_dump()


@step("Echo the customer")
def echo(customer_id: str) -> dict:
    return {"echoed": customer_id}


@workflow(initial_input_form=widget_demo_form, target=Target.SYSTEM)
def widget_demo() -> StepList:
    return init >> echo >> done


LazyWorkflowInstance("demo", "widget_demo")
