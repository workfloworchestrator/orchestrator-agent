"""Two workflows with a one-page input form, for the eval's form-fill cases. The eval never starts them."""

from typing import Annotated

from fastapi import APIRouter

from orchestrator.core.forms import FormPage
from orchestrator.core.targets import Target
from orchestrator.core.workflow import StepList, done, init, workflow
from orchestrator.core.workflows import LazyWorkflowInstance
from pydantic_forms.types import FormGenerator
from pydantic_forms.validators import Choice, OptionsSource


class Vendor(Choice):
    CISCO = "Cisco"
    NOKIA = "Nokia"


def create_node_form() -> FormGenerator:
    class CreateNodePage(FormPage):
        node_name: str
        vendor: Vendor
        note: str = ""

    user_input = yield CreateNodePage
    return user_input.model_dump()


@workflow("Create a network node", initial_input_form=create_node_form, target=Target.CREATE)
def create_node() -> StepList:
    return init >> done


LazyWorkflowInstance("demo_workflows", "create_node")


def _customer(n: int) -> dict[str, str]:
    name = "Testaccount" if n == 7 else f"Customer {n:02d}"
    return {"customer_id": f"cust-{n:02d}", "fullname": name}


router = APIRouter(prefix="/api/demo")


@router.get("/customers")
def list_customers() -> list[dict[str, str]]:
    """Thirty customers: a list too long for a page schema, so it is searched."""
    return [_customer(n) for n in range(30)]


Customer = Annotated[str, OptionsSource(url="/api/demo/customers", value="customer_id", label="fullname")]


class Speed(Choice):
    ONE = "1G"
    TEN = "10G"


def create_port_form() -> FormGenerator:
    class CreatePortPage(FormPage):
        customer_id: Customer
        speed: Speed

    user_input = yield CreatePortPage
    return user_input.model_dump()


@workflow("Create a customer port", initial_input_form=create_port_form, target=Target.CREATE)
def create_port() -> StepList:
    return init >> done


LazyWorkflowInstance("demo_workflows", "create_port")
