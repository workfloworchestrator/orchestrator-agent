"""What the widget integration tests need of a core: thirty customers over GraphQL and a task asking for them."""

from uuid import UUID

import strawberry
from oauth2_lib.strawberry import authenticated_field
from orchestrator.core.forms import FormPage
from orchestrator.core.forms.validators import CustomerId
from orchestrator.core.forms.validators.product_id import product_id
from orchestrator.core.graphql import Query
from orchestrator.core.graphql.pagination import Connection
from orchestrator.core.graphql.schemas.customer import CustomerType
from orchestrator.core.graphql.types import GraphqlFilter, GraphqlSort, OrchestratorInfo
from orchestrator.core.graphql.utils.to_graphql_result_page import to_graphql_result_page
from orchestrator.core.targets import Target
from orchestrator.core.workflow import StepList, done, init, step, workflow
from orchestrator.core.workflows import LazyWorkflowInstance
from pydantic_forms.types import FormGenerator

P1 = UUID("d8a3f9b0-1111-4000-8000-000000000001")
P2 = UUID("d8a3f9b0-1111-4000-8000-000000000002")


def _customer(n: int) -> CustomerType:
    if n == 7:
        return CustomerType(customer_id="cust-07", fullname="Testaccount", shortcode="TA")
    return CustomerType(customer_id=f"cust-{n:02d}", fullname=f"Customer {n:02d}", shortcode=f"C{n:02d}")


CUSTOMERS = [_customer(n) for n in range(30)]


async def resolve_customers(
    info: OrchestratorInfo,
    filter_by: list[GraphqlFilter] | None = None,
    sort_by: list[GraphqlSort] | None = None,
    first: int = 10,
    after: int = 0,
) -> Connection[CustomerType]:
    """Thirty customers, as a deployment's CRM-backed resolver would return them (core's own knows one)."""
    return to_graphql_result_page(CUSTOMERS[after : after + first + 1], first, after, len(CUSTOMERS))


@strawberry.type(description="Orchestrator queries")
class DemoQuery(Query):
    customers: Connection[CustomerType] = authenticated_field(resolver=resolve_customers, description="Customers")


def widget_demo_form() -> FormGenerator:
    class WidgetDemoPage(FormPage):
        customer_id: CustomerId
        product_id: product_id([P1, P2])  # type: ignore[valid-type]
        customers: list[CustomerId] = []
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
