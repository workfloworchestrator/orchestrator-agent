"""The rows core needs for the widget demo: three products, and the task (a process needs its workflows row)."""

from orchestrator.core import app_settings
from orchestrator.core.db import ProductTable, WorkflowTable, db, init_database
from sqlalchemy import select

PRODUCTS = [
    ("d8a3f9b0-1111-4000-8000-000000000001", "Widget Port 10G"),
    ("d8a3f9b0-1111-4000-8000-000000000002", "Widget Port 1G"),
    ("d8a3f9b0-1111-4000-8000-000000000003", "Widget Node"),
]


def seed() -> None:
    if db.session.scalars(select(WorkflowTable).filter(WorkflowTable.name == "widget_demo")).first() is not None:
        print("seed: already present, skipping")
        return
    db.session.add_all(
        [
            *(
                ProductTable(
                    product_id=pid, name=name, description=name, product_type="Widget", tag="WIDGET", status="active"
                )
                for pid, name in PRODUCTS
            ),
            WorkflowTable(name="widget_demo", target="SYSTEM", description="Widget demo", is_task=True),
        ]
    )
    db.session.commit()
    print("seed: created 3 products and the widget_demo task")


if __name__ == "__main__":
    init_database(app_settings)
    seed()
