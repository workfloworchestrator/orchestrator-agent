"""Deterministic demo data for the eval stack."""

from datetime import datetime, timezone

from sqlalchemy import select

from orchestrator.core import app_settings
from orchestrator.core.db import ProductTable, SubscriptionTable, db, init_database

CUSTOMER_ID = "e6c5c2ae-1b3a-4f5f-9c1e-000000000001"

PRODUCTS = [
    ("d8a3f9b0-0000-4000-8000-000000000001", "node Cisco", "Cisco network node", "Node", "NODE"),
    ("d8a3f9b0-0000-4000-8000-000000000002", "node Nokia", "Nokia network node", "Node", "NODE"),
    ("d8a3f9b0-0000-4000-8000-000000000003", "core link 10G", "10G core link between nodes", "CoreLink", "CORE_LINK"),
]

SUBSCRIPTIONS = [
    # (uuid, description, product idx, status, insync, start_date)
    (
        "aa1b2c3d-0000-4000-8000-000000000001",
        "node asd001 (active)",
        0,
        "active",
        True,
        datetime(2026, 1, 10, tzinfo=timezone.utc),
    ),
    (
        "aa1b2c3d-0000-4000-8000-000000000002",
        "node ams002 (active)",
        1,
        "active",
        True,
        datetime(2026, 2, 11, tzinfo=timezone.utc),
    ),
    (
        "aa1b2c3d-0000-4000-8000-000000000003",
        "core link 10G asd001 <-> ams002",
        2,
        "active",
        True,
        datetime(2026, 3, 12, tzinfo=timezone.utc),
    ),
    (
        "aa1b2c3d-0000-4000-8000-000000000004",
        "node rtd003 (terminated)",
        0,
        "terminated",
        True,
        datetime(2025, 6, 1, tzinfo=timezone.utc),
    ),
    (
        "aa1b2c3d-0000-4000-8000-000000000005",
        "node utr004 (provisioning)",
        1,
        "provisioning",
        False,
        datetime(2026, 4, 13, tzinfo=timezone.utc),
    ),
]


def seed() -> None:
    if db.session.scalars(select(ProductTable).limit(1)).first() is not None:
        print("seed: products already present, skipping")
        return
    products = [
        ProductTable(product_id=pid, name=name, description=desc, product_type=ptype, tag=tag, status="active")
        for pid, name, desc, ptype, tag in PRODUCTS
    ]
    subscriptions = [
        SubscriptionTable(
            subscription_id=sid,
            description=desc,
            status=status,
            insync=insync,
            product=products[idx],
            customer_id=CUSTOMER_ID,
            start_date=start,
        )
        for sid, desc, idx, status, insync, start in SUBSCRIPTIONS
    ]
    db.session.add_all(products + subscriptions)
    db.session.commit()
    print(f"seed: created {len(products)} products, {len(subscriptions)} subscriptions")


if __name__ == "__main__":
    init_database(app_settings)
    seed()
