"""Minimal domain models for the seeded demo products."""

from orchestrator.core.domain import SUBSCRIPTION_MODEL_REGISTRY
from orchestrator.core.domain.base import SubscriptionModel
from orchestrator.core.types import SubscriptionLifecycle


class DemoInactive(SubscriptionModel, is_base=True):
    pass


class DemoProvisioning(DemoInactive, lifecycle=[SubscriptionLifecycle.PROVISIONING]):
    pass


class Demo(DemoProvisioning, lifecycle=[SubscriptionLifecycle.ACTIVE, SubscriptionLifecycle.TERMINATED]):
    pass


SUBSCRIPTION_MODEL_REGISTRY.update(
    {
        "node Cisco": Demo,
        "node Nokia": Demo,
        "core link 10G": Demo,
    }
)
