# Copyright 2019-2026 SURF, GÉANT.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The widgets for the formats orchestrator-core itself defines: ``customerId`` and ``productId``.

Every other format a form may carry (subscriptions, ports, contacts, locations, ...) belongs to a deployment
and comes with its extender.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, TypeAdapter

from orchestrator_agent.form_fill.widgets.base import FieldWidget, Option, Widget, WidgetContext, field_hint
from orchestrator_agent.tool_names import LIST_PRODUCTS_TOOL

# The frontend's customer select asks the same (``useGetCustomersQuery``): every customer, once.
CUSTOMERS_QUERY = "query Customers { customers(first: 1000000, after: 0) { page { customerId fullname shortcode } } }"


class _Customer(BaseModel):
    customerId: str
    fullname: str
    shortcode: str = ""


class _Product(BaseModel):
    """What a widget needs of a row of ``list_products`` (core's ``ProductSchema``)."""

    product_id: str
    name: str


_CUSTOMERS: TypeAdapter[list[_Customer]] = TypeAdapter(list[_Customer])
_PRODUCTS: TypeAdapter[list[_Product]] = TypeAdapter(list[_Product])


def _customer_option(customer: _Customer) -> Option:
    label = f"{customer.fullname} ({customer.shortcode})" if customer.shortcode else customer.fullname
    names = (customer.fullname, customer.shortcode) if customer.shortcode else (customer.fullname,)
    return Option(customer.customerId, label, aliases=names)


class CustomerIdWidget(Widget):
    """``CustomerId``: a customer of core's ``customers`` query, shown by name and shortcode."""

    id = "customerId"

    def matches(self, field: Mapping[str, Any]) -> bool:
        return field.get("type") == "string" and field.get("format") == "customerId"

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        data = await ctx.graphql(CUSTOMERS_QUERY)
        return [_customer_option(customer) for customer in _CUSTOMERS.validate_python(data["customers"]["page"])]


class ProductIdWidget(Widget):
    """``ProductId`` / ``product_id([...])``: a product of core's catalogue, restricted to its ``productIds``."""

    id = "productId"

    def matches(self, field: Mapping[str, Any]) -> bool:
        return field.get("format") == "productId"

    async def fetch(self, field: Mapping[str, Any], ctx: WidgetContext) -> Sequence[Option] | None:
        allowed = {str(product_id) for product_id in field_hint(field, "productIds") or ()}
        products = _PRODUCTS.validate_python(await ctx.call_tool(LIST_PRODUCTS_TOOL, {}))
        return [
            Option(product.product_id, product.name)
            for product in products
            if not allowed or product.product_id in allowed
        ]


BUILTIN_WIDGETS: tuple[FieldWidget, ...] = (CustomerIdWidget(), ProductIdWidget())

__all__ = ["BUILTIN_WIDGETS", "CUSTOMERS_QUERY", "CustomerIdWidget", "ProductIdWidget"]
