# Authentication and authorization

The agent sits between a client and Orchestrator Core, so identity has two directions. **Inbound** is about who may
call the agent. **Outbound** is about who the agent acts as when it calls Orchestrator Core for its tools. They are
set up separately, but what a user can actually do through the agent comes from how the two work together.

To set it up, see [How to secure the agent](../guides/secure-agent.md). The settings, status codes and the input the
agent sends to OPA are listed in [Endpoints: authentication](../reference/endpoints.md#authentication).

## The same model as Orchestrator Core

The agent uses `oauth2_lib`, the library behind
[authentication and authorization in Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/reference-docs/auth-backend-and-frontend/):
OIDC to find out who is calling, and an Open Policy Agent (OPA) to decide whether they may. The reason is that the
agent is a client of Orchestrator Core, not a separate system. One identity provider, one kind of token and one
policy language cover both. A user's token can then pass from the agent to Orchestrator Core unchanged, and policies
for both can live in the same place.

## Inbound: who may call the agent

Every request, on every protocol, passes the same two checks. Only the health check and the A2A agent card are open,
because monitoring and agent discovery happen before anyone signs in.

### Authentication: who is this?

The agent does not validate the token itself. It asks the identity provider, through the OIDC userinfo endpoint, who
the token belongs to. This is a trade-off:

- **It works with any OIDC provider**, without the agent having to know signing keys, audiences or the claim
  layout of the provider. Providers with incomplete discovery documents, such as AWS Cognito, work as well.
- **A revoked token stops working at once**, because nothing is cached.
- **Each request costs a round trip to the identity provider.** That adds latency, and the provider has to handle
  the load. Validating the JWT locally would be faster, but it would trust a token until it expires.

For a chat agent, where a single answer already waits seconds for the model, the extra round trip barely shows.

### Authorization: may they do this?

After authentication the agent asks OPA for a decision about the request: this user, this endpoint, this body. The
answer is yes or no for the whole request.

That makes OPA in the agent a coarse filter. It decides **who may use which protocol**, not **which tools they may
run**. It cannot do the second, because the agent picks its tools only after the check, based on the question.
Which tools a user may actually run is decided one hop later, by Orchestrator Core.

So the agent's OPA mostly earns its place when you want to keep people away from the agent altogether, or from one
of its protocols. If every user with a valid token may use the agent, it adds a moving part without changing the
outcome, and turning it off is a reasonable choice.

One side effect to keep in mind: OPA receives the request body, and on the chat protocols that body holds the user's
message. Decision logs in OPA will therefore contain prompts.

## Outbound: who the agent acts as

The agent calls the MCP server of Orchestrator Core for all its tools: searching, reading subscriptions and starting
workflows. Orchestrator Core runs its own authentication and OPA checks on those calls. So the question that matters
most for safety is not "may this user call the agent?" but "whose token does the agent send to Orchestrator Core?".
There are two answers.

- **The user's token.** The agent forwards the token of the request to Orchestrator Core for that run. Orchestrator
  Core sees the real user, applies that user's policies and records that user in its audit log. The agent can do no
  more than the user could do in the Orchestrator UI.
- **A service account.** When there is no user token, the agent signs in as itself with the OAuth2
  client-credentials grant. Orchestrator Core then sees the agent, not the person.

Which of the two applies depends on the protocol:

| Protocol | Where the agent finds the user's token | Orchestrator Core sees |
| --- | --- | --- |
| OpenAI chat completions (`/v1`) | `Authorization` header | the user |
| AG-UI | `Authorization` header | the user |
| A2A | `auth_token` in the message metadata | the user, if the token is there; otherwise the service account |
| The agent's MCP (`/mcp`) | not forwarded | always the service account |

Two rows stand out. On A2A, the HTTP header only gets the caller past the agent's own check. The token that reaches
Orchestrator Core travels inside the A2A message, so a planner that sends only the header makes the agent act as its
service account. On the agent's MCP endpoint the user's token is not passed on at all: the caller is checked
inbound, but Orchestrator Core always sees the service account.

### Why the user's token is better than a service account

A service account has to be allowed everything that any of its users needs. Every user then gets the sum of those
rights, and the audit log of Orchestrator Core names the agent instead of the person. A service account is simpler to
set up and works for clients that have no user token, such as scheduled jobs, but it moves the decision about what a
user may do out of Orchestrator Core and into whoever picks the service account's rights.

Forwarding the user's token has its own limit. It only works when the agent and Orchestrator Core accept the same
token, which means the token has to be issued for an audience both trust. When they don't,
[token exchange (RFC 8693)](https://datatracker.ietf.org/doc/html/rfc8693) is the standard answer: it swaps the
user's token for one meant for the next hop and keeps the user's identity. The agent does not do this itself; an
identity provider or a gateway in front of it can.

## What the agent does not do: gateways

The agent answers two questions about a caller: who they are, and whether they may use it. It does not ask how often
or how much: there are no rate limits, no budgets per user or per agent, and no filtering of which tools a caller can
see. It also only guards its own front door. Clients can reach the MCP server of Orchestrator Core and other agents
directly, without passing the agent.

These are questions for a layer in front of all of them, not for each agent on its own. The
[example orchestrator](https://github.com/workfloworchestrator/example-orchestrator) shows the minimal version: a small
proxy that lets LibreChat talk to agents over A2A, but accepts any API key. That is fine on a laptop and a liability
anywhere else. In [gateways](production.md#gateways) you can read more about a gateway can solve these issues.