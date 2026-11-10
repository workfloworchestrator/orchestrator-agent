# Authentication and authorization

The agent sits between a client and Orchestrator Core, so identity has two directions. **Inbound** is about who may
call the agent. **Outbound** is about who the agent acts as when it calls Orchestrator Core for its tools. They are
set up separately, but what a user can actually do through the agent comes from how the two work together.

To set it up, see [How to configure authentication and authorization](../guides/secure-agent.md). The settings, status
codes and input sent to OPA are listed in [Endpoints: authentication](../reference/endpoints.md#authentication).

## The same model as Orchestrator Core

The agent uses `oauth2_lib`, the library behind
[authentication and authorization in Orchestrator Core](https://workfloworchestrator.org/orchestrator-core/reference-docs/auth-backend-and-frontend/):
OIDC to find out who is calling, and an Open Policy Agent (OPA) to decide whether they may. The reason is that the
agent is a client of Orchestrator Core, not a separate system. One identity provider, one kind of token and one
policy language cover both. A user's token can then pass from the agent to Orchestrator Core unchanged, and policies
for both can live in the same place.

## Inbound: who may call the agent

With both checks enabled, requests on every protocol pass through them. Only the health check and A2A agent card are open,
because monitoring and agent discovery happen before anyone signs in.

### Authentication: who is this?

The agent does not validate the token itself. It asks the identity provider, through the OIDC userinfo endpoint, who
the token belongs to. This is a trade-off:

- **The provider validates the token.** The agent needs a usable discovery document and userinfo endpoint, rather
  than its own signing-key and audience validation. It tolerates specific missing discovery fields, including those
  seen with AWS Cognito; this is not a guarantee of compatibility with every provider configuration.
- **Userinfo responses are not cached.** Each request asks the provider again. A revoked token stops working when
  that provider rejects it at userinfo; revocation behavior depends on the provider. The discovery document itself
  is cached by the authentication object.
- **Each request costs a round trip to the identity provider.** That adds latency, and the provider has to handle
  the load. Validating the JWT locally would be faster, but it would trust a token until it expires.

Its latency and availability therefore contribute to every authenticated agent request, even one that makes no model call.

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
message. If OPA decision logging is enabled without masking that input, its logs can contain prompts.

## Outbound: who the agent acts as

The agent calls the MCP server of Orchestrator Core for searching, reading subscriptions and starting workflows.
It also makes HTTP requests to fetch form-field options and, on AG-UI, download exports. These use the same
token selection as the MCP client. Orchestrator Core runs its own authentication and authorization checks on
the requests it receives. So the question that matters
most for safety is not "may this user call the agent?" but "whose token does the agent send to Orchestrator Core?".
There are two answers.

- **The user's token.** The agent forwards the token of the request to Orchestrator Core for that run. Orchestrator
  Core sees the user and applies its policies to the MCP-backed operations. The available operations and permissions
  depend on Core's route policies; forwarding a token does not establish that they match the Orchestrator UI exactly.
- **A service account.** When there is no user token, the agent signs in as itself with the OAuth2
  client-credentials grant when outbound authentication is enabled. Orchestrator Core then sees the agent, not the person.

With outbound authentication disabled and no forwarded token, the call carries no bearer token. Core must allow
that unauthenticated request for it to succeed. The table below describes authenticated calls.

Which of the two applies depends on the protocol:

| Protocol | Where the agent finds the user's token | Orchestrator Core sees |
| --- | --- | --- |
| OpenAI chat completions (`/v1`) | `Authorization` header | the user |
| AG-UI | `Authorization` header | the user |
| A2A | `auth_token` in the message metadata | the user, if the token is there; otherwise the service account |
| The agent's MCP (`/mcp`) | not forwarded | the service account when enabled |

Two rows stand out. On A2A, the HTTP header only gets the caller past the agent's own check. The token that reaches
Orchestrator Core travels inside the A2A message, so a planner that sends only the header makes the agent act as its
service account when that fallback is enabled. On the agent's MCP endpoint the user's token is not passed on at
all: the caller is checked inbound, but Core sees the service account, or an unauthenticated request when outbound
authentication is disabled.

Form-option URLs can point outside Core. Relative URLs use `WFO_CORE_API_URL`, or Core's MCP origin when it is
unset. Absolute URLs must use that base's host and port or an entry in `WFO_OPTIONS_HOSTS`. An allowed destination
receives the selected bearer token, so this list also determines where credentials may be sent. See
[Configuration](../reference/configuration.md#connection).

<a id="why-the-users-token-is-better-than-a-service-account"></a>

### User identity and service-account permissions

A shared service account exposes the same Core permissions to every caller using it. Giving it the union of all
callers' rights broadens what each caller can ask it to do, and Core sees the agent instead of the person. It is simpler
to set up and works for clients that have no user token, such as scheduled jobs, but it moves the decision about what a
user may do out of Orchestrator Core and into whoever picks the service account's rights.

Forwarding the user's token has its own limit. It only works when the agent's userinfo check and Core's authentication
and authorization accept the same token. When separate token audiences or scopes are required,
[token exchange (RFC 8693)](https://datatracker.ietf.org/doc/html/rfc8693) is the standard answer: it swaps the
user's token for one meant for the next hop; the identity carried and rights granted depend on the exchange policy.
The agent does not do this itself; an identity provider or a gateway in front of it can.

## What the agent does not do: gateways

The agent answers two questions about a caller: who they are, and whether they may use it. It does not ask how often
or how much: there are no rate limits, no budgets per user or per agent, and no filtering of which tools a caller can
see. It also only guards its own front door. Clients can reach the MCP server of Orchestrator Core and other agents
directly, without passing the agent.

These are questions for a layer in front of all of them, not for each agent on its own. The
[example orchestrator](https://github.com/workfloworchestrator/example-orchestrator) shows the minimal version: a small
proxy that lets LibreChat talk to agents over A2A, but accepts any API key. That is fine on a laptop and a liability
anywhere else. See [Gateways](https://workfloworchestrator.org/learn/orchestrator-agent/production/#gateways) for the wider deployment controls.

Conversation storage is a separate boundary. A2A and chat completions load snapshots by a caller-supplied conversation
identifier without checking the stored conversation's owner. Their trusted clients must isolate identifiers between
users. AG-UI binds a new thread to the authenticated subject and checks that subject and the thread's protocol before
continuing it; its chats API applies the same ownership checks. With authentication off, AG-UI threads have no owner
and are shared between callers. These AG-UI checks do not add ownership checks to the other protocols; see
[Conversation persistence](../reference/runtime.md#conversation-persistence).
