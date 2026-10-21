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

"""The conversations of the AG-UI route (the chat popup), for the person they belong to.

A thread belongs to the subject that created it, taken from the bearer the middleware verified; with OAuth2
off there is no subject and every thread is unowned, which is a development setting. Only this route's
threads are listed, read or deleted here: kagent's and LibreChat's conversations are theirs.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from orchestrator.core.db import db
from pydantic import BaseModel

from orchestrator_agent.adapters.ag_ui.chats import Chat, ChatSummary, pending_stop, title_of, transcript
from orchestrator_agent.adapters.ag_ui.worker import AGUI_AGENT_TYPES
from orchestrator_agent.api.dependencies import bearer_token
from orchestrator_agent.form_fill.pending import pending_of
from orchestrator_agent.form_fill.skill import FormFillSkill
from orchestrator_agent.mcp_client import bind_outbound_token
from orchestrator_agent.persistence import StoredThread, delete_thread, list_threads, load_thread
from orchestrator_agent.security import subject_of

router = APIRouter()


def _summary(thread: StoredThread) -> ChatSummary:
    return ChatSummary(
        thread_id=thread.thread_id,
        title=title_of(thread.state),
        updated_at=thread.updated_at,
        form_open=pending_stop(thread.state) is not None,
    )


def _mine(request: Request, thread_id: str) -> StoredThread:
    """The thread, when it is this route's and the caller's (an unowned one counts as theirs)."""
    thread = load_thread(db.session, thread_id)
    if thread is None or thread.agent_type not in AGUI_AGENT_TYPES:
        raise HTTPException(status_code=404, detail="No such chat")
    if thread.state.owner is not None and thread.state.owner != subject_of(request):
        raise HTTPException(status_code=403, detail="Not your chat")
    return thread


@router.get("/")
async def list_chats(request: Request, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[ChatSummary]:
    """The caller's chats, newest first."""
    threads = list_threads(db.session, agent_types=AGUI_AGENT_TYPES, owner=subject_of(request), limit=limit)
    return [_summary(thread) for thread in threads]


@router.get("/{thread_id}")
async def get_chat(request: Request, thread_id: str, last: Annotated[int, Query(ge=1, le=500)] = 50) -> Chat:
    """One chat: its last ``last`` messages and the form stop it waits on, if any."""
    thread = _mine(request, thread_id)
    messages = [
        m.model_dump(mode="json", by_alias=True, exclude_none=True) for m in transcript(thread.state, last=last)
    ]
    return Chat(**_summary(thread).model_dump(), messages=messages, pending=pending_stop(thread.state))


class FieldOption(BaseModel):
    """One option of a searchable question: the value to answer with, and how the person sees it."""

    value: str | int
    label: str


@router.get("/{thread_id}/options/{field}")
async def search_options(
    request: Request,
    thread_id: str,
    field: str,
    q: Annotated[str, Query(max_length=200)] = "",
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
) -> list[FieldOption]:
    """The options of a searchable question of the chat's pending form stop that ``q`` fits.

    For a client that lets the person search a long list instead of typing a name blind: the pick is
    answered with the option's value. The options come from the field's widget, as the person asking.
    """
    thread = _mine(request, thread_id)
    session = thread.state.form_fill
    pending = pending_of(session)
    searched = pending is not None and any(
        question.name == field and question.searchable for question in pending.questions
    )
    if session is None or not searched:
        raise HTTPException(status_code=404, detail="No question is being searched by that name")
    skill: FormFillSkill = request.app.state.form_fill
    with bind_outbound_token(bearer_token(request)):
        options = await skill.search_options(session, request.app.state.core_toolset.direct_call_tool, field, q, limit)
    if options is None:
        raise HTTPException(status_code=404, detail="No question is being searched by that name")
    return [FieldOption(value=option.value, label=option.label) for option in options]


@router.delete("/{thread_id}", status_code=204)
async def forget_chat(request: Request, thread_id: str) -> None:
    """Forget a chat: its messages, its state and its open form."""
    _mine(request, thread_id)
    delete_thread(db.session, thread_id)
