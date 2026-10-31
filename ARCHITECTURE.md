# OpenPoke architecture notes (as found, before changes)

Read-through of `server/` to locate exactly where agent overload originates.
Line references are to the upstream commit `5b5f635`.

## Request path for one user message

```
POST /chat  (routes/chat.py)
  └─ InteractionAgentRuntime.execute(user_message)         agents/interaction_agent/runtime.py:65
       ├─ transcript = working_memory_log (summarised) or raw conversation_log
       ├─ messages = prepare_message_with_history(...)      agents/interaction_agent/agent.py:20
       │     ├─ <conversation_history> … </conversation_history>
       │     ├─ <active_agents>  ← _render_active_agents()  agent.py:45   ★ OVERLOAD #1
       │     └─ <new_user_message> … </new_user_message>
       └─ loop ≤ 8×: request_chat_completion(model=interaction_agent_model, tools=TOOL_SCHEMAS)
            └─ tool send_message_to_agent(agent_name, instructions)   interaction_agent/tools.py:112
                 ├─ roster.add_agent(name) if new                      services/execution/roster.py
                 ├─ log_store.record_request(name, instructions)
                 └─ ExecutionBatchManager.execute_agent(...)           execution_agent/batch_manager.py:47
                      └─ ExecutionAgentRuntime(name).execute(instr)    execution_agent/runtime.py:43
                           ├─ system = ExecutionAgent.build_system_prompt_with_history()   agent.py:63  ★ OVERLOAD #2
                           │     └─ log_store.load_transcript(name)  → FULL history, conversation_limit=None
                           ├─ loop ≤ 8×: request_chat_completion(model=execution_agent_model, tools=gmail+triggers)
                           └─ record_response → batch drains → InteractionAgentRuntime.handle_agent_message(payload)
```

## The two overload problems, located

### #1 Roster overload (horizontal)

`services/execution/roster.py` is literally a JSON list of agent **names**. Every turn,
`_render_active_agents()` emits one `<agent name="…"/>` per entry, for every agent that has
ever existed. There is no description, no recency, no status, no pruning, no `remove`.
After a few days this is hundreds of lines in every Interaction Agent prompt, and the model
must guess an agent's purpose from its name alone.

Consequences:
- Prompt tokens grow O(#agents) on every user turn **and** every agent-result turn.
- Reuse decisions get worse as the list grows (name collisions: two "Email to Alice" agents).
- No signal distinguishes an agent touched 2 minutes ago from one touched 3 weeks ago.

### #2 Per-agent history overload (vertical)

`ExecutionAgent.build_system_prompt_with_history()` concatenates the agent's **entire** append-only
log (`<agent_request>`, `<agent_action>`, `<tool_response>`, `<agent_response>`) into the system
prompt. `conversation_limit` exists but is never set (`ExecutionAgentRuntime` constructs
`ExecutionAgent(agent_name)` with the default `None`), and even when set it only truncates by
request count — it does not compress.

The main conversation *does* get summarised (`services/conversation/summarization/`, threshold 100
messages, tail 10), so the design asymmetry is deliberate but unaddressed: execution agents keep
raw tool payloads (500 chars each) forever.

## Other things worth knowing

| Concern | Where | Note |
|---|---|---|
| Single LLM chokepoint | `openrouter_client/client.py:request_chat_completion` | Every call (5 call sites) goes through here → ideal place for routing + usage accounting + mocking. |
| Model selection | `config.py:54-58` | Five roles, all default `anthropic/claude-sonnet-4`. No cheap tier anywhere. |
| Triggers | `services/triggers/`, `services/trigger_scheduler.py` | SQLite; scheduler polls every 10s; wakes **the owning agent by name** via `ExecutionBatchManager`. A trigger therefore implies the agent must stay reachable. |
| Email watcher | `services/gmail/importance_watcher.py` + `importance_classifier.py` | Polls Gmail, runs an LLM classifier on **every** new email with the full Sonnet model. No pre-filter, no eval set. |
| Conversation summarisation | `services/conversation/summarization/` | Threshold 100 entries, tail 10, one LLM call. |
| Tests | — | None in the repository. |

## Call sites of `request_chat_completion`

1. `agents/interaction_agent/runtime.py` — interaction agent (strong model justified: user-facing).
2. `agents/execution_agent/runtime.py` — execution agent.
3. `agents/execution_agent/tasks/search_email/tool.py` — email search sub-task.
4. `services/conversation/summarization/summarizer.py` — conversation summariser.
5. `services/gmail/importance_classifier.py` — email importance classifier.

## Where the fixes attach (see README for design)

- **AgentRegistry** replaces `_render_active_agents()` with a budgeted, relevance-ranked roster and
  gives the Interaction Agent a `search_agents` fallback tool.
- **AgentMemory** replaces `load_transcript()` inside `build_system_prompt_with_history()` with a
  tiered structured memory + recent raw tail.
- **ModelRouter / usage ledger** wraps `request_chat_completion` behind a role-aware gateway.
- **Trigger ↔ registry coupling**: trigger wake-ups `touch()` the agent; agents with active
  triggers are never archived.
- **Email pre-filter** short-circuits the classifier for obvious bulk mail.
