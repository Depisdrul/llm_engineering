<!-- Generated copy. Source: notes/07-agents-and-deployment.md
     Edit the source file; this copy is overwritten on every extraction run. -->

# Week 8 — Agents, multi-agent systems, and deployment

**Covers Week 8 — the autonomous-agent capstone, the multi-agent pricing system, Modal deployment, and the Gradio front-end.** *(Exact lecture numbers are not recorded in the research file — don't rely on the numbering in these notes.)*

**How this was built:** written from primary sources — Anthropic and Cognition engineering posts, the MCP specification, LangGraph / OpenAI Agents SDK / CrewAI / vLLM / TGI / Modal docs, τ-bench, and the OpenTelemetry GenAI semantic-conventions repo. Course-repo facts come from direct inspection. Where the source material contradicts what a 2023–2025-era course teaches, that is flagged explicitly. Claims that could not be verified are marked `UNVERIFIED`.

> **Read the corrections table (§10) first.** Almost every MCP tutorial written before mid-2026 is wrong on at least three points, TGI is in maintenance mode, and the pre-1.0 LangChain agent APIs you will find in most tutorials no longer exist. This is the fastest-rotting material in the whole course.

---

## 0. The one-paragraph version

Three numbers should govern how you think about agents, and all three are uncomfortable. **First: tool-calling agents are much less reliable than leaderboards imply.** On τ-bench, *"even state-of-the-art function calling agents (like gpt-4o) succeed on <50% of the tasks"*, and **`pass^8` — the probability that all 8 independent attempts at the same task succeed — is under 25% in the retail domain** ([arXiv:2406.12045](https://arxiv.org/abs/2406.12045)). For anything customer-facing, `pass^k` is your SLA and `pass@k` is nearly meaningless. **Second: multi-agent systems mostly work by spending more tokens.** Anthropic's own research system beat single-agent Opus 4 by **+90.2%**, and on BrowseComp *"token usage by itself explains 80% of the variance"* in performance, at **~15× the token cost of chat** ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)). That is a real gain, honestly attributed to a real cost — not architectural magic. **Third: error compounding is arithmetic, not bad luck.** At 95% per-step reliability, a 20-step agent succeeds 36% of the time. So the entire discipline reduces to: *find the simplest solution possible, and only increase complexity when it demonstrably improves outcomes* ([Anthropic](https://www.anthropic.com/engineering/building-effective-agents)) — bound every loop, constrain every output, verify between steps, and put an approval gate in front of anything irreversible. On the practical side, Week 8 is the one week that needs **no GPU of your own at all**: the GPU is rented per-second from Modal, declared in code as `GPU = "T4"`.

---

## 1. Week 8 practicalities: no local GPU, and why that matters

**Course-repo fact:** the Week 8 code rents its GPU from Modal, declared inline. `GPU = "T4"` appears in **`pricer_service2.py`**, **`pricer_ephemeral.py`**, and **`llama.py`**.

This is the single biggest ergonomic difference from Week 7:

| Week | Compute | Where it runs | What you must set up |
|---|---|---|---|
| 6 | OpenAI's servers | hosted | API key — **but see `06-training-and-finetuning.md` §1 first; the fine-tuning API is being wound down** |
| 7 | **CUDA GPU required** | **Colab** — the repo's `pyproject.toml`/`uv.lock` contain no `peft`, `trl`, `bitsandbytes`, or `accelerate`, so QLoRA is deliberately not provisioned locally | GPU runtime + `pip install` in the notebook |
| **8** | **T4, rented per-second** | **Modal** | Modal account + token; **no local GPU, no CUDA, no Colab** |

**Why this design is worth internalising rather than just following:**

- **The Week 7 → Week 8 transition is the real production pattern.** You fine-tune on borrowed compute in a notebook, then serve on per-second serverless compute called from ordinary Python. The awkward middle step everybody skips — "how do I actually get this model behind an endpoint" — is what Week 8 is teaching.
- **T4 is a deliberate cost choice, not a capability choice.** Modal lists **T4 at $0.000164/s ($0.59/hr)** — the cheapest GPU on the [pricing page](https://modal.com/pricing) — against **H100 SXM5 at $0.001097/s ($3.95/hr)**. A T4 has 16 GB and is pre-Ampere, so it will serve a 4-bit or 8-bit small model and will not serve a large bf16 one. If the course code seems to be jumping through hoops on model size, that is why.
- **Idle costs nothing** (per-second billing), so leaving a Modal app deployed between monthly study blocks is cheap — but check `scaledown_window` and `min_containers` (§8.5), because `min_containers ≥ 1` means you *are* paying for a warm floor.
- Free credits: **$30/month Starter, $100/month Team.**

> **The two Week 8 files with different lifecycle semantics are the lesson.** `pricer_service2.py` (a deployed service) and `pricer_ephemeral.py` (a one-shot run) are the same model on the same GPU with completely different cost and cold-start profiles. That distinction — persistent endpoint vs ephemeral job — is the deployment decision, and it recurs in §8.6.

---

## 2. What "agent" means — the Anthropic framing

Primary source: Anthropic, [**"Building effective agents"**](https://www.anthropic.com/engineering/building-effective-agents)

### 2.1 The definition and the one distinction that matters

Anthropic uses **"agentic systems"** as the umbrella and draws a single architectural line:

- **Workflows** — *"LLMs and tools orchestrated through predefined code paths."* Predictable; for well-defined tasks.
- **Agents** — *"systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks."*

**The line is: who decides the control flow — your code, or the model?**

That is the whole taxonomy. Everything else is a pattern within it. It is also the right question to ask of any vendor pitch: *is the control flow in your code or in the model?* Because that determines your debuggability, your cost variance, and your blast radius.

### 2.2 The building block

**The augmented LLM** = model + retrieval + tools + memory, where the model itself *"generate[s] their own search queries, select[s] appropriate tools, and determin[es] what information to retain."*

Every pattern below composes this one block. If you only build one thing, build this.

### 2.3 The five workflow patterns

| Pattern | Mechanism | Use when |
|---|---|---|
| **Prompt chaining** | Decompose into sequential steps, each consuming the last output, with **programmatic gates** between steps to check trajectory | The task *"can be easily and cleanly decomposed into fixed subtasks"* |
| **Routing** | Classify the input, dispatch to a specialised downstream path | *"distinct categories are better handled separately."* Also **the main cost lever** — route easy traffic to a small model |
| **Parallelization** | **Sectioning**: split into independent concurrent subtasks. **Voting**: run the *same* task multiple times | *"multiple perspectives or attempts are needed for higher confidence results"* |
| **Orchestrator–workers** | A central LLM *"dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results"* | *"complex tasks where you can't predict the subtasks needed."* Distinguished from parallelization by subtasks being **determined at runtime** |
| **Evaluator–optimizer** | Generator LLM + critic LLM in a refinement loop | *"when we have clear evaluation criteria, and when iterative refinement provides measurable value"* |

**The gates in prompt chaining are the under-taught part.** A chain without programmatic checks between steps is just a longer prompt with more places to go wrong. The gate is what converts a chain from a liability into a controlled pipeline.

### 2.4 Autonomous agents

*"Typically just LLMs using tools based on environmental feedback in a loop."* They operate on open-ended problems where *"it's difficult or impossible to predict the required number of steps."* Suited to trusted environments over many iterations.

Costs, stated by Anthropic: agentic systems *"trade latency and cost for better task performance"*, plus **higher cost and the potential for compounding errors** — hence sandboxed testing and guardrails.

### 2.5 When NOT to

> ***"Find the simplest solution possible, and only increas[e] complexity when needed."*** For many applications, optimising a **single LLM call with retrieval and in-context examples is enough.** Add complexity only when it *"demonstrably improves outcomes."*

This is the most-quoted and least-followed sentence in the field. It is also the one that would have saved most of the failed enterprise pilots.

### 2.6 The three implementation principles

1. **Simplicity** of design.
2. **Transparency** — show the agent's planning steps explicitly.
3. **Well-crafted, well-documented, well-tested tool interfaces** (the "agent–computer interface", ACI) — **tool definitions materially determine agent performance.**

Principle 3 is the highest-leverage and most neglected. See §6.3.

---

## 3. The named techniques, mapped onto that framing

- **Tool use** — the substrate. Everything else is scaffolding around tool calls.
- **ReAct** (Yao et al., arXiv 2210.03629 — **`UNVERIFIED`, not fetched**): interleave Thought → Action → Observation. Historically implemented as **text parsing of "Thought:/Action:" blocks.**
  > **Flag as outdated: hand-rolled ReAct text parsing is obsolete.** Native tool-calling APIs plus reasoning models subsume it entirely. ReAct survives as a *conceptual description of the loop*, not as an implementation recipe. If a tutorial has you regex-parsing "Action:" out of a completion, it is from 2023.
- **Planning** — either an explicit plan-then-execute stage (which is a chained workflow) or emergent from the loop. **Explicit plans are more debuggable and more likely to be stale.** Pick which failure you prefer.
- **Reflection** — Anthropic's **evaluator–optimizer**. Needs an *external or objective* signal to be worth anything. **Pure self-critique without ground truth often just adds tokens and confident rewrites.**
- **Memory** — distinguish four things, because "we need memory" almost always means one specific one:
  1. **Context window / working memory** for this turn.
  2. **Thread/session state** — LangGraph checkpointers, Agents SDK Sessions.
  3. **Long-term store** — vector/document/graph, retrieved on demand.
  4. **Compaction/compression** of history when the window fills.

  **Most production "memory" is (2) + (4) plus retrieval, not a fancy memory architecture.** Building (3) before you need it is a common waste.

### 3.1 Single-agent vs multi-agent — the decisive variable

- **Single-agent** = one context, one message history, one decision-maker. Sub-capabilities are **tools**.
- **Multi-agent** = multiple LLM contexts with distinct instructions, communicating via messages, handoffs, or task descriptions.

> **The decisive variable is context. A subagent is a separate context window.** That is simultaneously the benefit — parallel exploration without polluting the main context, more total tokens of work — **and the cost**: the subagent does not know what the others decided.

Everything in §4 follows from that one sentence.

---

## 4. Multi-agent systems: when they help, and the documented failure modes

Two primary posts disagree productively. **Teach both — the disagreement is the lesson.**

### 4.1 The pro case, with numbers

Anthropic, [**"How we built our multi-agent research system"**](https://www.anthropic.com/engineering/multi-agent-research-system)

- **Architecture:** lead agent analyses the query, sets strategy, **spawns specialised subagents to explore aspects in parallel**; each subagent searches independently and returns findings; the lead synthesises and decides whether to iterate.
- **+90.2%** over single-agent Claude Opus 4 on their internal research eval (lead Opus 4 + Sonnet 4 subagents).
- On BrowseComp, *"token usage by itself explains 80% of the variance"* in performance; model choice and number of tool calls account for most of the rest.
- **Cost: multi-agent uses ~15× more tokens than chat; single agents ~4× more than chat.**

> **This is the honest mechanism, and Anthropic states it plainly: multi-agent works largely by spending more tokens in parallel across more context windows.** Which means the first question to ask of any multi-agent proposal is: *could a single agent with a bigger token budget get most of this?* Often yes.

**Works when:** breadth-first queries with multiple independent directions; task value high enough to justify the token cost; information exceeds a single context window; many complex tools to select among.

**Fails when:** all agents need the *same* context; heavy dependencies between agent tasks; **coding** (few genuinely parallelisable parts); real-time coordination needed.

**Documented failure modes, from Anthropic's own post:** spawning **50+ subagents for simple queries**; endless searching for nonexistent sources; **duplicated work from vague task descriptions**; preferring SEO content farms to authoritative sources; *"minor changes cascade into large behavioral changes"*; non-determinism making debugging hard; state management across long-running processes requiring **durable execution**; synchronous execution bottlenecking information flow; deployment complexity from continuously-running stateful systems. **Human evaluation was essential** for catching hallucinations and source-selection bias that automation missed.

### 4.2 The against case

Cognition, [**"Don't Build Multi-Agents"**](https://cognition.com/blog/dont-build-multi-agents)

Two principles:

1. ***"Share context, and share full agent traces, not just individual messages."***
2. ***"Actions carry implicit decisions, and conflicting decisions carry bad results."***

**The Flappy Bird example** is the whole argument in one story: a main agent splits work between two subagents; subagent 1 builds a Super-Mario-style background while subagent 2 builds an incompatible bird sprite. **Sharing only the *original task* does not fix it** — without seeing each other's work, they make clashing unstated decisions.

Recommended alternatives: a **single-threaded linear agent** with continuous context; and when context overflows, a **dedicated compression model** that summarises history and actions into key details.

**Cited practice:** **Claude Code deliberately restricts subagents to answering questions rather than doing parallel coding work**, precisely to avoid conflicting decisions. That is a strong signal: the team with the most production agent experience chose the conservative architecture for their own flagship product.

### 4.3 Reconciliation — the actual guidance

> **Multi-agent helps when subtasks are read-only, independent, and easily verifiable** — search, retrieval, breadth-first exploration, independent evaluation. **It hurts when subtasks write to shared state or make interdependent design decisions** — coding, document authoring, anything where consistency across outputs matters.

That single rule reconciles both posts and is the thing to carry into an architecture review.

### 4.4 The failure modes as a checklist

| Failure mode | Mechanism | Mitigation |
|---|---|---|
| **Error compounding** | Per-step reliability `p` over `n` steps → `p^n`. **At p=0.95, 20 steps → 36% success.** | Reduce step count; verify/gate between steps; make steps idempotent and retryable |
| **Context loss between agents** | The subagent has a fresh window; the parent's reasoning and other subagents' decisions are invisible | Pass **full traces, not summaries**; fix shared decisions *in the orchestrator prompt* before spawning; or don't split |
| **Conflicting implicit decisions** | Two agents independently resolve the same ambiguity differently | Have the orchestrator **make and state** the decisions; **single writer** for shared artifacts |
| **Cost/token blowup** | ~15× chat baseline; subagent-spawn loops | **Hard caps** on subagent count, depth, tool calls, tokens, wall-clock; budget per request |
| **Coordination overhead** | Synchronous joins mean the slowest subagent gates everything | Async/streaming results; timeouts with partial results |
| **Duplicated work** | Vague task descriptions | Explicit, disjoint, **machine-checkable** subtask specs |
| **Non-determinism / undebuggability** | Small prompt changes cascade | Tracing on every run; recorded/replayable traces; regression eval suite |
| **State loss on crash** | Long-running processes | Durable execution / checkpointing |

> **Memorise the error-compounding arithmetic.** `0.95^20 ≈ 0.36`. It explains, without any hand-waving, why long-horizon agents fail, why frontier models are worth their premium in agentic settings, and why "reduce the number of steps" is a more effective intervention than almost any prompt tweak.

**Flag as outdated / overclaimed:** "multi-agent systems outperform single agents." The defensible version: *on breadth-first read-only research tasks, orchestrator–subagent architectures buy large gains, and roughly 80% of that gain is explained by spending more tokens across more context windows; on tasks requiring consistent shared decisions they reliably underperform a single well-contexted agent.*

---

## 5. Frameworks in 2026, and what MCP standardises

### 5.1 LangGraph

[Docs](https://docs.langchain.com/oss/python/langgraph/overview)

Positioned as *"a low-level orchestration framework and runtime for building, managing, and deploying long-running, stateful agents."*

- **Core:** `StateGraph`, nodes, edges (with `START`/`END`), typed shared state with reducers (`MessagesState`), conditional edges, `Command` for control flow.
- **Differentiators:** **persistence via checkpointers** (thread-scoped, resumable state), **durable execution**, **human-in-the-loop interrupts**, streaming, time travel / state replay.
- **Explicit positioning statement in the docs:** *"If you are just getting started with agents or want a higher-level abstraction, we recommend you use LangChain's agents that provide prebuilt architectures for common LLM and tool-calling loops."* Choose LangGraph when you need *"fine-grained control to mix deterministic, hand-coded steps with LLM-driven agentic steps."*

> **Flag as outdated:** "LangGraph is part of LangChain / you need LangChain chains to use it," and **any pre-1.0 LangChain agent API** — `initialize_agent`, `AgentExecutor`, `LLMChain`. The 2026 split is: **LangChain v1 = prebuilt agent; LangGraph = the low-level runtime underneath it.** Docs also moved to `docs.langchain.com/oss/...`; old `langchain-ai.github.io/langgraph` URLs redirect.

**Human-in-the-loop specifics** ([docs](https://docs.langchain.com/oss/python/langgraph/add-human-in-the-loop)) — the most exam-worthy details in the whole framework section:

- `interrupt(payload)` pauses the graph and surfaces the payload; `Command(resume=value)` resumes, and `interrupt()` then *returns* `value` instead of pausing.
- **A checkpointer is mandatory:** *"A checkpointer is required to persist graph state. In production, this should be durable (e.g., backed by a database)."*
- ⚠️ **The critical caveat: on resume, the interrupted node re-executes from its beginning**, not from the interrupt point. Therefore *"interrupts are typically best placed at the start of a node or in a dedicated node,"* and **side effects (API calls, DB writes) must be placed *after* the interrupt** — or they run twice.
- Patterns: approve/reject (route with `Command(goto=...)`), edit state, review/edit tool-call arguments before execution, validate input via a loop with multiple `interrupt()` calls.
- **Multiple interrupts are matched by index** — do not conditionally add, remove, or reorder them, or resume values misalign. Parallel nodes can interrupt simultaneously; resume all at once with a dict keyed by interrupt ID.

> **The re-execution rule is the one that will bite you in production.** A node that charges a card and *then* interrupts for approval will charge the card twice. This is not a LangGraph quirk to complain about — it is the inevitable consequence of durable replay, and every durable-execution system has the same property.

### 5.2 OpenAI Agents SDK

[Docs](https://openai.github.io/openai-agents-python/)

- **Primitives:** **Agents** (LLM + instructions + tools), **Handoffs** (agent delegates to another agent), **Guardrails** (input/output validation), **Sessions** (memory), **Tracing**.
- **Built-in agent loop** — the runner manages turns and tool dispatch until completion. No manual orchestration.
- **Guardrails run *in parallel* with agent execution** and use a **tripwire** to fail fast before bad output propagates. Both input and output guardrails.
- **Sessions** — persistent conversation memory across turns, with SQLite / SQLAlchemy / encrypted implementations.
- **Tracing built in**, integrated with OpenAI's evals / fine-tuning / distillation tooling.
- Lineage: *"a production-ready upgrade of our previous experimentation for agents, Swarm."* Wraps the **Responses API** by default.
- Character: **lighter-weight and more opinionated than LangGraph** — handoffs instead of an explicit graph, and much less machinery for durability and checkpointing.

**Flag as outdated:** **Swarm** (educational, deprecated), **Assistants API**-based agent tutorials, and Chat-Completions-only agent scaffolding.

### 5.3 CrewAI

[Docs](https://docs.crewai.com/en/introduction)

*"The leading open-source framework for orchestrating autonomous AI agents and building complex workflows."*

The 2026 architecture is **two complementary abstractions**:

- **Flows** = the "manager" / process definition — *"defines the steps, the logic, and how data moves through your system"*; event-driven, handles state management and control flow.
- **Crews** = the "teams" that *"do the heavy lifting"* — role-playing agents that collaborate autonomously on a delegated task.

**The docs' own recommendation: "Use both."** Start with a Flow for structure, and drop Crews into Flow steps where autonomous problem-solving is genuinely needed. Example shapes: Flow + Python tasks (simple automation); Flow manages state → Crew researches; Flow handles request → Crew generates → Flow persists.

Core concepts: Agents (role/goal/backstory), Tasks, Tools, sequential vs hierarchical process, memory.

**Flag as outdated:** "CrewAI is a LangChain wrapper" — the intro docs make no LangChain claim and CrewAI has been standalone for some time (**`UNVERIFIED`** as to the exact version where the dependency was dropped). Also outdated: describing CrewAI as **Crews-only**; **Flows are now the recommended backbone.**

Character: **the highest-level and most opinionated of the three.** The role-play metaphor gets a demo running fastest and gives you the least control over exactly the failure modes in §4.4. That trade is fine for a prototype and dangerous for production.

### 5.4 MCP (Model Context Protocol)

[Intro](https://modelcontextprotocol.io/docs/getting-started/intro) · [Architecture](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture) · **current spec version `2026-07-28`**

**What it is:** an open standard for connecting AI applications to external systems — the docs' analogy is *"like a USB-C port for AI applications."* Scope note: *"MCP focuses solely on the protocol for context exchange—it does not dictate how AI applications use LLMs or manage the provided context."*

**What it standardises** — the answer to "what does MCP actually give you":

1. **A wire protocol** — JSON-RPC 2.0 messages: requests, responses, notifications.
2. **Tool description and invocation** — `tools/list` returns `name`, `title`, `description`, `inputSchema` (JSON Schema); `tools/call` invokes with `arguments`; results are a typed `content` array (text, images, resources).
3. **A primitive taxonomy** (below).
4. **Discovery, versioning and capability negotiation** — the mandatory `server/discover` request; per-request protocol version + capabilities.
5. **Transports** — stdio and Streamable HTTP, with the same message format over both.
6. **Authorization** — standard HTTP auth (bearer tokens, API keys, custom headers); **OAuth recommended** for obtaining tokens.
7. **Change notifications, pagination, caching, progress.**

**What it does NOT standardise:** agent architecture, prompting, context management, orchestration, evaluation. Those are still your problem.

**Participants:** a **Host** (the AI app — Claude Code, VS Code) manages one or more **Clients**; each Client holds a dedicated connection to one **Server**. Local servers (stdio) typically serve one client; remote servers (Streamable HTTP) serve many.

**Layers:** the **data layer** (JSON-RPC protocol, primitives, notifications) is inner; the **transport layer** (channels, framing, auth) is outer.

**Server primitives:** **Tools** (executable functions), **Resources** (contextual data), **Prompts** (reusable templates / few-shot). Each has `*/list`, `*/get`, and for tools `tools/call`.

**Client primitives:** **Elicitation** (`elicitation/create`) — the server asks the user for input or confirmation, delivered via the Multi-Round-Trip-Requests pattern. *(This is the protocol-level human-in-the-loop primitive — see §6.5.)*

**Extensions** beyond the core, e.g. the **Tasks extension** — servers return a durable handle for long-running requests, and clients poll for status and retrieve results later. Directly relevant to long-running agent tools.

> 🚩 **MAJOR OUTDATED-CLAIM CLUSTER — almost every MCP tutorial written before mid-2026 is wrong on these six points:**
>
> 1. **Sampling is DEPRECATED** as of `2026-07-28`. The docs say *"New implementations should integrate directly with LLM provider APIs."* Sampling — "the server borrows the client's model" — was the flagship feature in every 2025 explainer.
> 2. **Logging (as a client primitive) is DEPRECATED.** *"New implementations should log to `stderr` (stdio transport) or use OpenTelemetry."*
> 3. **MCP is now a *stateless* protocol.** *"Every request carries the protocol version and the capabilities relevant to that request in its `_meta` field, so the server can process each request on its own."* The old stateful `initialize` handshake framing is gone; discovery is via the mandatory **`server/discover`** request, which is itself **optional to call** and **cacheable** (`ttlMs`, `cacheScope`).
> 4. **Change notifications are opt-in** via a long-lived **`subscriptions/listen`** stream naming the notification types wanted; the server acknowledges with `notifications/subscriptions/acknowledged` reflecting the subset it will honour, and every notification carries `io.modelcontextprotocol/subscriptionId` in `_meta`. Notifications are explicitly **best-effort** — *"There are no guarantees that every notification will be sent or received, particularly across transport reconnects"* — **so clients should also poll.**
> 5. **Transports are stdio and Streamable HTTP.** The **HTTP+SSE transport of the original spec is gone.** SSE now appears only as an optional streaming mechanism *within* Streamable HTTP.
> 6. **"Roots"** did not appear as a current client primitive in the `2026-07-28` architecture doc — **`UNVERIFIED`** whether it was removed, deprecated, or merely not listed on that page. Check `/specification/2026-07-28/` before teaching it.

New surface worth knowing: **MCP Apps** (interactive apps running inside AI clients) and **progressive tool discovery** for clients federating many servers.

### 5.5 How they differ, in one table

| | LangGraph | OpenAI Agents SDK | CrewAI | MCP |
|---|---|---|---|---|
| **Kind** | low-level graph runtime | lightweight agent runtime | high-level multi-agent framework | **protocol, not a framework** |
| **Control flow** | explicit graph you author | built-in loop + handoffs | Flows (structure) + Crews (autonomy) | n/a |
| **State/persistence** | checkpointers, threads, durable execution, time travel | Sessions (SQLite/SQLAlchemy/encrypted) | Flow state | n/a (Tasks extension for long-running calls) |
| **HITL** | first-class `interrupt` / `Command(resume=)` | via guardrails/tooling | via Flow steps | **Elicitation** |
| **Guardrails** | you build them as nodes/edges | first-class, parallel, tripwires | agent/task-level | n/a |
| **Observability** | LangSmith-integrated | built-in tracing | callbacks/integrations | OTel conventions exist for MCP |
| **Vendor coupling** | model-agnostic | OpenAI-first (Responses API) | model-agnostic | vendor-neutral standard |
| **Best for** | complex, long-running, auditable, durable workflows | fast production agents in the OpenAI stack | quick multi-agent prototypes, role-decomposed work | exposing/consuming tools across any client |

> **They are not alternatives to MCP.** All three frameworks can *consume* MCP servers. The correct mental model: **frameworks = orchestration; MCP = the integration layer.** Anyone presenting "LangGraph vs MCP" as a choice has misunderstood the stack.

**Framing for a Gewiss context:** MCP is the piece with the strongest strategic argument, because it is vendor-neutral and because it decouples "we exposed our PLM/ERP/catalogue as tools" from "we picked a framework and a model this quarter." The framework you pick in 2026 will probably be replaced; an MCP server over your own systems will not be.

---

## 6. Structured output, tool-calling reliability, guardrails, HITL

### 6.1 Structured Outputs

[OpenAI docs](https://developers.openai.com/api/docs/guides/structured-outputs)

- **The guarantee:** *"the model will always generate responses that adhere to your supplied JSON Schema"* — no omitted required keys, no hallucinated enum values. Sells **type-safety, explicit refusals, simpler prompting.**
- **vs JSON mode:** both produce valid JSON; **only Structured Outputs guarantees schema adherence.** JSON mode validates syntax only. **Flag as outdated:** `response_format={"type":"json_object"}` plus "please output JSON" prompting.
- **vs function calling:** the docs' own split — *"If you are connecting the model to tools, functions, data, etc. in your system, then you should use function calling"*; use a structured `response_format` when shaping the model's reply to the user.
- **Strict mode supports only a subset of JSON Schema** — *"some features are unavailable either for performance or technical reasons."* The docs fetched did not enumerate the unsupported keywords. **`UNVERIFIED`** — the practical constraints to check on the live page: `additionalProperties: false` required on every object; all properties required (optionality expressed via nullable unions); limits on nesting depth / total properties / enum size; and unsupported validation keywords (`minLength`, `pattern`, `format` — verify current status).
- **Refusals are programmatically detectable** via a `refusal` field. **You must branch on it, not just parse.**
- Responses can still be **incomplete** due to token limits — **always check the finish reason before parsing.**
- Latency note: for fine-tuned models, *"the first request you make with any schema will have additional latency as our API processes the schema, but subsequent requests with the same schema will not."* ⇒ **keep schemas stable and few; do not generate schemas per-request.**

### 6.2 Tool-calling reliability — the honest numbers

τ-bench (Yao et al.) — [arXiv:2406.12045](https://arxiv.org/abs/2406.12045) · [τ²-bench](https://github.com/sierra-research/tau2-bench)

- **Design:** an LLM plays the **user**; the agent has **domain API tools plus policy guidelines** (retail, airline). Evaluated by comparing the **final database state to an annotated goal state** — objective, not judge-based. *This design choice is why the numbers are believable.*
- ***"Even state-of-the-art function calling agents (like gpt-4o) succeed on <50% of the tasks."***
- **`pass^k`** — the probability that **all k** independent trials of the same task succeed. Result: **`pass^8 < 25%` in retail.**
- Authors' conclusion: the field needs methods that *"improve the ability of agents to act consistently and follow rules reliably."*

> **`pass^k` is the single most important eval concept for production agents**, and it is the exact inverse of `pass@k` (which rewards *any* success in k tries). For a customer-facing agent, **`pass@k` is nearly meaningless and `pass^k` is the SLA.** Headline leaderboard scores are single-trial — treat them as upper bounds.

Leaderboards, secondary: [Steel](https://leaderboard.steel.dev/leaderboards/tau-bench/), [HAL](https://hal.cs.princeton.edu/taubench_retail).

### 6.3 Reliability techniques, in order of leverage

1. **Constrain the output space** — Structured Outputs, strict function schemas, grammar-constrained decoding (vLLM structured outputs, TGI "Guidance", llama.cpp GBNF). **Turns a probabilistic parse into a guaranteed one.**
2. **Design the tool interface for the model** (Anthropic's ACI principle): few tools; unambiguous names; descriptions that state *when to use* and *when not to*; examples in the description; flat argument schemas; no overlapping tools. **Most "the model can't use tools" problems are tool-design problems.**
3. **Validate then repair** — Pydantic / JSON-Schema validation on every tool result and final output; on failure, feed the validation error back for **one bounded retry.** Cap retries.
4. **Make tools idempotent and give them dry-run modes** — the only real defence against error compounding *and* against LangGraph's interrupt-replay semantics (§5.1).
5. **Return errors as tool results, not exceptions** — the model can often recover if it sees the error text. Raising kills the loop; returning lets it adapt.
6. **Bound everything:** max steps, max tool calls, max tokens, wall-clock timeout, max cost. **An unbounded loop is a production incident**, and Anthropic's own post documents agents spawning 50+ subagents for trivial queries.
7. **Sandbox side effects**; require explicit approval for irreversible ones.

### 6.4 Guardrails

| Layer | Checks |
|---|---|
| **Input** | relevance/topicality, PII detection, prompt-injection and jailbreak detection, schema validation of user input |
| **Output** | schema/type validation, policy and safety checks, groundedness/citation checks against retrieved sources, PII redaction, business-rule checks (e.g. "never quote a price below floor") |
| **Tool-level** | allowlists, argument range checks, rate limits, per-tool auth scopes |

The [Agents SDK](https://openai.github.io/openai-agents-python/) model: guardrails run **in parallel with the agent** and **trip a tripwire** to abort early. The design point is to **fail before an expensive or harmful action, not after.**

> **Prompt injection is the guardrail case that matters most for agents**, because **tool results are untrusted input that reaches the model.** Structural mitigations — least-privilege tool scopes, no secrets in context, human approval on irreversible actions, separating instruction and data channels — **beat classifier-based detection.** A classifier is a probabilistic filter in front of a deterministic capability; scoping removes the capability.

### 6.5 Human-in-the-loop checkpoints

- **Place approval gates before irreversible or externally-visible actions** — writes, payments, sends, deletes, deploys.
- **LangGraph is the reference implementation:** `interrupt()` + `Command(resume=)` + a **durable checkpointer**, with the **side-effects-after-the-interrupt** rule because the node re-runs from its start (§5.1).
- **MCP's Elicitation primitive** is the protocol-level equivalent — a server can request user input or confirmation mid-call.
- **Design notes that determine whether HITL survives contact with users:**
  - Make the approval payload contain **exactly what a human needs to decide** — the diff, the amount, the recipient — **not a raw state dump.**
  - Log approver identity and decision.
  - Support timeouts with a safe default.
  - **Batch approvals to avoid alert fatigue, which is what actually kills HITL in production.** An approval gate that fires 200 times a day is a rubber stamp within a week.
- Anthropic's multi-agent post: **human evaluation was essential** for catching hallucinations and source bias that automated eval missed.

---

## 7. Agent evaluation and observability

### 7.1 Why agent eval is harder than single-turn eval

1. **Multi-step trajectories** — the final answer can be right for the wrong reasons, or wrong because of step 3 of 14. You must evaluate both **outcome** and **trajectory**.
2. **No single ground-truth path** — many valid tool sequences reach the same goal, so trajectory comparison cannot be exact-match.
3. **Compounding stochasticity** — variance multiplies across steps. **A single run tells you almost nothing**; hence `pass^k`.
4. **Environment state** — correctness is a property of the *world* after the run (τ-bench compares final DB state), so you need a **resettable, seeded environment.** This is real infrastructure work and it is the reason most teams never build proper agent evals.
5. **Multi-turn user interaction** — needs a **simulated user**, which is itself a model with its own error and its own drift across versions.
6. **Cost and latency are first-class outcomes**, not footnotes. **A 15×-token multi-agent system that scores 3 points higher may be a bad trade.**
7. **Non-stationarity** — *"minor changes cascade into large behavioral changes"* (Anthropic), and the underlying model changes under you. **Regression suites are mandatory, not optional.**
8. **Partial credit / rubric design** — hard to define, and LLM judges introduce their own bias. **Judges must themselves be validated against human labels.**

### 7.2 What to measure

| Axis | Metrics |
|---|---|
| **Outcome** | task success (state-based where possible), **`pass^k` for k = 3–8**, single-trial pass rate |
| **Trajectory** | number of steps, tool-call validity rate, tool-call error rate, redundant/repeated calls, loop detection, adherence to required steps |
| **Policy/rule compliance** | did it follow the domain rules — **τ-bench's core insight: rule-following is a separate axis from capability** |
| **Cost** | tokens in/out per task, $/task, tool-call count |
| **Latency** | p50/p95/p99 end-to-end, time-to-first-token, per-step latency |
| **Safety** | guardrail trip rate, escalation/HITL rate, injection-resistance suite |
| **Groundedness** | citation/attribution accuracy for research agents |

### 7.3 Tooling and the standard

**The standard:** OpenTelemetry **GenAI semantic conventions**, which as of 2026 live in their **own repository** — [github.com/open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) — covering *"spans, metrics, and events for GenAI clients, MCP (Model Context Protocol), and provider-specific conventions (OpenAI, etc.)"*, with Anthropic / AWS Bedrock / Azure AI provider conventions and **agent spans**.

- The old `opentelemetry.io/docs/specs/semconv/gen-ai/` page is now a redirect/deprecation notice.
- **Stability status: not clearly declared.** The repo still has TODOs and 136 open issues — **treat the conventions as evolving.**
- **`UNVERIFIED`:** specific attribute names (`gen_ai.operation.name`, `gen_ai.agent.*`, `gen_ai.tool.*`) — read `model/*.yaml` in that repo before teaching attribute names.
- **Why it matters:** vendor-neutral tracing means you can switch observability backends and **correlate LLM spans with the rest of your service traces.** Prefer OTel-native tooling. For an enterprise like Gewiss that already has APM, this is the difference between agent traces being a silo and being part of the existing observability estate.

**Platforms** (all secondary sources; verify features before recommending): LangSmith (tightest LangGraph integration, trace→dataset→eval loop), Langfuse (open-source, self-hostable, OTel-compatible), Arize Phoenix (open-source, OTel/OpenInference-native), Braintrust (eval-first, prompt playground + CI), MLflow tracing, W&B Weave, Laminar. Comparisons: [marktechpost](https://www.marktechpost.com/2026/08/09/top-llm-observability-and-evaluation-platforms-in-2026-langfuse-langsmith-braintrust-arize-and-more-compared/), [MLflow](https://mlflow.org/top-5-agent-observability-tools/), [Laminar](https://laminar.sh/article/2026-04-23-top-6-agent-observability-platforms).

**Framework-native:** the OpenAI Agents SDK ships tracing wired into OpenAI evals; TGI emits OTel distributed traces and Prometheus metrics.

**Benchmarks:** τ-bench / τ²-bench (tool-agent-user, state-based), plus (secondary/orientation) BrowseComp for research agents and the SWE-bench family for coding agents.

### 7.4 The workflow to teach

Instrument with **OTel from day one** → collect production traces → curate failing traces into a **regression dataset** → define outcome + trajectory + cost + latency metrics → run the suite **with k ≥ 3 repeats** on every prompt/model/tool change → **gate deploys on it** → keep a **human review queue**, because Anthropic found human evaluation caught what automation missed.

> **The ordering is the point.** Instrumentation comes before evaluation, because your eval dataset should be made of real failures, not invented ones. Teams that write synthetic eval sets first end up measuring a distribution their users do not produce.

---

## 8. Deployment

### 8.1 vLLM

[Docs](https://docs.vllm.ai/en/latest/)

- **PagedAttention** — the KV cache is managed in fixed-size blocks like OS virtual memory, eliminating fragmentation and enabling sharing / copy-on-write across sequences. **This is what makes large batch sizes possible.**
- **Continuous (iteration-level) batching** — finished sequences leave the batch and new requests join every decode step, instead of waiting for the slowest member of a static batch. **The single biggest throughput lever in LLM serving.**
- **Automatic prefix caching** — reuses KV for shared prompt prefixes. A large win for long system prompts and for **agent loops that resend history**, which is exactly the Week 8 shape.
- Quantization backends listed: AutoAWQ, bitsandbytes, GGUF, GPTQModel, plus FP8 and INT4 paths.
- **Speculative decoding** — EAGLE, MLP-based, and n-gram draft strategies.
- **OpenAI-compatible HTTP server** — drop-in `/v1/chat/completions`, so client code and Gradio front-ends do not change.
- Distributed: tensor/pipeline parallel for multi-GPU and multi-node.
- **vLLM V1** engine referenced in the docs' user guide as a rearchitected core.
- Structured outputs / grammar-constrained decoding supported (guided JSON/regex/grammar) — the serving-side half of §6.3 item 1.

### 8.2 TGI — read this before following any 2023–2024 tutorial

[Docs](https://huggingface.co/docs/text-generation-inference/index)

> 🚩 **TGI is in MAINTENANCE MODE.** The docs state it accepts PRs only for minor bug fixes, documentation, and lightweight maintenance, and is **no longer in active development.** HuggingFace explicitly recommends going forward: **vLLM** (primary alternative), **SGLang**, and for local use **llama.cpp** and **MLX**.

**Flag as outdated:** any course material presenting TGI as the standard production server, or as vLLM's peer.

Teach it as **historical context**, which is genuinely interesting: TGI started the movement of optimised engines building on `transformers` architectures, and pioneered continuous batching, Flash/Paged Attention integration, SSE token streaming, "Guidance" structured output for function calling, speculative decoding (Medusa, ngram), OTel distributed tracing + Prometheus metrics, and unusually broad hardware coverage (NVIDIA, AMD, Intel Gaudi, AWS Trainium/Inferentia, Google TPU, Intel GPU).

### 8.3 Ollama / llama.cpp / LM Studio / MLX

- **Ollama** wraps llama.cpp with model pulling, a Modelfile, an OpenAI-compatible endpoint, and automatic GPU/CPU offload. Optimised for **single-user local development**, quantised GGUF weights, laptop-class hardware, and zero-config startup.
- **The consistent finding across 2026 comparisons: Ollama's throughput collapses under concurrency** — roughly single-digit concurrent users — because it is not built around continuous batching + paged KV the way vLLM is. Reported vLLM-vs-Ollama throughput gaps are large (one source claims ~9×). All secondary — **do not cite specific multipliers as fact:** [SitePoint](https://www.sitepoint.com/ollama-vs-vllm-performance-benchmark-2026/), [InsiderLLM](https://insiderllm.com/guides/llamacpp-vs-ollama-vs-vllm/), [The AI Engineer](https://theaiengineer.substack.com/p/vllm-vs-ollama-vs-sglang-vs-tensorrt).

> **The usage rule:** **Ollama / llama.cpp for local dev, prototyping, single-user, and air-gapped/CPU/Apple-Silicon deployment; vLLM (or SGLang / TensorRT-LLM) for anything multi-tenant. Do not ship Ollama as a multi-user service.** This is one of the most common and most expensive architecture mistakes in 2026, because Ollama works beautifully in the demo and dies at 10 users.

- **SGLang** is now routinely named alongside vLLM (RadixAttention prefix caching, strong structured-output support). **`UNVERIFIED`** — SGLang docs not fetched.

### 8.4 Serving memory arithmetic — the thing that actually determines your bill

```
weights_bytes  = n_params × bytes_per_param
                 (bf16 = 2, fp8/int8 = 1, 4-bit ≈ 0.5–0.6 incl. scales)

kv_per_token   = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
                 (the leading 2 = K and V; n_kv_heads < n_heads under GQA)
```

Worked example, **Llama-3-8B** (32 layers, 8 KV heads under GQA, head_dim 128), bf16 KV cache:

```
kv_per_token = 2 × 32 × 8 × 128 × 2 B = 131,072 B = 128 KiB / token
8k context   → 8192 × 128 KiB = 1.0 GiB per concurrent sequence
weights bf16 → 8e9 × 2 = 16 GB

⇒ on an 80 GB A100/H100 at ~90% utilization:
   72 GB − 16 GB (weights) − ~4 GB (activations/CUDA/framework overhead) ≈ 52 GB KV
   ≈ 52 concurrent 8k-token sequences   (or ~416 at 1k tokens each)
```

> **This arithmetic — not weight size — is what determines your concurrency and therefore your $/token.** Almost everyone sizes GPUs from the weights and is then surprised by concurrency. The levers: GQA/MQA (already in the architecture), **KV-cache quantization to fp8/int8 (halves the per-token cost)**, prefix caching (deduplicates shared system prompts), shorter max context, and paged allocation so you do not reserve max-length KV per request.

**These are derived figures from published architecture parameters. The arithmetic is standard but was not verified against a measured vLLM profile — treat the concurrency number as an estimate.** `UNVERIFIED`

### 8.5 Serverless GPU — Modal, which is what Week 8 uses

[Pricing](https://modal.com/pricing) · [Cold-start guide](https://modal.com/docs/guide/cold-start)

- **Per-second billing**, "by the CPU cycle". **Idle costs nothing.**

| GPU | $/second | $/hour |
|---|---|---|
| H100 SXM5 | 0.001097 | **3.95** |
| A100 80GB | 0.000694 | 2.50 |
| A100 40GB | 0.000583 | 2.10 |
| L40S | 0.000542 | 1.95 |
| A10 | 0.000306 | 1.10 |
| L4 | 0.000222 | 0.80 |
| **T4** *(what Week 8 uses)* | **0.000164** | **0.59** |

Plus: CPU $0.0000131/core/s (min 0.125 cores); memory $0.00000222/GiB/s; volumes $0.09/GiB-month with 1 TiB/month free. Free credits **$30/month Starter, $100/month Team.**

- **Cold starts:** *"Containers boot in about one second."* Two latency sources: **queueing** (no warm container available) and **initialization** (first-invocation work, e.g. loading model weights). Reduction techniques:
  - **Concurrent file loading** — download/load multiple large shards in parallel rather than sequentially. Substantial for transformer weights.
  - **`@modal.enter`** — move init into the warm-up phase rather than the first request. *Shifts latency; does not remove it.*
  - **Memory snapshots** — *"captures the state of a container's memory at user-controlled points"* and reuses it on future boots; cuts both cold-start and warm-up.
  - **`scaledown_window`** — idle container lifetime, **2 s to 20 min, default 60 s.**
  - **`min_containers`** — always-warm floor. **`buffer_containers`** — pre-provisioned headroom for bursts.
  - Explicit trade-off stated in the docs: warm containers "increase the resources consumed."

> **Cold start is dominated by weight loading, not container boot.** One second of boot versus tens of seconds for tens of GB of weights. Attack it with memory snapshots, parallel loading, smaller/quantised weights, and a local cache/volume — **not by switching platforms.** This is the single most useful deployment insight in Week 8, and it generalises to every serverless-GPU provider.

Alternatives (**`UNVERIFIED`** — not fetched; verify pricing and cold-start behaviour before citing): RunPod (serverless + community pods), Replicate (model-as-API), Baseten (Truss, dedicated + autoscale), Together AI / Fireworks / Groq (hosted inference APIs, no cold start you control), Beam, Cerebrium, SageMaker Serverless/Async, Fly.io GPUs, Cloud Run GPUs.

### 8.6 Cost / latency / cold-start — the decision frame

| Pattern | Cold start | $ profile | Fits |
|---|---|---|---|
| Hosted API (OpenAI/Anthropic/Together) | none | per-token, no idle cost, highest marginal | spiky, low-volume, general capability |
| **Serverless GPU (Modal etc.)** | seconds → minutes (weights load) | per-second, ~zero idle | **bursty, batch, custom weights, demos — i.e. Week 8** |
| Serverless + `min_containers ≥ 1` | ~none | pay for the warm floor | interactive, latency-SLA'd |
| Dedicated GPU (vLLM on a rented/owned box) | one-time | flat hourly; cheapest **only at high utilization** | steady high volume |
| Local (Ollama/llama.cpp/MLX) | seconds | hardware only | dev, privacy, single-user, offline |

**Rules of thumb:**

- **The break-even between per-token API and dedicated GPU is a utilization question.** An H100 at $3.95/hr is **~$2,850/month**. Compute your token throughput at target latency and compare to API pricing at your actual volume. **Below ~30–40% utilization, dedicated hosting usually loses.**
- **Quantize for the regime** (per "Give Me BF16 or Give Me Death", see `06-training-and-finetuning.md` §7.3): **W4A16 for latency-bound single-stream; W8A8 for throughput-bound high-concurrency.**
- **Batching is the cheapest optimization you have.** Continuous batching + prefix caching typically beats every model-level trick you would try first.
- **Always instrument TTFT separately from total latency.** For streaming UIs, TTFT is the *perceived* latency — and **prefill (prompt-length-bound) and decode (KV/bandwidth-bound) are governed by different bottlenecks**, so one number hides both.
- **Set a max concurrency per replica derived from the KV arithmetic in §8.4.** Over-admitting requests turns a latency problem into an OOM.

---

## 9. Putting a Gradio UI in front of a model

- **Three entry points:** **`gr.Interface`** (single function in/out), **`gr.Blocks`** (arbitrary layout and event wiring), **`gr.ChatInterface`** (chat, with streaming and message history built in). **Generator functions give you token streaming** — this is the whole trick.
- **`share=True`** creates a public tunnel URL (FRP-based) with a **1-week expiry that is explicitly best-effort** — see the maintainers' own issues asking to reword the message to say so: [#13398](https://github.com/gradio-app/gradio/issues/13398), [#7267](https://github.com/gradio-app/gradio/issues/7267).

> **Do not use `share=True` links for anything real.** They are for showing a colleague something today. ~1 week, best-effort, by the maintainers' own admission. A share link in a report, a ticket, or a stakeholder email is a broken link with a deadline.

- **Permanent hosting:** **HuggingFace Spaces** (free CPU tier, paid GPU tiers, `gradio deploy`), or mount the app in FastAPI (`gr.mount_gradio_app`) and containerise.
- **Production-ish concerns:**
  - Enable the queue with concurrency limits (`.queue(...)`, `concurrency_limit` per event) so a GPU-bound function does not get trampled. **On a T4 this matters immediately** — two simultaneous users on an unqueued GPU function is an OOM.
  - `auth=` for basic gating — **this is not real authentication.** Do not treat it as an access control for anything sensitive.
  - Programmatic access via **`gradio_client`**.
  - Gradio can expose an app **as an MCP server** (`mcp_server=True`) — [guide](https://gradio.app/guides/building-mcp-server-with-gradio). Nice symmetry: your demo UI becomes a tool other agents can call.
- **`UNVERIFIED`:** exact current expiry wording, queue defaults, and the MCP-mode flag name. The Gradio docs pages returned binary content to the fetcher; the two GitHub issues above are the load-bearing citations. Verify on [sharing-your-app](https://gradio.app/guides/sharing-your-app) and [the Interface reference](https://gradio.app/docs/gradio/interface).

### 9.1 The reference architecture worth remembering

**Gradio (Spaces or FastAPI) → OpenAI-compatible HTTP → vLLM on Modal or a dedicated GPU**, with `min_containers=1` if a human is waiting.

> **Because vLLM speaks the OpenAI API, the same front-end works against a hosted API, a local Ollama, or your own vLLM.** That substitutability is the actual lesson of the deployment week — the interface is the portable asset, and the model behind it is a swappable, negotiable, re-tenderable component. For an enterprise buyer that is a procurement argument, not just an engineering one.

**Alternatives:** Streamlit (broader dashboards, worse streaming ergonomics), Chainlit / AG-UI / assistant-ui (chat-and-agent-native, tool-call rendering, HITL approval widgets), plain Next.js + Vercel AI SDK (production front-ends).

---

## 10. Corrections table — claims that are wrong or outdated

| Claim | Status |
|---|---|
| "Multi-agent systems outperform single agents" | **Domain-dependent.** ~80% of the measured gain on breadth-first research tasks is explained by **token spend** (~15× chat); reliably underperforms a single well-contexted agent on tasks needing consistent shared decisions ([Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system); [Cognition](https://cognition.com/blog/dont-build-multi-agents)). |
| "Split the work between subagents to go faster" | **Only if the subtasks are read-only, independent, and verifiable.** Otherwise you get conflicting implicit decisions — the Flappy Bird failure. Claude Code deliberately restricts subagents to answering questions. |
| "Passing the original task to each subagent is enough context" | **Explicitly refuted by Cognition.** *"Share context, and share full agent traces, not just individual messages."* |
| "Agents work — gpt-4o gets high tool-use scores" | **τ-bench: <50% single-trial, `pass^8` < 25% in retail.** Leaderboard scores are single-trial upper bounds. |
| "`pass@k` shows the agent is reliable" | **Wrong metric.** `pass@k` rewards *any* success in k tries. Production needs **`pass^k`** — all k succeed. |
| "Implement ReAct by parsing Thought:/Action: from the completion" | **Obsolete.** Native tool-calling APIs and reasoning models subsume it. ReAct survives as a concept, not a recipe. |
| "Add a self-reflection step to improve quality" | **Needs an external/objective signal.** Pure self-critique without ground truth often just adds tokens and confident rewrites. |
| "LangGraph is part of LangChain" | **Outdated split.** LangChain v1 = prebuilt agent; **LangGraph = the low-level runtime underneath it.** |
| `initialize_agent` / `AgentExecutor` / `LLMChain` | **Pre-1.0 LangChain APIs.** Also: `langchain-ai.github.io/langgraph` URLs now redirect to `docs.langchain.com/oss/...`. |
| "Put your API call in the node, then interrupt for approval" | **It will run twice.** On resume, **the interrupted node re-executes from its beginning.** Side effects must come *after* the interrupt. |
| "OpenAI Swarm is the OpenAI agent framework" | **Educational/deprecated** → **Agents SDK** on the **Responses API**. Also skip Assistants-API agent tutorials. |
| "CrewAI is a LangChain wrapper" | **No LangChain claim in the current intro docs.** Standalone for some time (`UNVERIFIED` exact version). |
| "CrewAI = Crews" | **Crews *and* Flows**, with **Flows as the recommended backbone**; the docs say "use both." |
| **"MCP sampling lets a server borrow the client's model"** | **DEPRECATED in spec `2026-07-28`.** *"New implementations should integrate directly with LLM provider APIs."* |
| "MCP logging is a client primitive" | **DEPRECATED.** Log to `stderr` (stdio) or use OpenTelemetry. |
| "MCP is a stateful protocol with an `initialize` handshake" | **Now stateless.** Per-request version + capabilities in `_meta`; discovery via the mandatory (but optional-to-call, cacheable) **`server/discover`**. |
| "MCP uses HTTP+SSE transport" | **Removed.** Transports are **stdio and Streamable HTTP**; SSE is only an optional streaming mechanism *within* Streamable HTTP. |
| "MCP notifications are pushed reliably" | **Opt-in via `subscriptions/listen`, and explicitly best-effort.** *"There are no guarantees that every notification will be sent or received."* **Poll as well.** |
| "MCP has Roots as a client primitive" | **Absent from the `2026-07-28` architecture page** — `UNVERIFIED` whether removed, deprecated, or just unlisted. |
| "LangGraph vs MCP — pick one" | **Category error.** Frameworks = orchestration; MCP = the integration layer. All three frameworks can consume MCP servers. |
| `response_format={"type":"json_object"}` + "please output JSON" | **Superseded by Structured Outputs**, which is the only option that guarantees *schema* adherence, not just valid JSON. |
| "Structured Outputs means you can just parse the response" | **No.** Branch on the **`refusal`** field, and **check the finish reason** — responses can be truncated by token limits. |
| **"TGI is the production standard"** | **TGI is in MAINTENANCE MODE.** HF recommends **vLLM / SGLang / llama.cpp / MLX** instead. |
| "Ollama for production serving" | **Collapses under concurrency** (roughly single-digit users); no continuous batching + paged KV. Local dev and single-user only. |
| "Size the GPU from the model weights" | **Concurrency is set by the KV cache, not the weights.** 128 KiB/token for Llama-3-8B ⇒ ~1 GiB per 8k sequence. |
| "Cold start is a container-boot problem" | **It is a weight-loading problem.** ~1 s boot vs tens of seconds of weights. Fix with memory snapshots, parallel loading, quantised weights. |
| "Dedicated GPU is cheaper than an API" | **Only at high utilization.** H100 ≈ $2,850/month; below ~30–40% utilization the API usually wins. |
| "`share=True` is how you host a Gradio demo" | **~1 week, best-effort**, by the maintainers' own admission. Use Spaces or FastAPI. |
| "Week 8 needs a GPU" | **No.** The GPU is rented from Modal per-second: `GPU = "T4"` in `pricer_service2.py`, `pricer_ephemeral.py`, `llama.py`. (Contrast **Week 7**, which needs a **CUDA** GPU and is not provisioned in the repo's `pyproject.toml`/`uv.lock`.) |

---

## 11. Items I could not verify

- **ReAct paper specifics** (arXiv 2210.03629) — not fetched. The conceptual description here is safe; do not attribute specific claims or numbers to it.
- **The exact unsupported JSON Schema keywords in OpenAI strict mode** — the docs page fetched said only that "some features are unavailable." Check `additionalProperties`, required-properties behaviour, nesting/property/enum limits, and `minLength`/`pattern`/`format` support on the live page.
- **Whether MCP "Roots"** was removed, deprecated, or merely absent from the `2026-07-28` architecture page.
- **OTel GenAI attribute names and stability level** — the repo still has TODOs and 136 open issues; read the `model/*.yaml` files before teaching attribute names.
- **CrewAI's exact LangChain-independence version.**
- **SGLang specifics** — docs not fetched.
- **Serverless-GPU alternatives to Modal** — RunPod, Replicate, Baseten, Together, Fireworks, Groq, Beam, Cerebrium, SageMaker, Fly.io, Cloud Run: pricing and cold-start behaviour unverified.
- **All "vLLM is N× faster than Ollama" multipliers** — secondary blog sources only. The *qualitative* finding (Ollama collapses under concurrency) is consistent across sources; the numbers are not citable.
- **The KV-cache concurrency estimate for Llama-3-8B (§8.4)** — arithmetic from published architecture parameters, not a measured vLLM profile.
- **Gradio share-link expiry wording, queue defaults, and the MCP-mode flag name** — the docs returned binary content to the fetcher; the two GitHub issues are the only load-bearing citations.
- **Observability platform feature comparisons** — all secondary; verify before recommending any of them internally.
- **Exact lecture numbers** for Week 8 in Ed Donner's course — not recorded in the research file.

**Where a number matters to a decision, re-check it.** The MCP spec and the LangChain/LangGraph docs in particular changed materially in 2026 and will change again before your next study block.
