# Agentic equation discovery — LangGraph + Claude

A 1-hour, tutorial-style demo for the **AI in Science** summer school (Chicago, 2026).

You hand the system **noisy data** from a hidden formula. A small team of agents then
recovers the underlying equation — illustrating the features that distinguish an
*agentic* system from a single LLM call:

- **Coding** — an agent writes Python to fit the data
- **Tool execution** — an agent runs that code in a real interpreter
- **Self-repair** — when the code fails to run, a Debugger agent sees the exact failing code + error and fixes it (up to `MAX_AUTO_REPAIRS` times)
- **Critic / reflection** — an agent *looks at the plot* (and the numbers), judges the fit, and can trigger an automatic retry
- **Human-in-the-loop** — a human decides whether to *improve* or *stop*

They are orchestrated as a **LangGraph state graph with cycles**, so the system iterates
until the human is satisfied. The Plotter runs *before* the Critic so the Critic can see
the fitted curve, not just R². If the code crashes it never reaches the Critic — the
Debugger repairs it first.

```
  START → CODER → EXECUTOR → PLOTTER → CRITIC → HUMAN → END ("stop")
            ▲        │ ▲                          │
            │        ▼ │ failure: auto-fix (×N)   │
            │     DEBUGGER ┘                       │
            └──────────── "improve" + feedback ───┘
```

## Files

| File | What it is |
|---|---|
| `agentic_discovery.py` | The agents + the LangGraph graph. Run it for a terminal demo. |
| `demo_notebook.ipynb` | Step-by-step walkthrough for the live lecture. |
| `requirements.txt` | Dependencies. |
| `plots/` | Fit plots written by the Plotter agent (created on first run). |
| `code/` | The Python the Coder writes each iteration (plus the Debugger's repairs), saved one file per attempt. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=...        # your Google (Gemini) API key
```

### Choosing the model provider

The default backend is **Google (Gemini)**. To use **Anthropic (Claude)** instead,
set the API key and pick the provider at runtime — no code edit needed:

```bash
export ANTHROPIC_API_KEY=...
LLM_PROVIDER=anthropic python agentic_discovery.py
```

| Env var | Meaning | Default |
|---|---|---|
| `LLM_PROVIDER` | `google` or `anthropic` | `google` |
| `LLM_MODEL` | override the model | `gemini-3.5-flash` / `claude-opus-4-8` |
| `LLM_MAX_TOKENS` | max output tokens per call (must fit the Coder's full code; thinking models also spend it) | `16384` |

Both default models support vision (the Critic is shown the fit plot) and structured output.

## Run

**Terminal demo** (prompts you to improve/stop at each step):

```bash
python agentic_discovery.py
```

At startup it asks you for the analytic function used to **generate** the data
(in terms of `x`, e.g. `3*exp(-0.35*x)*cos(2.4*x+0.5)+0.2`; press Enter for the
default). The agents never see this formula — they must rediscover it from the
noisy points. You can use `exp/log/sqrt/sin/cos/tan/tanh/pi/e` and `**`.

**Notebook** (best for the live lecture):

```bash
jupyter notebook demo_notebook.ipynb
```

## Knobs worth turning during the talk

- Model/provider — see the table above (`LLM_PROVIDER` / `LLM_MODEL`); defaults to
  Google `gemini-3.5-flash`. Edit `DEFAULT_MODELS` in `agentic_discovery.py` to change
  the baked-in defaults.
- `MAX_AUTO_REPAIRS` — how many times the Critic auto-retries when generated code crashes.
- Dataset — by default you **type the data-generating formula** at startup
  (`make_from_expression(...)`, default = a damped oscillation). Ready-made example
  formulas live in `EXAMPLE_EXPRS` (feed any to `make_from_expression(...)`):
  `trend_plus_oscillation` (dominant trend + small oscillation — a line fits at a
  deceptively-high R²≈0.97 but leaves periodic residuals), `double_exponential`
  (equifinality demo), and `damped_oscillation`.

## Note on safety

The Executor runs **LLM-generated code with `exec`**. That is fine for a trusted
classroom demo but is **not a sandbox**. In production, run untrusted code in a
container / restricted subprocess, or use Anthropic's server-side code-execution tool.
This is called out in the code so students see the trade-off.
