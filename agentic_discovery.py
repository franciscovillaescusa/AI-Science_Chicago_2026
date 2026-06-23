"""
agentic_discovery.py
====================================================================
A minimal multi-agent system that discovers the equation hidden in
noisy data, built with LangGraph + Claude.

Tutorial: "AI in Science" summer school (Chicago, 2026).

The point of this file is *pedagogical*: it shows, in as little code
as possible, the features that make an "agentic" system more than
a single LLM call:

    1. CODING            -- an agent writes Python to fit the data
    2. TOOL EXECUTION    -- an agent runs that code in a real interpreter
    3. SELF-REPAIR       -- an agent debugs the code when it fails to run
    4. CRITIC / REFLECT  -- an agent judges the result and can trigger a retry
    5. HUMAN-IN-THE-LOOP -- a human decides whether to improve or stop

These agents are wired together as a LangGraph *state graph* with
cycles, so the system can iterate until the human is satisfied. The
PLOTTER runs before the CRITIC so the CRITIC can *see* the fitted plot
(data points + curve + residuals), not just the numbers. If the code
fails to run, the DEBUGGER fixes it (up to MAX_AUTO_REPAIRS times)
before we ever bother the CRITIC or the human:

  START → CODER → EXECUTOR → PLOTTER → CRITIC → HUMAN → END ("stop")
            ▲        │ ▲                          │
            │        ▼ │ failure: auto-fix (×N)   │
            │     DEBUGGER ┘                       │
            └──────────── "improve" + feedback ───┘

Run it as a script for a terminal demo:

    export GOOGLE_API_KEY=...           # for the default Google/Gemini backend
    python agentic_discovery.py

It also works with Anthropic's Claude API -- just pick the provider at runtime:

    export ANTHROPIC_API_KEY=...
    LLM_PROVIDER=anthropic python agentic_discovery.py

(Optionally override the model with LLM_MODEL=...) Or import the pieces from
the companion notebook `demo_notebook.ipynb`.
"""

from __future__ import annotations

import base64
import operator
import os
import shutil
import textwrap
import traceback
from typing import Annotated, Any, Optional, TypedDict

import numpy as np

# --- LangChain / LangGraph imports -------------------------------------------
# Provider clients (ChatAnthropic / ChatGoogleGenerativeAI) are imported lazily
# inside get_llm(), so you only need the package for the provider you actually
# use. Everything else is provider-agnostic LangChain / LangGraph.
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# =============================================================================
# 0.  Configuration
# =============================================================================
# Which LLM backend to use: "google" (Gemini) or "anthropic" (Claude). Override
# at runtime without editing the file via the LLM_PROVIDER env var, e.g.:
#     LLM_PROVIDER=anthropic python agentic_discovery.py
PROVIDER = os.environ.get("LLM_PROVIDER", "google").lower()

# Default model per provider. BOTH must support vision (the CRITIC is shown the
# fit plot) and structured output. For a classroom demo with many students you
# may want a cheaper/faster model: "claude-haiku-4-5" / "claude-sonnet-4-6" for
# Anthropic, or "gemini-2.5-flash" for Google. Override with LLM_MODEL.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "google": "gemini-3.5-flash",
}
MODEL = os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(PROVIDER, "")

# The env var each provider expects to find its API key in.
API_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY"}

# Max output tokens per LLM call. This must comfortably fit the CODER's whole
# structured reply (equation + reasoning + the FULL fitting code) -- a too-small
# value silently truncates the code mid-line. Note that "thinking" models (e.g.
# Gemini 3.x) also spend this budget on internal reasoning before the answer, so
# keep it generous. Override with LLM_MAX_TOKENS.
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "16384"))

# Stop the CODER/EXECUTOR/CRITIC self-repair cycle from running forever when
# the generated code keeps crashing. The human loop is separate and unbounded.
MAX_AUTO_REPAIRS = 3

# Where fit plots are written.
PLOT_DIR = "plots"

# Where the code written by the CODER (and the DEBUGGER's repairs) is saved,
# one file per attempt, so you can inspect exactly what ran each iteration.
CODE_DIR = "code"


def reset_output_dirs() -> None:
    """Delete and recreate the plots/ and code/ folders for a clean run.

    Called at the start of the script so each run starts fresh; the folders are
    otherwise appended to (and old iterations would linger).
    """
    for d in (PLOT_DIR, CODE_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)


def _save_code(iteration: int, code: str, label: str, header: str = "") -> str:
    """Write a generated code snippet to CODE_DIR and return its path.

    `label` distinguishes the author within an iteration (e.g. "coder",
    "debug_1"); `header` is an optional comment block (equation, reasoning, ...).
    """
    os.makedirs(CODE_DIR, exist_ok=True)
    path = os.path.join(CODE_DIR, f"iter_{iteration:02d}_{label}.py")
    with open(path, "w") as fh:
        if header:
            fh.write(textwrap.indent(header.strip(), "# ") + "\n\n")
        fh.write(code.rstrip() + "\n")
    return path


_LLM: Optional[BaseChatModel] = None


def get_llm() -> BaseChatModel:
    """Lazily build the chat model for the selected PROVIDER so `import` works
    without an API key (and without needing both provider packages installed)."""
    global _LLM
    if _LLM is None:
        if PROVIDER == "anthropic":
            from langchain_anthropic import ChatAnthropic
            _LLM = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS, timeout=120)
        elif PROVIDER == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            _LLM = ChatGoogleGenerativeAI(model=MODEL, max_tokens=MAX_TOKENS, timeout=120)
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER {PROVIDER!r}; use 'anthropic' or 'google'."
            )
    return _LLM


def _ask(system: str, human: Any, schema: type[BaseModel], retries: int = 2) -> Any:
    """Make one structured LLM call and return the parsed result.

    Every agent talks to the model the same way -- a system prompt plus a single
    human turn (`human` is either a string or a multimodal content list), forced
    into the Pydantic `schema`. The three agents funnel through here so the call
    pattern (and the retry below) lives in one place.

    A provider sometimes returns no parseable structured output -- notably Gemini
    when a "thinking" model spends its whole token budget on internal reasoning,
    or when a reply is empty/blocked. That surfaces as a `None` here, which would
    otherwise crash the caller with a confusing AttributeError. Retry a few times
    and, failing that, raise a clear, actionable error instead.
    """
    llm = get_llm().with_structured_output(schema)
    messages = [SystemMessage(system), HumanMessage(human)]
    for _ in range(retries + 1):
        parsed = llm.invoke(messages)
        if parsed is not None:
            return parsed
    raise RuntimeError(
        f"The model returned no parseable {schema.__name__} after {retries + 1} "
        f"attempts. This usually means the response was empty or the token budget "
        f"was exhausted -- try raising LLM_MAX_TOKENS (thinking models spend it on "
        f"internal reasoning) or switching LLM_MODEL / LLM_PROVIDER."
    )


# =============================================================================
# 1.  The shared state
# =============================================================================
# In LangGraph every node receives the state and returns a *partial* update.
# By default a returned key overwrites the old value; keys with a reducer
# (here `history`, via `operator.add`) are merged instead.
class DiscoveryState(TypedDict, total=False):
    # ---- inputs (set once, at the start) ----
    x: list[float]                 # independent variable (e.g. time)
    y: list[float]                 # measured, noisy observations
    hint: str                      # optional natural-language hint about the data

    # ---- working fields (rewritten each iteration) ----
    iteration: int                 # how many times the CODER has run
    auto_repairs: int              # consecutive automatic retries after a crash
    equation: str                  # human-readable form the CODER proposed
    reasoning: str                 # why the CODER chose that form
    code: str                      # the Python the CODER wrote
    result: dict[str, Any]         # what the EXECUTOR produced (params, R^2, error...)
    critique: str                  # the CRITIC's assessment
    good_enough: bool              # the CRITIC's verdict
    plot_path: str                 # file written by the PLOTTER
    decision: str                  # "improve" | "stop" from the human
    feedback: str                  # free-text guidance from the human

    # ---- append-only log of every attempt (uses a reducer) ----
    history: Annotated[list[dict[str, Any]], operator.add]


# =============================================================================
# 2.  Structured outputs (so we don't have to parse free text)
# =============================================================================
class CodeProposal(BaseModel):
    """What we force the CODER agent to return."""

    equation: str = Field(description="The analytic form you chose, as a human-readable string.")
    reasoning: str = Field(description="One or two sentences on why this equation fits the data.")
    code: str = Field(description="Python code that fits the data, following the required contract.")


class Critique(BaseModel):
    """What we force the CRITIC agent to return."""

    assessment: str = Field(description="A short, plain-language judgement of the fit quality.")
    good_enough: bool = Field(description="True if the fit is scientifically convincing.")
    suggestions: str = Field(description="Concrete advice for the next attempt (empty if good enough).")


class CodeFix(BaseModel):
    """What we force the DEBUGGER agent to return."""

    diagnosis: str = Field(description="One sentence: what was wrong with the code.")
    code: str = Field(description="The corrected Python code, following the same contract.")


# =============================================================================
# 3.  The CODER agent  --  writes Python to fit the data
# =============================================================================
# Contract: the code it writes is exec'd in a namespace that already contains
# `x`, `y` (numpy arrays), `np`, and `curve_fit`. It MUST build a dict named
# `result` with these keys.  We tell the model this explicitly.
CODER_SYSTEM = textwrap.dedent(
    """
    You are given a noisy 1-D dataset: arrays `x` and `y`. We want to find an
    analytic equation y = f(x) that explains the data. The data is noisy, so
    capture the underlying trend rather than fitting every point exactly.

    Write Python that fits your proposed equation to the data. When your code is
    executed in a namespace that already contains `x` and `y` (numpy arrays),
    `np`, and `curve_fit` (scipy.optimize.curve_fit), it MUST define a dict named
    `result` with EXACTLY these keys:

        result = {
            "equation":    str,          # the analytic form you chose, as a string
            "param_names": list[str],    # names of the fitted parameters
            "popt":        list[float],  # fitted parameter values (json-friendly)
            "y_pred":      list[float],  # your equation evaluated at x with popt
            "r2":          float,        # coefficient of determination
            "rmse":        float,        # root-mean-square error
        }

    Guidelines:
      - Define a model function and fit it with `curve_fit`; choose an initial
        guess derived from the data so the fit converges.
      - Prefer a simple equation, but make it rich enough to capture the real
        trend in the data.
      - Convert numpy arrays to plain lists in `result` (use .tolist()).
      - Do NOT call plt.show(), print(), or read/write files; keep it
        self-contained (you may `import` from numpy/scipy if needed).
    """
).strip()


def _format_data_for_prompt(x: list[float], y: list[float]) -> str:
    xs = ", ".join(f"{v:.4g}" for v in x)
    ys = ", ".join(f"{v:.4g}" for v in y)
    return f"x = [{xs}]\n\ny = [{ys}]"


def _history_digest(history: list[dict[str, Any]]) -> str:
    """Summarise past attempts so the CODER learns from them."""
    if not history:
        return "(no previous attempts)"
    lines = []
    for h in history:
        tag = f"  attempt {h['iteration']}: {h['equation']!r}"
        if h.get("error"):
            tag += "  -> CRASHED:\n" + textwrap.indent(h["error"].strip().splitlines()[-1], "      ")
        else:
            tag += f"  -> R^2={h.get('r2'):.4f}, RMSE={h.get('rmse'):.4g}"
        if h.get("critique"):
            tag += f"\n      critic: {h['critique']}"
        if h.get("feedback"):
            tag += f"\n      human: {h['feedback']}"
        lines.append(tag)
    return "\n".join(lines)


def coder_node(state: DiscoveryState) -> dict[str, Any]:
    """Agent #1 -- proposes a functional form and writes the fitting code."""
    iteration = state.get("iteration", 0) + 1
    history = state.get("history", [])

    instructions = [
        _format_data_for_prompt(state["x"], state["y"]),
    ]
    if state.get("hint"):
        instructions.append(f"\nHint about the data: {state['hint']}")
    if history:
        instructions.append("\nPrevious attempts (learn from these, do not repeat mistakes):")
        instructions.append(_history_digest(history))
    if state.get("feedback"):
        instructions.append(f"\nThe human reviewer asked you to: {state['feedback']}")

    instructions.append("\nPropose a (possibly new) functional form and write the fitting code.")

    proposal: CodeProposal = _ask(CODER_SYSTEM, "\n".join(instructions), CodeProposal)

    # The model sometimes writes the equation with a leading "y =" / "f(x) =";
    # strip it so display strings like "y = {equation}" don't read "y = y = ...".
    equation = proposal.equation.strip()
    for prefix in ("y =", "y=", "f(x) =", "f(x)=", "f =", "f="):
        if equation.lower().startswith(prefix):
            equation = equation[len(prefix):].strip()
            break

    print(f"\n[CODER] iteration {iteration}: proposing  y = {equation}")
    print(f"        reasoning: {proposal.reasoning}")

    path = _save_code(iteration, proposal.code, "coder",
                      header=f"iteration {iteration} — CODER\n"
                             f"equation: y = {equation}\n"
                             f"reasoning: {proposal.reasoning}")
    print(f"        saved code to {path}")

    return {
        "iteration": iteration,
        "equation": equation,
        "reasoning": proposal.reasoning,
        "code": proposal.code,
    }


# =============================================================================
# 4.  The EXECUTOR agent  --  runs the generated code in a real interpreter
# =============================================================================
# NOTE ON SAFETY: this runs LLM-generated code with `exec`. That is fine for a
# trusted classroom demo, but it is NOT a sandbox. In production you would run
# this in a container / restricted subprocess (or use Anthropic's server-side
# code-execution tool). We keep it simple here so the mechanism is visible.
def _to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python types.

    The CODER's code often leaves numpy.float64 (or arrays) in `result` even
    though we ask for plain floats. Those values end up in the LangGraph state,
    which is checkpointed with msgpack at the human-in-the-loop pause -- and
    msgpack cannot serialize numpy types. Coerce everything here so the graph
    can always be paused/resumed regardless of what the generated code produced.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):  # numpy scalar (float64, int64, bool_, ...)
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def execute_code(code: str, x: list[float], y: list[float]) -> dict[str, Any]:
    """Execute the CODER's code and return its `result` dict (or an error)."""
    from scipy.optimize import curve_fit  # local import keeps module import light

    # Catch incomplete/truncated code up front with an actionable message (a raw
    # SyntaxError traceback is cryptic; this tells the DEBUGGER what to do).
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return {"error": f"SyntaxError: {exc}\nThe code looks incomplete or "
                         "truncated. Rewrite it COMPLETELY as valid, "
                         "self-contained Python."}

    namespace: dict[str, Any] = {
        "np": np,
        "numpy": np,
        "curve_fit": curve_fit,
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
    }
    try:
        exec(code, namespace)  # noqa: S102 -- intentional, see safety note above
    except Exception:
        return {"error": traceback.format_exc()}

    result = namespace.get("result")
    if not isinstance(result, dict):
        return {"error": "The code ran but did not define a dict named `result`."}

    # Be forgiving: recompute R^2 / RMSE from y_pred if the model omitted them.
    if "y_pred" in result and ("r2" not in result or "rmse" not in result):
        try:
            y_arr = np.asarray(y, dtype=float)
            y_pred = np.asarray(result["y_pred"], dtype=float)
            ss_res = float(np.sum((y_arr - y_pred) ** 2))
            ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
            result.setdefault("r2", 1.0 - ss_res / ss_tot if ss_tot else float("nan"))
            result.setdefault("rmse", float(np.sqrt(np.mean((y_arr - y_pred) ** 2))))
        except Exception:
            pass

    # Coerce numpy types to native Python so the result is JSON/msgpack-friendly
    # (required for LangGraph to checkpoint the state at the human pause).
    return _to_native(result)


def executor_node(state: DiscoveryState) -> dict[str, Any]:
    """Agent #2 -- runs the code and reports back what happened."""
    result = execute_code(state["code"], state["x"], state["y"])
    if result.get("error"):
        print("[EXECUTOR] the code failed (will route to the DEBUGGER).")
    else:
        print(f"[EXECUTOR] fit succeeded: R^2 = {result.get('r2'):.4f}, "
              f"RMSE = {result.get('rmse'):.4g}")
    return {"result": result}


# =============================================================================
# 4b.  The DEBUGGER agent  --  repairs code that failed to run
# =============================================================================
# When the EXECUTOR reports an error, we do NOT throw the attempt away and ask
# the CODER for a brand-new equation (it would only see a one-line error summary
# and tends to repeat the same mistake). Instead a dedicated DEBUGGER sees the
# EXACT failing code and the FULL error, and returns a corrected version that
# keeps the same equation. It loops DEBUGGER -> EXECUTOR until the code runs or
# the MAX_AUTO_REPAIRS budget is spent.
DEBUGGER_SYSTEM = textwrap.dedent(
    """
    You are a Python debugging agent. You are given a snippet that was supposed
    to fit a noisy dataset and FAILED -- either it raised an exception (a
    traceback is shown) or it ran to completion but did not leave a top-level
    dict named `result`.

    The snippet is executed with `exec` in a namespace that already contains
    `x` and `y` (numpy arrays), `np`, and `curve_fit` (scipy.optimize.curve_fit).
    Return a CORRECTED version that runs without error and, at the TOP LEVEL
    (not inside a function, not guarded by try/except that can swallow it),
    defines a dict named `result` with EXACTLY these keys:

        result = {
            "equation":    str,          # the analytic form, as a string
            "param_names": list[str],    # names of the fitted parameters
            "popt":        list[float],  # fitted parameter values
            "y_pred":      list[float],  # the equation evaluated at x with popt
            "r2":          float,        # coefficient of determination
            "rmse":        float,        # root-mean-square error
        }

    Rules:
      - Keep the SAME equation / functional form; only fix the bug.
      - Make sure `result` is ALWAYS assigned when the code finishes. If a step
        can fail (e.g. curve_fit not converging), fix the cause (better initial
        guess / bounds / maxfev) rather than hiding it behind a bare except.
      - Convert numpy arrays to plain lists in `result` (use .tolist()).
      - Do NOT call plt.show(), print(), or read/write files.
    """
).strip()


def debugger_node(state: DiscoveryState) -> dict[str, Any]:
    """Agent #2b -- given the failing code + error, return corrected code."""
    attempt = state.get("auto_repairs", 0) + 1
    error = state["result"].get("error", "(no error text)")

    message = "\n".join([
        f"Proposed equation (keep this form): y = {state['equation']}",
        "\nThe following code FAILED:",
        "```python\n" + state["code"].strip() + "\n```",
        "\nError / problem reported by the executor:",
        "```\n" + error.strip() + "\n```",
        "\nReturn corrected code that runs and defines `result` per the contract.",
    ])

    fix: CodeFix = _ask(DEBUGGER_SYSTEM, message, CodeFix)
    path = _save_code(state["iteration"], fix.code, f"debug_{attempt}",
                      header=f"iteration {state['iteration']} — DEBUGGER repair "
                             f"{attempt}/{MAX_AUTO_REPAIRS}\n"
                             f"equation: y = {state['equation']}\n"
                             f"diagnosis: {fix.diagnosis}")
    print(f"[DEBUGGER] repair {attempt}/{MAX_AUTO_REPAIRS}: {fix.diagnosis}")
    print(f"           saved fixed code to {path}")
    return {"code": fix.code, "auto_repairs": attempt}


# =============================================================================
# 5.  The CRITIC agent  --  judges the result, can trigger an automatic retry
# =============================================================================
CRITIC_SYSTEM = textwrap.dedent(
    """
    You are a meticulous scientific reviewer. You are shown (1) a proposed
    equation and the numbers from fitting it to noisy data, and (2) a PLOT
    containing the raw data points, the fitted curve overlaid, and the
    residuals underneath. Judge whether the fit is scientifically convincing.

    Use BOTH the numbers and the picture. Look at the plot: does the curve
    track the data across the whole range, or miss systematically in some
    region? Do the residuals scatter like structureless noise around zero, or
    is there leftover structure (curvature, drift, periodicity) the model
    failed to capture? Is the model unnecessarily complex or too simple?
    Be concise and concrete in your suggestions.
    """
).strip()


def critic_node(state: DiscoveryState) -> dict[str, Any]:
    """Agent #3 -- reflects on the fit and writes feedback for the next round."""
    result = state["result"]
    history_entry = {
        "iteration": state["iteration"],
        "equation": state["equation"],
    }

    # We only reach the CRITIC with an error after the DEBUGGER has used up its
    # whole repair budget (see `route_after_executor`). Nothing to judge -- just
    # report the failure to the human, who can ask for a fresh attempt or stop.
    if result.get("error"):
        last_line = result["error"].strip().splitlines()[-1]
        critique = (f"The code still failed after {MAX_AUTO_REPAIRS} automatic "
                    f"repair attempts ({last_line}).")
        print(f"[CRITIC] code unrepairable after {MAX_AUTO_REPAIRS} attempts.")
        history_entry.update(error=result["error"], critique=critique,
                             feedback=state.get("feedback", ""))
        return {
            "critique": critique,
            "good_enough": False,
            "history": [history_entry],
        }

    summary = (
        f"Proposed equation: y = {state['equation']}\n"
        f"Fitted parameters: "
        + ", ".join(f"{n}={v:.4g}" for n, v in zip(result.get("param_names", []),
                                                    result.get("popt", [])))
        + f"\nGoodness of fit: R^2 = {result.get('r2'):.5f}, RMSE = {result.get('rmse'):.4g}"
        + "\n\nThe attached plot shows the data points, this fitted curve, and the "
          "residuals. Judge the fit using both the numbers and the picture."
    )

    # Multimodal message: text summary + the plot the PLOTTER just rendered, so
    # the CRITIC literally *sees* the fit instead of judging on R^2 alone.
    content: list[dict[str, Any]] = [{"type": "text", "text": summary}]
    plot_path = state.get("plot_path")
    if plot_path and os.path.exists(plot_path):
        with open(plot_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )

    review: Critique = _ask(CRITIC_SYSTEM, content, Critique)
    verdict = "looks good" if review.good_enough else "could be better"
    print(f"[CRITIC] {verdict}: {review.assessment}")

    history_entry.update(
        r2=result.get("r2"),
        rmse=result.get("rmse"),
        critique=review.assessment + (f" Suggestion: {review.suggestions}" if review.suggestions else ""),
        feedback=state.get("feedback", ""),
    )
    return {
        "critique": history_entry["critique"],
        "good_enough": review.good_enough,
        "auto_repairs": 0,  # a successful run resets the crash counter
        "history": [history_entry],
    }


# =============================================================================
# 6.  The PLOTTER agent  --  visualises the current fit
# =============================================================================
def plotter_node(state: DiscoveryState) -> dict[str, Any]:
    """Agent #4 -- draws data vs. fitted curve (+ residuals) and saves a PNG.

    We build the figure with matplotlib's object-oriented `Figure` API rather
    than pyplot, so saving works headlessly (scripts, servers) without touching
    the global backend -- the companion notebook keeps its inline plotting.
    """
    from matplotlib.figure import Figure

    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, f"fit_iter_{state['iteration']:02d}.png")

    x = np.asarray(state["x"], dtype=float)
    y = np.asarray(state["y"], dtype=float)
    result = state["result"]

    if result.get("error") or "y_pred" not in result:
        # The code never produced a fit -- still show the raw data for the human.
        fig = Figure(figsize=(8, 4.5))
        ax = fig.subplots()
        ax.scatter(x, y, s=18, alpha=0.7, label="data")
        ax.set_title(f"iteration {state['iteration']} — no successful fit yet")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.legend()
        fig.tight_layout(); fig.savefig(path, dpi=110)
        print(f"[PLOTTER] saved {path} (raw data only)")
        return {"plot_path": path}

    order = np.argsort(x)
    y_pred = np.asarray(result["y_pred"], dtype=float)

    fig = Figure(figsize=(8, 6))
    ax1, ax2 = fig.subplots(2, 1, sharex=True,
                            gridspec_kw={"height_ratios": [3, 1]})
    ax1.scatter(x, y, s=18, alpha=0.7, label="data")
    ax1.plot(x[order], y_pred[order], "r-", lw=2, label="fit")
    ax1.set_title(f"iteration {state['iteration']}:  y = {state['equation']}\n"
                  f"R$^2$ = {result.get('r2'):.4f}")
    ax1.set_ylabel("y"); ax1.legend()

    ax2.scatter(x, y - y_pred, s=14, color="gray")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("x"); ax2.set_ylabel("residual")

    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"[PLOTTER] saved {path}")
    return {"plot_path": path}


# =============================================================================
# 7.  The HUMAN-IN-THE-LOOP node  --  pause and ask whether to improve or stop
# =============================================================================
def human_node(state: DiscoveryState) -> dict[str, Any]:
    """Pause the graph and hand control to a person.

    `interrupt(payload)` stops execution and surfaces `payload` to the caller.
    The graph resumes when the caller invokes it again with
    `Command(resume=<value>)`; that value becomes the return of `interrupt()`.
    """
    result = state.get("result", {})
    payload = {
        "iteration": state["iteration"],
        "equation": state["equation"],
        "r2": result.get("r2"),
        "rmse": result.get("rmse"),
        "critique": state.get("critique", ""),
        "good_enough": state.get("good_enough", False),
        "plot_path": state.get("plot_path", ""),
        "question": "Improve the fit or stop? Respond with "
                    "{'action': 'improve'|'stop', 'feedback': '...'}.",
    }
    decision = interrupt(payload)  # <-- execution pauses here

    # On resume, `decision` is whatever the caller passed to Command(resume=...).
    if isinstance(decision, str):
        action = "stop" if decision.strip().lower().startswith("s") else "improve"
        feedback = ""
    else:
        action = (decision or {}).get("action", "stop")
        feedback = (decision or {}).get("feedback", "")

    print(f"[HUMAN] decision: {action}" + (f" — feedback: {feedback!r}" if feedback else ""))
    return {"decision": action, "feedback": feedback, "auto_repairs": 0}


# =============================================================================
# 8.  Routing (the conditional edges that create the cycles)
# =============================================================================
def route_after_executor(state: DiscoveryState) -> str:
    """Error + budget left -> DEBUGGER (auto-fix).  Otherwise -> PLOTTER."""
    if state["result"].get("error") and state.get("auto_repairs", 0) < MAX_AUTO_REPAIRS:
        return "debugger"
    return "plotter"


def route_after_human(state: DiscoveryState) -> str:
    """'improve' -> back to CODER (with feedback).  'stop' -> END."""
    return "coder" if state.get("decision") == "improve" else END


# =============================================================================
# 9.  Build the graph
# =============================================================================
def build_graph():
    """Wire the agents into a LangGraph state machine with a human gate."""
    g = StateGraph(DiscoveryState)

    g.add_node("coder", coder_node)
    g.add_node("executor", executor_node)
    g.add_node("debugger", debugger_node)
    g.add_node("critic", critic_node)
    g.add_node("plotter", plotter_node)
    g.add_node("human", human_node)

    g.add_edge(START, "coder")
    g.add_edge("coder", "executor")
    # On a failure the DEBUGGER fixes the code and re-runs it; once it works (or
    # the repair budget is spent) we move on to plot + critique.
    g.add_conditional_edges("executor", route_after_executor,
                            {"debugger": "debugger", "plotter": "plotter"})
    g.add_edge("debugger", "executor")
    g.add_edge("plotter", "critic")     # plot first, so the critic can see it
    g.add_edge("critic", "human")
    g.add_conditional_edges("human", route_after_human, {"coder": "coder", END: END})

    # A checkpointer is REQUIRED for interrupt()/resume to work: it stores the
    # paused state so we can continue the same run later.
    return g.compile(checkpointer=MemorySaver())


# =============================================================================
# 10.  Datasets:  generate noisy (x, y) from a hidden ground-truth formula
# =============================================================================
# The user supplies the analytic function that GENERATES the data; the agents
# never see it and must rediscover it from the noisy points alone.

# Math names the user may use in their expression, mapped to numpy. No Python
# builtins are exposed, so `eval` here only evaluates arithmetic + these.
_EXPR_NAMESPACE: dict[str, Any] = {
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt,
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "abs": np.abs, "sign": np.sign, "power": np.power,
    "pi": np.pi, "e": np.e,
}


def make_from_expression(expr: str, n: int = 60, noise: float = 0.15,
                         seed: int = 0, xmin: float = 0.0, xmax: float = 10.0):
    """Generate noisy data from a user-supplied analytic expression y = f(x).

    `expr` is a Python/math expression in the variable `x` (you may also write
    `t`) with concrete numeric coefficients, e.g.
        "3.0*exp(-0.35*x)*cos(2.4*x + 0.5) + 0.2"
    It may use exp/log/sqrt/sin/cos/tan/tanh/.../pi/e and `**` for powers.

    Returns (x, y, truth) where `truth` records the exact formula used so you
    can compare against what the agents recover. Raises ValueError on a bad
    expression so the caller can re-prompt.
    """
    expr = expr.replace("^", "**")  # be friendly: treat ^ as exponentiation
    rng = np.random.default_rng(seed)
    x = np.linspace(xmin, xmax, n)
    namespace = {**_EXPR_NAMESPACE, "x": x, "t": x}
    try:
        clean = eval(expr, {"__builtins__": {}}, namespace)  # noqa: S307 -- math only
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not evaluate {expr!r}: {exc}") from exc
    # Broadcast constants (an expr with no `x`) to the full grid.
    clean = np.asarray(clean, dtype=float) * np.ones_like(x)
    y = clean + rng.normal(0, noise, size=n)
    truth = {"equation": expr, "params": {}}
    return x.tolist(), y.tolist(), truth


# Ready-made example formulas to paste at the prompt (or feed to
# make_from_expression in the notebook). The agents only ever see the noisy
# points, never the formula. Each one exercises a different part of the loop:
#
#   damped_oscillation     -- clean and recognisable; a strong CODER one-shots it.
#   double_exponential     -- deliberately *deceptive*: a monotonic decay looks
#                             like a SINGLE exponential, so the natural first
#                             guess A e^{-k t}+C fits with a tempting R^2 (~0.97),
#                             but its residuals are visibly CURVED, not noise. It
#                             is really a fast + slow decay (a classic
#                             multi-exponential relaxation: fluorescence
#                             lifetimes, pharmacokinetics, RC networks); the fix
#                             raises R^2 to ~0.998.
#   trend_plus_oscillation -- a dominant linear TREND with a smaller periodic
#                             term on top. A straight line already scores a
#                             high-looking R^2 (~0.97), but the leftover
#                             residuals are clearly periodic (amplitude ~4x the
#                             noise). Adding an oscillatory term reaches ~0.998.
#
# The last two are where the vision CRITIC (which sees the residual panel) earns
# its keep: R^2 alone looks fine, so the Coder -> Critic -> improve cycle only
# does *visible* work when a too-simple model leaves structure in the residuals.
EXAMPLE_EXPRS: dict[str, str] = {
    "damped_oscillation":     "3.0*exp(-0.35*x)*cos(2.4*x + 0.5) + 0.2",
    "double_exponential":     "4.0*exp(-2.5*x) + 2.0*exp(-0.35*x) + 0.5",
    "trend_plus_oscillation": "0.6*x + 1.0 + 0.5*sin(2.2*x + 0.5)",
}


# =============================================================================
# 11.  Script entry point: drive the graph from the terminal
# =============================================================================
# Default data-generating formula if the user just presses Enter: a damped
# oscillation y = A e^{-b t} cos(w t + phi) + C (an exponentially decaying sinusoid).
DEFAULT_EXPR = EXAMPLE_EXPRS["damped_oscillation"]


def prompt_for_dataset():
    """Ask the user for the analytic function used to GENERATE the data.

    The agents never see this formula -- they must rediscover it from the noisy
    points. Re-prompts on a bad expression; a blank line uses DEFAULT_EXPR.
    """
    print("\nEnter the analytic function used to GENERATE the data, in terms of x.")
    print("  - usable functions: exp, log, sqrt, sin, cos, tan, tanh, pi, e, and **")
    print("  - the agents will NOT see it; they must discover it from the noisy data")
    print(f"  - press Enter to use the default:  y = {DEFAULT_EXPR}")
    print("  - or copy one of these examples (good for showing off the critic loop):")
    for name, expr in EXAMPLE_EXPRS.items():
        print(f"      {name:24s} y = {expr}")
    while True:
        expr = input("y = ").strip() or DEFAULT_EXPR
        try:
            x, y, truth = make_from_expression(expr)
            return x, y, truth
        except ValueError as exc:
            print(f"  !! {exc}\n     Try again (e.g. 3*exp(-0.35*x)*cos(2.4*x+0.5)+0.2).")


def main():
    key_env = API_KEY_ENV.get(PROVIDER, "GOOGLE_API_KEY")
    if not os.environ.get(key_env):
        raise SystemExit(f"Please set {key_env} before running (LLM_PROVIDER={PROVIDER}).")

    print("=" * 70)
    print(f"  Agentic equation discovery  —  LangGraph + {PROVIDER} ({MODEL})")
    print("=" * 70)

    reset_output_dirs()  # start each run fresh: wipe old plots/ and code/
    print(f"Cleared '{PLOT_DIR}/' and '{CODE_DIR}/' for a fresh run.")

    x, y, truth = prompt_for_dataset()
    print(f"\nGenerated {len(x)} noisy points from your hidden formula.")
    print(f"(For reference — the agents do NOT see this — truth = {truth['equation']})\n")

    graph = build_graph()
    config = {"configurable": {"thread_id": "tutorial-demo"}}

    state0: DiscoveryState = {
        "x": x,
        "y": y,
        # No hint on purpose: the agent must infer the functional form from the
        # data alone. (The `hint` field still works if you want to supply one.)
        "iteration": 0,
        "auto_repairs": 0,
        "history": [],
    }

    # First invocation runs CODER -> EXECUTOR -> CRITIC -> PLOTTER -> (pause at HUMAN).
    result = graph.invoke(state0, config)

    # Each time the graph pauses, `result["__interrupt__"]` holds the payload we
    # built in human_node. We show it, ask the user, then resume.
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n" + "-" * 70)
        print(f"  HUMAN REVIEW (iteration {payload['iteration']})")
        print(f"  equation : y = {payload['equation']}")
        if payload.get("r2") is not None:
            print(f"  R^2      : {payload['r2']:.4f}")
        if payload.get("rmse") is not None:
            print(f"  RMSE     : {payload['rmse']:.4g}")
        print(f"  critic   : {payload['critique']}")
        print(f"  plot     : {payload['plot_path']}  (open it to inspect)")
        print("-" * 70)

        choice = input("Improve or stop?  [i = improve / s = stop]: ").strip().lower()
        if choice.startswith("s"):
            result = graph.invoke(Command(resume={"action": "stop"}), config)
        else:
            fb = input("Optional feedback for the coder (press Enter to skip): ").strip()
            result = graph.invoke(
                Command(resume={"action": "improve", "feedback": fb}), config
            )

    print("\n" + "=" * 70)
    print("  DONE")
    print(f"  Final equation : y = {result.get('equation')}")
    if isinstance(result.get("result"), dict) and result["result"].get("r2") is not None:
        print(f"  Final R^2      : {result['result']['r2']:.4f}")
    print(f"  True equation  : y = {truth['equation']}")
    print(f"  Iterations     : {result.get('iteration')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
