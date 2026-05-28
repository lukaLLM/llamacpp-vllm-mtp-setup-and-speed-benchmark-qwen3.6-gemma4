#!/usr/bin/env python3

import argparse
import csv
import json
import re
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import pyfiglet
except ImportError:
    pyfiglet = None


console = Console()

MTP_DISPLAY_NAME = "MTP"
DEFAULT_MODE = MTP_DISPLAY_NAME
YOUTUBE_CHANNEL = "@LukaszGawendaAI"
YOUTUBE_URL = "www.youtube.com/@LukaszGawendaAI"
DEFAULT_MTP_PORT = 8000
DEFAULT_NO_MTP_PORT = 8001
DEFAULT_MTP_URL = f"http://localhost:{DEFAULT_MTP_PORT}"
DEFAULT_NO_MTP_URL = f"http://localhost:{DEFAULT_NO_MTP_PORT}"
DEFAULT_RESULTS_CSV = Path(__file__).with_name("comparison_runs.csv")
DEFAULT_CONTEXT_SIZE = 3000
DEFAULT_CONTEXT_RESERVE = 128
DEFAULT_GENERATION_TOKENS = 1500
CSV_FIELDS = [
    "session_name",
    "mode",
    "run_index",
    "tokens",
    "elapsed_s",
    "tok_per_s",
]


@dataclass
class BenchState:
    name: str
    url: str
    color: str
    output: str = ""
    running: bool = False
    error: Optional[str] = None
    run_index: int = 0
    elapsed: float = 0.0
    token_count: int = 0
    last_tokens: int = 0
    last_elapsed: float = 0.0
    scores: List[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def live_tps(self) -> float:
        with self.lock:
            if self.elapsed <= 0:
                return 0.0
            return self.token_count / self.elapsed

    def rolling_avg(self) -> float:
        with self.lock:
            values = list(self.scores)
            if self.running and self.elapsed > 0 and self.token_count > 0:
                values.append(self.token_count / self.elapsed)
            return statistics.mean(values) if values else 0.0

def big_number(value: float, color: str) -> Text:
    number = f"{value:.1f}"

    if pyfiglet:
        rendered = pyfiglet.figlet_format(number, font="small")
    else:
        rendered = f"\n{number}\n"

    return Text(rendered, style=f"bold {color}")


def wrap_text(text: str, width: int) -> List[str]:
    if not text:
        return ["Waiting for benchmark..."]

    lines = []
    for raw_line in text.splitlines() or [text]:
        line = raw_line.rstrip()
        if not line:
            lines.append("")
            continue

        while len(line) > width:
            lines.append(line[:width])
            line = line[width:]
        lines.append(line)

    return lines


def trim_output(text: str, max_lines: int, max_width: int) -> str:
    lines = wrap_text(text, max_width)
    lines = lines[-max_lines:]
    return "\n".join(lines)


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    word_like = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    char_like = max(1, round(len(text.encode("utf-8")) / 4))
    return max(word_like, char_like)


def auto_generation_tokens(prompt: str, context_size: int, reserve_tokens: int) -> int:
    prompt_tokens = estimate_token_count(prompt)
    return max(1, context_size - prompt_tokens - reserve_tokens)


def append_csv_row(
    csv_file: Path,
    session_name: str,
    state: BenchState,
) -> None:
    with state.lock:
        run_index = state.run_index
        tokens = state.last_tokens
        elapsed = state.last_elapsed
    if tokens <= 0 or elapsed <= 0:
        return

    tok_per_s = tokens / elapsed
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_file.exists() or csv_file.stat().st_size == 0

    with csv_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "session_name": session_name,
                "mode": state.name,
                "run_index": run_index,
                "tokens": tokens,
                "elapsed_s": f"{elapsed:.6f}",
                "tok_per_s": f"{tok_per_s:.6f}",
            }
        )


def parse_stream_line(line: bytes):
    text = line.decode("utf-8", errors="ignore").strip()

    if not text:
        return None

    if text.startswith("data:"):
        text = text[5:].strip()

    if text == "[DONE]":
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def is_premature_stream_end(exc: Exception) -> bool:
    text = str(exc).lower()
    return "response ended prematurely" in text or "incomplete read" in text


def run_single_stream(state: BenchState, prompt: str, n_predict: int, seed: int):
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": True,
    }

    with state.lock:
        state.output = ""
        state.error = None
        state.running = True
        state.elapsed = 0.0
        state.token_count = 0
        state.last_tokens = 0
        state.last_elapsed = 0.0
        state.run_index += 1

    start = time.perf_counter()
    final_tps = None

    try:
        response = requests.post(
            f"{state.url.rstrip('/')}/completion",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            stream=True,
            timeout=900,
        )
        response.raise_for_status()

        previous_server_tokens = 0

        for raw_line in response.iter_lines():
            now = time.perf_counter()
            obj = parse_stream_line(raw_line)
            if not obj:
                continue

            content = obj.get("content", "")
            timings = obj.get("timings") or {}

            if timings.get("predicted_per_second"):
                final_tps = float(timings["predicted_per_second"])

            server_tokens = (
                obj.get("tokens_predicted")
                or obj.get("predicted_n")
                or timings.get("predicted_n")
            )

            with state.lock:
                state.elapsed = now - start

                if content:
                    state.output += content

                    if len(state.output) > 10000:
                        state.output = state.output[-10000:]

                    if isinstance(server_tokens, int) and server_tokens > previous_server_tokens:
                        state.token_count = server_tokens
                        previous_server_tokens = server_tokens
                    else:
                        state.token_count += 1

    except Exception as exc:
        with state.lock:
            if state.token_count > 0 and is_premature_stream_end(exc):
                state.error = None
            else:
                state.error = str(exc)

    end = time.perf_counter()

    with state.lock:
        state.elapsed = max(end - start, 0.001)
        state.last_elapsed = state.elapsed
        state.last_tokens = state.token_count

        if final_tps is None:
            final_tps = state.token_count / state.elapsed if state.token_count else 0.0

        if final_tps > 0:
            state.scores.append(final_tps)

        state.running = False


def make_panel(state: BenchState, total_runs: int, panel_width: int, output_lines: int) -> Panel:
    rolling_avg = state.rolling_avg()
    live_tps = state.live_tps()

    with state.lock:
        output = state.output
        running = state.running
        error = state.error
        run_index = state.run_index
        completed = len(state.scores)
        last_tokens = state.last_tokens
        last_elapsed = state.last_elapsed

    status = "running" if running else "ready"
    if error:
        status = "error"

    stats = Table.grid(expand=True)
    stats.add_column(justify="left")
    stats.add_column(justify="right")

    stats.add_row("status", status)
    stats.add_row("run", f"{min(run_index, total_runs)}/{total_runs}")
    stats.add_row("completed", str(completed))
    stats.add_row("live tok/s", f"{live_tps:.1f}")
    stats.add_row("avg tok/s", f"{rolling_avg:.1f}")
    if last_tokens > 0 and last_elapsed > 0:
        stats.add_row("last run", f"{last_tokens} tokens in {last_elapsed:.2f}s")
    else:
        stats.add_row("last run", "-")

    output_width = max(40, panel_width - 12)

    if error:
        body = Group(
            Text(f"ERROR: {error}", style="bold red"),
            stats,
        )
    else:
        output_box = Panel(
            Text(
                trim_output(
                    output,
                    max_lines=output_lines,
                    max_width=output_width,
                ),
                style="white",
            ),
            title="Live output",
            border_style="dim",
            height=output_lines + 3,
            box=box.ROUNDED,
        )

        body = Group(
            Align.center(big_number(rolling_avg, state.color)),
            stats,
            output_box,
        )

    return Panel(
        body,
        title=f"[bold {state.color}]{state.name}[/bold {state.color}]",
        border_style=state.color,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def make_ui(
    no_mtp_state: BenchState,
    mtp_state: BenchState,
    prompt: str,
    total_runs: int,
    n_predict: int,
    context_size: int,
    output_lines: int,
    prompt_lines: int,
):
    no_mtp_avg = no_mtp_state.rolling_avg()
    mtp_avg = mtp_state.rolling_avg()
    speedup = mtp_avg / no_mtp_avg if no_mtp_avg > 0 else 0.0

    terminal_width = console.size.width
    panel_width = max(60, terminal_width // 2 - 4)
    prompt_width = max(60, terminal_width - 12)

    prompt_display = prompt.replace("<|im_start|>user", "User:")
    prompt_display = prompt_display.replace("<|im_start|>assistant", "Assistant:")
    prompt_display = prompt_display.replace("<|im_end|>", "")
    prompt_display = "\n".join(line.rstrip() for line in prompt_display.splitlines()).strip()

    header = Table.grid(expand=True)
    header.add_column(ratio=1)
    header.add_column(width=18)

    title = Text(f"NO MTP vs {MTP_DISPLAY_NAME} Speed Comparison", style="bold white")
    speed = Text(f"{speedup:.2f}x", style="bold green")

    meta = Text()
    meta.append("Mode: ", style="dim")
    meta.append(DEFAULT_MODE, style="bold cyan")
    meta.append(" | Channel: ", style="dim")
    meta.append(YOUTUBE_CHANNEL, style="bold magenta")
    meta.append(" | ", style="dim")
    meta.append(YOUTUBE_URL, style="bold blue")
    meta.append(" | ctx: ", style="dim")
    meta.append(str(context_size), style="bold white")
    meta.append(" | gen: ", style="dim")
    meta.append(str(n_predict), style="bold white")

    header.add_row(
        title,
        Align.right(speed),
    )
    header.add_row(meta, Text(""))

    panels = Table.grid(expand=True)
    panels.add_column(ratio=1)
    panels.add_column(ratio=1)

    first_state, second_state = (
        (mtp_state, no_mtp_state)
        if DEFAULT_MODE == MTP_DISPLAY_NAME
        else (no_mtp_state, mtp_state)
    )

    prompt_panel = Panel(
        Text(
            trim_output(prompt_display, max_lines=prompt_lines, max_width=prompt_width),
            style="white",
        ),
        title=f"Prompt ({prompt_lines} lines)",
        border_style="white",
        box=box.ROUNDED,
        padding=(1, 2),
        height=prompt_lines + 4,
    )

    panels.add_row(
        make_panel(first_state, total_runs, panel_width, output_lines),
        make_panel(second_state, total_runs, panel_width, output_lines),
    )

    return Group(
        Panel(header, border_style="white", box=box.ROUNDED),
        prompt_panel,
        panels,
    )


def check_server(url: str) -> bool:
    try:
        response = requests.get(f"{url.rstrip('/')}/health", timeout=5)
        return response.status_code < 500
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no-mtp-url",
        "--default-url",
        dest="no_mtp_url",
        default=DEFAULT_NO_MTP_URL,
        help=f"NO MTP server URL (default: {DEFAULT_NO_MTP_URL})",
    )
    parser.add_argument(
        "--mtp-url",
        "--onebonsai-url",
        dest="mtp_url",
        default=DEFAULT_MTP_URL,
        help=f"MTP server URL (default: {DEFAULT_MTP_URL})",
    )
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--n-predict",
        type=int,
        default=DEFAULT_GENERATION_TOKENS,
        help=f"Generated tokens per request (default: {DEFAULT_GENERATION_TOKENS}).",
    )
    parser.add_argument(
        "--use-full-context",
        action="store_true",
        help="Use context_size - estimated_prompt_tokens - reserve as generation budget.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
        help=f"Server context window from Docker config (default: {DEFAULT_CONTEXT_SIZE}).",
    )
    parser.add_argument(
        "--context-reserve",
        type=int,
        default=DEFAULT_CONTEXT_RESERVE,
        help=f"Tokens reserved so prompt + output stays under context (default: {DEFAULT_CONTEXT_RESERVE}).",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-lines", type=int, default=16)
    parser.add_argument("--prompt-lines", type=int, default=4)
    parser.add_argument(
        "--session-name",
        default=f"compare-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        help="Session name saved in CSV rows.",
    )
    parser.add_argument(
        "--csv-file",
        default=str(DEFAULT_RESULTS_CSV),
        help=f"CSV output path (default: {DEFAULT_RESULTS_CSV}).",
    )

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run one server after the other. Fairer benchmark, less flashy demo.",
    )

    parser.add_argument(
        "--prompt",
        default=(
            "<|im_start|>user\n"
            "Write a Python program to find the nth Fibonacci number using recursion. "
            "Explain the solution briefly and include the code. Be concise don't include unnecessary details and emojis.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    )

    args = parser.parse_args()
    csv_path = Path(args.csv_file)
    n_predict = (
        auto_generation_tokens(
            args.prompt,
            context_size=args.context_size,
            reserve_tokens=args.context_reserve,
        )
        if args.use_full_context
        else args.n_predict
    )

    no_mtp_state = BenchState(
        name="NO MTP",
        url=args.no_mtp_url,
        color="red",
    )

    mtp_state = BenchState(
        name=MTP_DISPLAY_NAME,
        url=args.mtp_url,
        color="green",
    )

    print("Checking servers...")
    print(f"NO MTP:  {args.no_mtp_url} -> {'OK' if check_server(args.no_mtp_url) else 'not ready'}")
    print(f"{MTP_DISPLAY_NAME}:    {args.mtp_url} -> {'OK' if check_server(args.mtp_url) else 'not ready'}")
    time.sleep(1)

    with Live(
        make_ui(
            no_mtp_state,
            mtp_state,
            args.prompt,
            args.runs,
            n_predict,
            args.context_size,
            args.output_lines,
            args.prompt_lines,
        ),
        refresh_per_second=8,
        screen=True,
        console=console,
    ) as live:
        for _ in range(args.runs):
            if args.sequential:
                t1 = threading.Thread(
                    target=run_single_stream,
                    args=(no_mtp_state, args.prompt, n_predict, args.seed),
                    daemon=True,
                )
                t1.start()

                while t1.is_alive():
                    live.update(
                        make_ui(
                            no_mtp_state,
                            mtp_state,
                            args.prompt,
                            args.runs,
                            n_predict,
                            args.context_size,
                            args.output_lines,
                            args.prompt_lines,
                        )
                    )
                    time.sleep(0.12)

                t2 = threading.Thread(
                    target=run_single_stream,
                    args=(mtp_state, args.prompt, n_predict, args.seed),
                    daemon=True,
                )
                t2.start()

                while t2.is_alive():
                    live.update(
                        make_ui(
                            no_mtp_state,
                            mtp_state,
                            args.prompt,
                            args.runs,
                            n_predict,
                            args.context_size,
                            args.output_lines,
                            args.prompt_lines,
                        )
                    )
                    time.sleep(0.12)

            else:
                t1 = threading.Thread(
                    target=run_single_stream,
                    args=(no_mtp_state, args.prompt, n_predict, args.seed),
                    daemon=True,
                )
                t2 = threading.Thread(
                    target=run_single_stream,
                    args=(mtp_state, args.prompt, n_predict, args.seed),
                    daemon=True,
                )

                t1.start()
                t2.start()

                while t1.is_alive() or t2.is_alive():
                    live.update(
                        make_ui(
                            no_mtp_state,
                            mtp_state,
                            args.prompt,
                            args.runs,
                            n_predict,
                            args.context_size,
                            args.output_lines,
                            args.prompt_lines,
                        )
                    )
                    time.sleep(0.12)

            live.update(
                make_ui(
                    no_mtp_state,
                    mtp_state,
                    args.prompt,
                    args.runs,
                    n_predict,
                    args.context_size,
                    args.output_lines,
                    args.prompt_lines,
                )
            )
            append_csv_row(csv_path, args.session_name, no_mtp_state)
            append_csv_row(csv_path, args.session_name, mtp_state)

    no_mtp_avg = statistics.mean(no_mtp_state.scores) if no_mtp_state.scores else 0.0
    mtp_avg = statistics.mean(mtp_state.scores) if mtp_state.scores else 0.0
    speedup = mtp_avg / no_mtp_avg if no_mtp_avg > 0 else 0.0

    print()
    print("Final result")
    print(f"NO MTP avg:  {no_mtp_avg:.2f} tok/s")
    print(f"{MTP_DISPLAY_NAME} avg:    {mtp_avg:.2f} tok/s")
    print(f"{MTP_DISPLAY_NAME} speedup:{speedup:.2f}x")
    print(f"Saved per-request rows to: {csv_path}")


if __name__ == "__main__":
    main()
