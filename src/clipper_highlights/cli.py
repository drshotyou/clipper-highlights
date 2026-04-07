from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from clipper_highlights.config import ProjectConfig
from clipper_highlights.pipeline import run_pipeline

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.command()
def init_config(
    destination: Path = typer.Argument(..., help="Where to write the starter YAML config."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing file."),
) -> None:
    if destination.exists() and not force:
        raise typer.BadParameter(f"{destination} already exists. Pass --force to overwrite it.")

    config = ProjectConfig()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(config.dump_yaml(), encoding="utf-8")
    console.print(f"Wrote starter config to {destination}")


@app.command()
def run(
    input_video: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input recording."),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Run directory. Defaults to ./runs/<input-stem>.",
    ),
    config_path: Path | None = typer.Option(None, "--config", "-c", help="YAML config path."),
    game: str | None = typer.Option(None, "--game", help="Game name for metadata and prompts."),
    keyword: list[str] = typer.Option(
        None,
        "--keyword",
        help="Override keywords using phrase or phrase=weight. Repeat to add many.",
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="Disable Gemini and use heuristics only."),
    no_export: bool = typer.Option(False, "--no-export", help="Skip final ffmpeg clip export."),
    force: bool = typer.Option(False, "--force", help="Recompute cached artifacts."),
) -> None:
    config = ProjectConfig.load(config_path)
    if no_llm:
        config.llm.provider = "none"
    _apply_keyword_overrides(config, keyword or [])

    run_dir = output_dir or Path("runs") / input_video.stem
    result = run_pipeline(
        input_video,
        run_dir,
        config,
        game=game,
        export_enabled=not no_export,
        force=force,
    )

    summary = Table(title="Clipper Highlights")
    summary.add_column("Item")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Run dir", str(result.run_dir))
    summary.add_row("Candidates", str(len(result.bundle.candidate_windows)))
    summary.add_row("Ranked clips", str(len(result.ranked_clips)))
    summary.add_row("Exported clips", str(len(result.exported_paths)))
    console.print(summary)

    if result.ranked_clips:
        clips = Table(title="Top Clips")
        clips.add_column("#")
        clips.add_column("Start")
        clips.add_column("End")
        clips.add_column("Score")
        clips.add_column("Title", overflow="fold")
        for index, clip in enumerate(result.ranked_clips, start=1):
            clips.add_row(
                str(index),
                f"{clip.start:.2f}",
                f"{clip.end:.2f}",
                f"{clip.score:.2f}",
                clip.title,
            )
        console.print(clips)


def _apply_keyword_overrides(config: ProjectConfig, raw_keywords: list[str]) -> None:
    for item in raw_keywords:
        term, weight = _parse_keyword_override(item)
        config.candidate.keywords[term] = weight


def _parse_keyword_override(raw: str) -> tuple[str, float]:
    if "=" not in raw:
        return raw.strip().lower(), 1.0

    term, weight = raw.split("=", 1)
    try:
        return term.strip().lower(), float(weight)
    except ValueError as exc:
        raise typer.BadParameter(f"Invalid keyword override: {raw}") from exc
