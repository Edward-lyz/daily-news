import datetime
import json
import os
import time
from typing import Any, Dict, List, Tuple

import feedparser
import requests
import yaml

UTC = datetime.timezone.utc


def utc_today() -> datetime.date:
    return datetime.datetime.now(tz=UTC).date()


def date_range_for_yesterday() -> Tuple[
    datetime.date, datetime.datetime, datetime.datetime
]:
    today = utc_today()
    y_date = today - datetime.timedelta(days=1)
    start = datetime.datetime(
        y_date.year, y_date.month, y_date.day, 0, 0, 0, tzinfo=UTC
    )
    end = start + datetime.timedelta(days=1)
    return y_date, start, end


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return text[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


def limit_diff_budget(
    commits: Dict[str, List[Dict[str, Any]]], max_total_chars: int
) -> int:
    if max_total_chars <= 0:
        return 0
    remaining = max_total_chars
    truncated = 0
    for repo_items in commits.values():
        for item in repo_items:
            for file_item in item.get("files") or []:
                patch = file_item.get("patch") or ""
                if not patch:
                    continue
                if remaining <= 0:
                    file_item["patch"] = ""
                    file_item["patch_truncated"] = True
                    truncated += 1
                    continue
                if len(patch) > remaining:
                    file_item["patch"] = truncate_text(patch, remaining)
                    file_item["patch_truncated"] = True
                    truncated += 1
                    remaining = 0
                else:
                    remaining -= len(patch)
    return truncated


def fetch_arxiv(query: str, max_results: int = 50) -> List[Any]:
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    feed = feedparser.parse(response.text)
    return feed.entries


def entry_date(entry: Any) -> datetime.date:
    parsed = getattr(entry, "updated_parsed", None) or getattr(
        entry, "published_parsed", None
    )
    if not parsed:
        return datetime.datetime.now(tz=UTC).date()
    return datetime.datetime(*parsed[:6], tzinfo=UTC).date()


def filter_yesterday_arxiv(
    entries: List[Any], y_date: datetime.date
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if entry_date(entry) != y_date:
            continue
        out.append(
            {
                "title": normalize_space(entry.title),
                "link": entry.link,
                "authors": [a.name for a in getattr(entry, "authors", [])],
                "summary": normalize_space(entry.summary),
            }
        )
    return out


def fetch_github_commits(
    repo: str, since_iso: str, token: str | None = None
) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/commits"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"since": since_iso, "per_page": 100}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    out: List[Dict[str, str]] = []
    for item in data:
        commit = item.get("commit", {})
        message = (commit.get("message") or "").splitlines()[0]
        author = (commit.get("author") or {}).get("name", "")
        date = (commit.get("author") or {}).get("date", "")
        out.append(
            {
                "sha": item.get("sha", "")[:7],
                "full_sha": item.get("sha", ""),
                "msg": message,
                "url": item.get("html_url", ""),
                "author": author,
                "date": date,
            }
        )
    return out


def fetch_commit_detail(
    repo: str,
    full_sha: str,
    token: str | None,
    max_files: int,
    max_diff_chars: int,
) -> Dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}/commits/{full_sha}"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    files: List[Dict[str, Any]] = []
    raw_files = data.get("files", []) or []
    for file_item in raw_files[: max_files if max_files > 0 else 0]:
        patch = file_item.get("patch") or ""
        files.append(
            {
                "filename": file_item.get("filename", ""),
                "status": file_item.get("status", ""),
                "additions": file_item.get("additions", 0),
                "deletions": file_item.get("deletions", 0),
                "changes": file_item.get("changes", 0),
                "patch": truncate_text(patch, max_diff_chars),
            }
        )

    return {
        "files": files,
        "stats": data.get("stats") or {},
        "truncated_files": len(raw_files) > max_files,
    }


def openrouter_summarize(
    prompt: str, model: str, retry_max: int, retry_base_seconds: int
) -> str:
    key = os.environ["OPENROUTER_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise AI infra newsletter editor. Output Markdown only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    attempts = max(1, retry_max)
    for attempt in range(attempts):
        response = requests.post(url, headers=headers, json=body, timeout=60)
        if response.ok:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        detail = truncate_text(response.text.strip(), 2000)
        retriable = response.status_code in {429, 500, 502, 503, 504}
        if retriable and attempt < attempts - 1:
            retry_after = response.headers.get("Retry-After", "").strip()
            if retry_after.isdigit():
                delay = int(retry_after)
            else:
                delay = retry_base_seconds * (2**attempt)
            time.sleep(min(delay, 60))
            continue
        raise RuntimeError(
            f"OpenRouter {response.status_code} {response.reason}: {detail}"
        )
    raise RuntimeError("OpenRouter summarize failed: retry budget exhausted.")


def build_prompt(
    date_str: str,
    papers: List[Dict[str, Any]],
    commits: Dict[str, List[Dict[str, Any]]],
) -> str:
    payload = {
        "date": date_str,
        "papers": papers,
        "commits": commits,
        "requirements": [
            "Return a Markdown report with three top-level sections: 摘要, 具体内容分析, 总结.",
            "For papers: include a 1-line why it matters (AI infra angle).",
            "For repos: use file-level diffs in commits[*].files[*].patch to summarize changes.",
            "For repos: include a brief evaluation (impact/risk/regression) per repo.",
            "If diffs are missing or patch_truncated is true, say so explicitly.",
            "Write the report in Chinese.",
            "总结仓库的改动时，需要在你说出的每个总结的点上都附上对应的 github 的 PR 链接，方便跳转。没有 PR 链接就给出 commit 名称。",
            "总结不能过于潦草，需要严谨地指出，哪里组件/模块改动了什么具体细节，最好有简短的代码片段。",
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def render_fallback_markdown(
    date_str: str,
    papers: List[Dict[str, Any]],
    commits: Dict[str, List[Dict[str, Any]]],
    errors: List[str],
) -> str:
    lines: List[str] = []
    lines.append(f"# Daily AI Infra Report - {date_str}")
    lines.append("")
    lines.append("Generated from raw data because summarization was unavailable.")
    lines.append("")
    lines.append("## Papers")
    if not papers:
        lines.append("- No papers matched the date filter.")
    else:
        for paper in papers:
            title = paper.get("title", "").strip()
            link = paper.get("link", "").strip()
            authors = ", ".join(paper.get("authors", []))
            summary = paper.get("summary", "").strip()
            header = f"- [{title}]({link})"
            if authors:
                header += f" - {authors}"
            lines.append(header)
            if summary:
                lines.append(f"  - {summary}")
    lines.append("")
    lines.append("## Repos")
    if not commits:
        lines.append("- No repos configured.")
    else:
        for repo, items in commits.items():
            lines.append(f"### {repo}")
            if not items:
                lines.append("- No commits in the window.")
                continue
            for item in items:
                sha = item.get("sha", "")
                msg = item.get("msg", "")
                url = item.get("url", "")
                author = item.get("author", "")
                suffix = f" ({author})" if author else ""
                lines.append(f"- [`{sha}`]({url}) {msg}{suffix}")
                stats = item.get("stats") or {}
                if stats:
                    lines.append(
                        f"  - Stats: +{stats.get('additions', 0)}/-{stats.get('deletions', 0)} "
                        f"(total {stats.get('total', 0)})"
                    )
                files = item.get("files") or []
                if not files:
                    lines.append("  - Diff not available.")
                    continue
                if item.get("truncated_files"):
                    lines.append("  - Files (truncated):")
                else:
                    lines.append("  - Files:")
                for file_item in files:
                    filename = file_item.get("filename", "")
                    status = file_item.get("status", "")
                    additions = file_item.get("additions", 0)
                    deletions = file_item.get("deletions", 0)
                    truncated = file_item.get("patch_truncated")
                    lines.append(
                        f"    - {status}: {filename} (+{additions}/-{deletions})"
                    )
                    patch = (file_item.get("patch") or "").strip()
                    if patch:
                        lines.append("      ```diff")
                        lines.extend([f"      {line}" for line in patch.splitlines()])
                        lines.append("      ```")
                    if truncated:
                        lines.append("      - Diff truncated to fit prompt budget.")
    if errors:
        lines.append("")
        lines.append("## Notes")
        for err in errors:
            lines.append(f"- {err}")
    lines.append("")
    return "\n".join(lines)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    cfg = load_config("config.yaml")
    y_date, y_start, _ = date_range_for_yesterday()
    since_iso = y_start.isoformat().replace("+00:00", "Z")
    date_str = str(y_date)

    errors: List[str] = []

    try:
        arxiv_entries = fetch_arxiv(
            cfg["arxiv"]["query"], cfg["arxiv"].get("max_results", 50)
        )
    except Exception as exc:  # noqa: BLE001 - want a single fallback path
        arxiv_entries = []
        errors.append(f"arXiv fetch failed: {exc}")

    papers = filter_yesterday_arxiv(arxiv_entries, y_date)[
        : cfg["arxiv"].get("top_n", 15)
    ]

    gh_token = os.environ.get("GITHUB_TOKEN")
    commits: Dict[str, List[Dict[str, Any]]] = {}
    max_commits = int(cfg["github"].get("max_commits_per_repo", 8))
    max_files = int(cfg["github"].get("max_files_per_commit", 6))
    max_diff_chars = int(cfg["github"].get("max_diff_chars", 1200))
    max_total_diff_chars = int(cfg["github"].get("max_total_diff_chars", 0))
    for repo in cfg["github"].get("repos", []):
        try:
            repo_commits = fetch_github_commits(repo, since_iso, token=gh_token)[
                :max_commits
            ]
            for item in repo_commits:
                full_sha = item.get("full_sha", "")
                if not full_sha:
                    continue
                try:
                    detail = fetch_commit_detail(
                        repo, full_sha, gh_token, max_files, max_diff_chars
                    )
                    item["files"] = detail["files"]
                    item["stats"] = detail["stats"]
                    item["truncated_files"] = detail["truncated_files"]
                except Exception as exc:  # noqa: BLE001 - want a single fallback path
                    item["files"] = []
                    errors.append(
                        f"GitHub diff fetch failed for {repo}@{item.get('sha')}: {exc}"
                    )
            commits[repo] = repo_commits
        except Exception as exc:  # noqa: BLE001 - want a single fallback path
            commits[repo] = []
            errors.append(f"GitHub fetch failed for {repo}: {exc}")

    truncated_count = limit_diff_budget(commits, max_total_diff_chars)
    if truncated_count:
        errors.append("Diff content truncated to fit prompt budget.")

    report = ""
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            prompt = build_prompt(date_str, papers, commits)
            models = cfg.get("openrouter", {}).get("models")
            if not models:
                models = [cfg.get("openrouter", {}).get("model", "openrouter/auto")]
            retry_max = int(cfg["openrouter"].get("retry_max", 3))
            retry_base_seconds = int(cfg["openrouter"].get("retry_base_seconds", 10))
            for model in models:
                try:
                    report = openrouter_summarize(
                        prompt, model, retry_max, retry_base_seconds
                    )
                    if report:
                        break
                except Exception as exc:  # noqa: BLE001 - want a single fallback path
                    errors.append(f"OpenRouter summarize failed for {model}: {exc}")
        except Exception as exc:  # noqa: BLE001 - want a single fallback path
            errors.append(f"OpenRouter summarize failed: {exc}")

    if not report:
        report = render_fallback_markdown(date_str, papers, commits, errors)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report.strip() + "\n")

    with open("README.md", "w", encoding="utf-8") as handle:
        handle.write("# Daily AI Infra Report\n\n")
        handle.write(f"- Latest: [{date_str}](reports/{date_str}.md)\n")


if __name__ == "__main__":
    main()
