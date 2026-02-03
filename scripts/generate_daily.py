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


def limit_issue_budget(
    issues: Dict[str, List[Dict[str, Any]]], max_total_chars: int
) -> int:
    if max_total_chars <= 0:
        return 0
    remaining = max_total_chars
    truncated = 0
    for repo_items in issues.values():
        for item in repo_items:
            body = item.get("body", "") or ""
            if not body:
                continue
            if remaining <= 0:
                item["body"] = ""
                item["body_truncated"] = True
                truncated += 1
                continue
            if len(body) > remaining:
                item["body"] = truncate_text(body, remaining)
                item["body_truncated"] = True
                truncated += 1
                remaining = 0
            else:
                remaining -= len(body)
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
                "patch_truncated": len(patch) > max_diff_chars,
            }
        )

    return {
        "files": files,
        "stats": data.get("stats") or {},
        "truncated_files": len(raw_files) > max_files,
    }


def fetch_github_issues(
    repo: str,
    since_iso: str,
    start: datetime.datetime,
    end: datetime.datetime,
    token: str | None = None,
) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"since": since_iso, "state": "all", "per_page": 100}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    out: List[Dict[str, Any]] = []
    for item in data:
        if "pull_request" in item:
            continue
        created_at = item.get("created_at") or ""
        try:
            created_dt = datetime.datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if not (start <= created_dt < end):
            continue
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("html_url", ""),
                "author": (item.get("user") or {}).get("login", ""),
                "created_at": created_at,
                "comments": item.get("comments", 0),
                "body": item.get("body") or "",
            }
        )
    return out


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
    issues: Dict[str, List[Dict[str, Any]]],
) -> str:
    payload = {
        "date": date_str,
        "papers": papers,
        "commits": commits,
        "issues": issues,
        "requirements": [
            "Return a Markdown report with three top-level sections: 摘要, 具体内容分析, 总结 (use ## headings).",
            "For papers: include a 1-line why it matters (AI infra angle).",
            "For repos: use file-level diffs in commits[*].files[*].patch to summarize changes.",
            "For repos: include a brief evaluation (impact/risk/regression) per repo.",
            "For issues: summarize newly created issues per repo and explain potential impact.",
            "If diffs are missing or patch_truncated is true, say so explicitly.",
            "If issue body is missing or body_truncated is true, say so explicitly.",
            "Write the report in Chinese.",
            "总结仓库的改动时，需要在你说出的每个总结的点上都附上对应的 github 的 PR 链接，方便跳转。没有 PR 链接就给出 commit 名称。",
            "总结不能过于潦草，需要严谨地指出，哪里组件/模块改动了什么具体细节，最好有简短的代码片段。",
            "开头必须是：今日的 AI Infra 的新闻如下。"
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def render_fallback_markdown(
    date_str: str,
    papers: List[Dict[str, Any]],
    commits: Dict[str, List[Dict[str, Any]]],
    issues: Dict[str, List[Dict[str, Any]]],
    errors: List[str],
) -> str:
    lines: List[str] = []
    lines.append(f"# AI Infra 日报 - {date_str}")
    lines.append("")
    lines.append("## 摘要")
    lines.append("模型摘要不可用，以下为原始数据整理。")
    lines.append("")
    lines.append("## 具体内容分析")
    lines.append("### 论文")
    if not papers:
        lines.append("- 未找到符合日期过滤的论文。")
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
    lines.append("### 代码提交")
    if not commits:
        lines.append("- 未配置仓库。")
    else:
        for repo, items in commits.items():
            lines.append(f"#### {repo}")
            if not items:
                lines.append("- 时间窗口内无提交。")
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
                    lines.append("  - 无法获取 diff。")
                    continue
                if item.get("truncated_files"):
                    lines.append("  - 文件列表（截断）：")
                else:
                    lines.append("  - 文件列表：")
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
                        lines.append("      - Diff 已截断以满足 prompt 预算。")
    lines.append("")
    lines.append("### 新增 Issues")
    if not issues:
        lines.append("- 未配置仓库。")
    else:
        for repo, items in issues.items():
            lines.append(f"#### {repo}")
            if not items:
                lines.append("- 时间窗口内无新增 issue。")
                continue
            for item in items:
                title = item.get("title", "").strip()
                url = item.get("url", "").strip()
                author = item.get("author", "")
                comments = item.get("comments", 0)
                header = f"- [{title}]({url})"
                if author:
                    header += f" ({author})"
                if comments:
                    header += f" - {comments} 条评论"
                lines.append(header)
                body = (item.get("body") or "").strip()
                if body:
                    lines.append(f"  - {body}")
                if item.get("body_truncated"):
                    lines.append("  - Issue 内容已截断以满足 prompt 预算。")
    lines.append("")
    lines.append("## 总结")
    if errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append(
            f"- 论文: {len(papers)}; 仓库: {len(commits)}; Issues: {sum(len(v) for v in issues.values())}."
        )
    lines.append("")
    return "\n".join(lines)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def list_report_dates(report_dir: str = "reports") -> List[str]:
    if not os.path.isdir(report_dir):
        return []
    dates: List[str] = []
    for name in os.listdir(report_dir):
        if not name.endswith(".md"):
            continue
        stem = name[:-3]
        try:
            datetime.date.fromisoformat(stem)
        except ValueError:
            continue
        dates.append(stem)
    return sorted(dates, reverse=True)


def normalize_report_for_readme(report: str) -> str:
    lines = report.strip().splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
        if lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def build_readme(date_str: str, report: str, archive_dates: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Daily AI Infra Report")
    lines.append("")
    lines.append("## 往期回顾")
    filtered_dates = [d for d in archive_dates if d != date_str]
    filtered_dates = filtered_dates[:7]
    if filtered_dates:
        for report_date in filtered_dates:
            lines.append(f"- [{report_date}](reports/{report_date}.md)")
    else:
        lines.append("- 暂无往期记录")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## 最新解读 ({date_str})")
    body = normalize_report_for_readme(report)
    if body:
        lines.append(body)
    else:
        lines.append("暂无内容")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config("config.yaml")
    y_date, y_start, y_end = date_range_for_yesterday()
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
    issues: Dict[str, List[Dict[str, Any]]] = {}
    max_issues = int(cfg["github"].get("max_issues_per_repo", 8))
    max_issue_body_chars = int(cfg["github"].get("max_issue_body_chars", 800))
    max_total_issue_body_chars = int(cfg["github"].get("max_total_issue_body_chars", 0))
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

    for repo in cfg["github"].get("repos", []):
        try:
            repo_issues = fetch_github_issues(
                repo, since_iso, y_start, y_end, token=gh_token
            )
            repo_issues.sort(key=lambda item: item.get("created_at", ""), reverse=True)
            repo_issues = repo_issues[:max_issues]
            for item in repo_issues:
                body = item.get("body") or ""
                item["body_truncated"] = False
                if body:
                    item["body"] = truncate_text(body, max_issue_body_chars)
                    item["body_truncated"] = len(body) > max_issue_body_chars
            issues[repo] = repo_issues
        except Exception as exc:  # noqa: BLE001 - want a single fallback path
            issues[repo] = []
            errors.append(f"GitHub issues fetch failed for {repo}: {exc}")

    truncated_count = limit_diff_budget(commits, max_total_diff_chars)
    if truncated_count:
        errors.append("Diff 内容已截断以满足 prompt 预算。")

    issue_truncated_count = limit_issue_budget(issues, max_total_issue_body_chars)
    if issue_truncated_count:
        errors.append("Issue 内容已截断以满足 prompt 预算。")

    report = ""
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            prompt = build_prompt(date_str, papers, commits, issues)
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
        report = render_fallback_markdown(date_str, papers, commits, issues, errors)

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report.strip() + "\n")

    with open("README.md", "w", encoding="utf-8") as handle:
        archive_dates = list_report_dates()
        handle.write(build_readme(date_str, report, archive_dates))


if __name__ == "__main__":
    main()
