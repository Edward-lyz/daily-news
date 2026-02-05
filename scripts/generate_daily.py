import datetime
import json
import os
import time
from typing import Any, Dict, List, Tuple

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
    prompt: str, models: List[str], retry_max: int, retry_base_seconds: int
) -> str:
    key = os.environ["OPENROUTER_API_KEY"]
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    base_messages = [
        {
            "role": "system",
            "content": "You are a concise AI infra newsletter editor. Output Markdown only.",
        },
        {"role": "user", "content": prompt},
    ]
    attempts = max(1, retry_max)
    for attempt in range(attempts):
        if attempt == 0 or len(models) <= 1:
            model = models[0]
        else:
            fallback_models = models[1:]
            model = fallback_models[(attempt - 1) % len(fallback_models)]
        body = {
            "model": model,
            "messages": base_messages,
            "temperature": 0.2,
        }
        response = requests.post(url, headers=headers, json=body, timeout=60)
        if response.ok:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            time.sleep(60)
            return content
        detail = truncate_text(response.text.strip(), 2000)
        time.sleep(60)
        if attempt < attempts - 1:
            continue
        raise RuntimeError(
            f"OpenRouter {response.status_code} {response.reason} ({model}): {detail}"
        )
    raise RuntimeError("OpenRouter summarize failed: retry budget exhausted.")


def build_repo_prompt(
    date_str: str,
    repo: str,
    commits: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
) -> str:
    payload = {
        "date": date_str,
        "repo": repo,
        "commits": commits,
        "issues": issues,
        "requirements": [
            "只输出 Markdown，不要顶层标题。",
            "先写提交，再写 Issues；用清晰的短句。",
            "总结仓库改动时，每个要点必须附上对应的 GitHub PR 链接；没有 PR 链接就给出 commit 短 SHA，并附上 commit URL。",
            "涉及文件改动时，必须指出具体模块/文件名，必要时给简短代码片段。",
            "如果 diff 缺失或 patch_truncated 为 true，需要明确说明。",
            "如果 issue body 缺失或 body_truncated 为 true，需要明确说明。",
            "输出中文。专业术语保留英文。",
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def build_global_prompt(
    date_str: str,
    repo_summaries: List[Dict[str, str]],
    repos_without_updates: List[str],
) -> str:
    payload = {
        "date": date_str,
        "repo_summaries": repo_summaries,
        "repos_without_updates": repos_without_updates,
        "requirements": [
            "只输出 JSON，包含 summary 和 conclusion 两个字段，字段值为 Markdown 字符串。",
            "summary 用 4-6 句概括关键变化与影响。",
            "conclusion 需要给出风险/回归/待关注点（如有）。",
            "不要输出 '今日的 AI Infra 的新闻如下。' 这行。",
            "输出中文。专业术语保留英文。",
        ],
    }
    return json.dumps(payload, ensure_ascii=True)


def render_repo_fallback(
    repo: str,
    commits: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
) -> str:
    if not commits and not issues:
        return "昨日无更新。"
    lines: List[str] = []
    if commits:
        lines.append("提交：")
        for item in commits:
            sha = item.get("sha", "")
            msg = item.get("msg", "")
            url = item.get("url", "")
            lines.append(f"- {msg} [`{sha}`]({url})")
            files = item.get("files") or []
            if files:
                names = ", ".join([f.get("filename", "") for f in files if f.get("filename")])
                if names:
                    lines.append(f"- 受影响文件: {names}")
            if item.get("truncated_files"):
                lines.append("- 文件列表已截断。")
    if issues:
        lines.append("Issues：")
        for item in issues:
            title = item.get("title", "").strip()
            url = item.get("url", "").strip()
            lines.append(f"- {title} ({url})")
            if item.get("body_truncated"):
                lines.append("- Issue 内容已截断。")
    return "\n".join(lines)


def render_fallback_markdown(
    date_str: str,
    repo_sections: Dict[str, str],
    repos_without_updates: List[str],
    errors: List[str],
) -> str:
    lines: List[str] = []
    lines.append("今日的 AI Infra 的新闻如下。")
    lines.append("")
    lines.append("## 摘要")
    if errors:
        lines.append("模型摘要不可用，以下为原始数据整理。")
    else:
        lines.append("模型摘要不可用，以下为原始数据整理。")
    lines.append("")
    lines.append("## 具体内容分析")
    for repo, body in repo_sections.items():
        lines.append(f"### {repo}")
        if body:
            lines.append(body)
        else:
            lines.append("昨日无更新。")
    lines.append("")
    lines.append("## 总结")
    if errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append(
            f"- 有更新的仓库: {len(repo_sections) - len(repos_without_updates)}; 无更新的仓库: {len(repos_without_updates)}."
        )
    lines.append("")
    return "\n".join(lines)


def render_report(
    date_str: str,
    summary: str,
    conclusion: str,
    repo_sections: Dict[str, str],
    repos_without_updates: List[str],
    errors: List[str],
) -> str:
    lines: List[str] = []
    lines.append("今日的 AI Infra 的新闻如下。")
    lines.append("")
    lines.append("## 摘要")
    if summary:
        lines.append(summary.strip())
    else:
        lines.append("模型摘要不可用，以下为原始数据整理。")
    lines.append("")
    lines.append("## 具体内容分析")
    for repo, body in repo_sections.items():
        lines.append(f"### {repo}")
        if body:
            lines.append(body.strip())
        else:
            lines.append("昨日无更新。")
    lines.append("")
    lines.append("## 总结")
    if conclusion:
        lines.append(conclusion.strip())
    elif errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append(
            f"- 有更新的仓库: {len(repo_sections) - len(repos_without_updates)}; 无更新的仓库: {len(repos_without_updates)}."
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

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    models = cfg.get("openrouter", {}).get("models")
    if not models:
        models = [cfg.get("openrouter", {}).get("model", "openrouter/auto")]
    retry_max = int(cfg["openrouter"].get("retry_max", 3))
    retry_base_seconds = int(cfg["openrouter"].get("retry_base_seconds", 10))

    repo_sections: Dict[str, str] = {}
    repos_without_updates: List[str] = []
    for repo in cfg["github"].get("repos", []):
        repo_commits = commits.get(repo, [])
        repo_issues = issues.get(repo, [])
        if not repo_commits and not repo_issues:
            repos_without_updates.append(repo)
            repo_sections[repo] = "昨日无更新。"
            continue
        if openrouter_key:
            prompt = build_repo_prompt(date_str, repo, repo_commits, repo_issues)
            repo_summary = ""
            try:
                repo_summary = openrouter_summarize(
                    prompt, models, retry_max, retry_base_seconds
                )
            except Exception as exc:  # noqa: BLE001 - want a single fallback path
                errors.append(
                    f"OpenRouter repo summarize failed for {repo}: {exc}"
                )
            if repo_summary:
                repo_sections[repo] = repo_summary.strip()
                continue
        repo_sections[repo] = render_repo_fallback(repo, repo_commits, repo_issues)

    summary = ""
    conclusion = ""
    if openrouter_key:
        repo_payload = [
            {"repo": repo, "summary": body}
            for repo, body in repo_sections.items()
            if repo not in repos_without_updates
        ]
        prompt = build_global_prompt(date_str, repo_payload, repos_without_updates)
        global_output = ""
        try:
            global_output = openrouter_summarize(
                prompt, models, retry_max, retry_base_seconds
            )
        except Exception as exc:  # noqa: BLE001 - want a single fallback path
            errors.append(f"OpenRouter global summarize failed: {exc}")
        if global_output:
            try:
                data = json.loads(global_output)
                summary = (data.get("summary") or "").strip()
                conclusion = (data.get("conclusion") or "").strip()
            except json.JSONDecodeError:
                summary = global_output.strip()

    if summary or conclusion:
        report = render_report(
            date_str, summary, conclusion, repo_sections, repos_without_updates, errors
        )
    else:
        report = render_fallback_markdown(
            date_str, repo_sections, repos_without_updates, errors
        )

    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", f"{date_str}.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(report.strip() + "\n")

    with open("README.md", "w", encoding="utf-8") as handle:
        archive_dates = list_report_dates()
        handle.write(build_readme(date_str, report, archive_dates))


if __name__ == "__main__":
    main()
