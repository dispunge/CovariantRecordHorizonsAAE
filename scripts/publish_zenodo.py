#!/usr/bin/env python3
"""
Create and publish a new Zenodo version with a top-level compiled PDF
and clean source archive via the Zenodo Deposit REST API.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://zenodo.org/api"
USER_AGENT = (
    "CovariantRecordHorizons-ReleaseCI/1.0 "
    "(+https://github.com/dispunge/CovariantRecordHorizonsAAE)"
)
DEFAULT_TITLE = (
    "Record Leakage in Local Friendliness Tests: "
    "Randomness Trade-off, Two-Friend Bounds, and Noise Budgets"
)
DEFAULT_DESCRIPTION = (
    "Companion Python numerical optimization and circuit noise-budget suite "
    "(lf_leak_tests.py, lf_noise_budget.py, lf_dimension_checks.py) and Lean 4 "
    "machine-checked formal proofs (LFPolytope.lean, LFQuantumBounds.lean) for "
    '"Record Leakage in Local Friendliness Tests: Randomness Trade-off, '
    'Two-Friend Bounds, and Noise Budgets".'
)


def api_request(
    method: str,
    url: str,
    token: str | None = None,
    json_body: dict | None = None,
    raw_bytes: bytes | None = None,
    content_type: str | None = None,
) -> dict | list | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif raw_bytes is not None:
        data = raw_bytes
        headers["Content-Type"] = content_type or "application/octet-stream"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Zenodo API {method} {url} failed with HTTP {exc.code}: {err_text}"
        ) from exc


def resolve_latest_record_id(seed_record_id: str, token: str | None) -> tuple[str, str]:
    """Return (latest_published_record_id, conceptrecid) from any version or concept ID."""
    rec = api_request("GET", f"{API_BASE}/records/{seed_record_id}", token=token)
    assert isinstance(rec, dict)
    latest_url = rec.get("links", {}).get("latest")
    if latest_url:
        latest_rec = api_request("GET", latest_url, token=token)
        assert isinstance(latest_rec, dict)
        return str(latest_rec["id"]), str(latest_rec.get("conceptrecid", seed_record_id))
    return str(rec["id"]), str(rec.get("conceptrecid", seed_record_id))


def get_or_create_newversion_draft(
    latest_id: str, concept_id: str, token: str
) -> dict:
    """Create a new-version draft, or recover an existing unpublished draft if present."""
    try:
        nv = api_request(
            "POST",
            f"{API_BASE}/deposit/depositions/{latest_id}/actions/newversion",
            token=token,
            json_body={},
        )
        assert isinstance(nv, dict)
        draft_url = nv.get("links", {}).get("latest_draft") or (
            f"{API_BASE}/deposit/depositions/{nv['id']}"
        )
        draft = api_request("GET", draft_url, token=token)
        assert isinstance(draft, dict)
        return draft
    except RuntimeError as err:
        print(f"Note on newversion call ({err}); checking for existing open draft...")
        q = urllib.parse.urlencode({"status": "draft", "size": "20"})
        drafts = api_request("GET", f"{API_BASE}/deposit/depositions?{q}", token=token)
        if isinstance(drafts, list):
            for d in drafts:
                if str(d.get("conceptrecid")) == str(concept_id):
                    print(f"Recovered existing draft ID {d['id']} for concept {concept_id}.")
                    full_draft = api_request(
                        "GET",
                        f"{API_BASE}/deposit/depositions/{d['id']}",
                        token=token,
                    )
                    assert isinstance(full_draft, dict)
                    return full_draft
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", required=True, help="Seed or Concept Zenodo Record ID")
    parser.add_argument("--version", required=True, help="Release tag (e.g. v1.0.1)")
    parser.add_argument("--pdf", required=True, type=pathlib.Path, help="Compiled manuscript PDF")
    parser.add_argument("--source-zip", required=True, type=pathlib.Path, help="Clean git archive zip")
    parser.add_argument("--extra-file", action="append", default=[], type=pathlib.Path)
    parser.add_argument("--repo-url", default="https://github.com/dispunge/CovariantRecordHorizonsAAE")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Zenodo record title for the published version")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION, help="Zenodo record description")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs & API read access without publishing")
    args = parser.parse_args()

    for required_path in [args.pdf, args.source_zip, *args.extra_file]:
        if not required_path.is_file() or required_path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty release artifact: {required_path}")

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    latest_id, concept_id = resolve_latest_record_id(args.record_id.strip(), token or None)
    print(f"Resolved Zenodo seed {args.record_id} -> latest published ID {latest_id} (concept {concept_id})")

    if args.dry_run:
        if token:
            api_request("GET", f"{API_BASE}/deposit/depositions?size=1", token=token)
            print("Dry-run OK: ZENODO_TOKEN authenticated and local PDF/ZIP artifacts verified.")
        else:
            print("Dry-run OK: Local PDF/ZIP artifacts & public Zenodo record verified (ZENODO_TOKEN not set).")
        return 0

    if not token:
        raise RuntimeError("ZENODO_TOKEN environment variable is required to publish to Zenodo.")

    draft = get_or_create_newversion_draft(latest_id, concept_id, token)
    draft_id = draft["id"]
    print(f"Preparing Zenodo draft {draft_id}...")

    # 1. Remove inherited files (such as v1.0.0's zip-only archive)
    for old_file in draft.get("files", []):
        fid = old_file["id"]
        fname = old_file.get("filename", fid)
        print(f"Removing inherited file from previous version: {fname}")
        api_request(
            "DELETE",
            f"{API_BASE}/deposit/depositions/{draft_id}/files/{fid}",
            token=token,
        )

    # Refresh draft after file deletions to guarantee an active bucket URL
    draft = api_request("GET", f"{API_BASE}/deposit/depositions/{draft_id}", token=token)
    assert isinstance(draft, dict)
    bucket_url = draft["links"]["bucket"]

    # 2. Upload standalone PDF FIRST so Zenodo renders the inline PDF viewer at top of page
    uploads: list[tuple[str, pathlib.Path]] = [
        ("CovariantRecordHorizons.pdf", args.pdf),
        (args.source_zip.name, args.source_zip),
    ]
    for extra in args.extra_file:
        uploads.append((extra.name, extra))

    for remote_name, local_path in uploads:
        raw = local_path.read_bytes()
        encoded_name = urllib.parse.quote(remote_name)
        print(f"Uploading {remote_name} ({len(raw)} bytes) to Zenodo bucket...")
        api_request(
            "PUT",
            f"{bucket_url}/{encoded_name}",
            token=token,
            raw_bytes=raw,
            content_type="application/octet-stream",
        )

    # 3. Refresh draft and update version/date metadata while preserving prereserve_doi & creators
    refreshed = api_request("GET", f"{API_BASE}/deposit/depositions/{draft_id}", token=token)
    assert isinstance(refreshed, dict)
    meta = refreshed.get("metadata", {})
    clean_ver = args.version.lstrip("v")
    meta["version"] = clean_ver
    meta["publication_date"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    if args.title:
        meta["title"] = args.title
    if args.description:
        meta["description"] = args.description

    # Strip null subfields in creators/related_identifiers and read-only relations key
    meta.pop("relations", None)
    if isinstance(meta.get("creators"), list):
        meta["creators"] = [
            {k: v for k, v in c.items() if v is not None}
            for c in meta["creators"]
            if isinstance(c, dict)
        ]

    # Ensure clean GitHub tag link in related_identifiers without duplicating old tree links
    rel_ids = [
        {k: v for k, v in r.items() if v is not None}
        for r in meta.get("related_identifiers", [])
        if isinstance(r, dict) and "/tree/" not in r.get("identifier", "")
    ]
    rel_ids.append(
        {
            "identifier": f"{args.repo_url}/tree/{args.version}",
            "relation": "isSupplementedBy",
            "scheme": "url",
        }
    )
    meta["related_identifiers"] = rel_ids

    api_request(
        "PUT",
        f"{API_BASE}/deposit/depositions/{draft_id}",
        token=token,
        json_body={"metadata": meta},
    )

    # 4. Publish the new Zenodo version
    published = api_request(
        "POST",
        f"{API_BASE}/deposit/depositions/{draft_id}/actions/publish",
        token=token,
    )
    assert isinstance(published, dict)
    doi = published.get("doi") or published.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    record_url = published.get("links", {}).get(
        "record_html", f"https://zenodo.org/records/{published['id']}"
    )
    print(f"Successfully published Zenodo version {clean_ver}: DOI={doi} URL={record_url}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as sf:
            sf.write(
                f"### Zenodo Release Published\n"
                f"- **Version:** `{args.version}`\n"
                f"- **Version DOI:** `{doi}`\n"
                f"- **Zenodo Record:** {record_url}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
