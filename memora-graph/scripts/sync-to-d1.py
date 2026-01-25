#!/usr/bin/env python3
"""
One-way sync from memora SQLite database to Cloudflare D1.

This script exports memories and crossrefs from the local memora database
and syncs them to D1 for the web graph visualization.

Usage:
    python scripts/sync-to-d1.py           # Local D1 (development)
    python scripts/sync-to-d1.py --remote  # Remote D1 (production)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Add parent directory to path to import memora
SCRIPT_DIR = Path(__file__).resolve().parent
MEMORA_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MEMORA_ROOT))

try:
    from memora.backends import parse_backend_uri, CloudSQLiteBackend
except ImportError:
    print("Error: Could not import memora. Make sure it's installed or in PYTHONPATH.")
    print(f"Tried to import from: {MEMORA_ROOT}")
    sys.exit(1)


def get_source_backend(source_uri: str = None):
    """Get the source backend for syncing.

    If source_uri is provided, use it. Otherwise, try R2 from environment.
    """
    if source_uri:
        return parse_backend_uri(source_uri)

    # Default: try to get R2 URI from environment or .mcp.json
    r2_uri = os.getenv("MEMORA_R2_URI") or os.getenv("MEMORA_SOURCE_URI")
    if r2_uri:
        return parse_backend_uri(r2_uri)

    # Fallback: construct R2 URI from AWS settings
    endpoint = os.getenv("AWS_ENDPOINT_URL", "")
    if "r2.cloudflarestorage.com" in endpoint:
        return parse_backend_uri("s3://memora/memories.db")

    raise ValueError(
        "No source backend configured for sync.\n"
        "Set MEMORA_SOURCE_URI or MEMORA_R2_URI to your R2 bucket URI.\n"
        "Example: MEMORA_SOURCE_URI=s3://memora/memories.db"
    )


def escape_sql_string(s: str) -> str:
    """Escape a string for SQL insertion."""
    if s is None:
        return "NULL"
    # Replace single quotes with two single quotes
    escaped = s.replace("'", "''")
    return f"'{escaped}'"


def export_to_d1(remote: bool = False, source_uri: str = None, database: str = "memora-graph"):
    """Export memora data to D1."""
    print("Connecting to source database...")
    backend = get_source_backend(source_uri)
    print(f"Using backend: {backend.get_info().get('backend_type', 'unknown')}")
    print(f"Target D1 database: {database}")
    conn = backend.connect()

    try:
        # Get memories
        print("Fetching memories...")
        cursor = conn.execute(
            "SELECT id, content, metadata, tags, created_at, updated_at FROM memories"
        )
        memories = cursor.fetchall()
        print(f"Found {len(memories)} memories")

        # Get crossrefs
        print("Fetching crossrefs...")
        cursor = conn.execute("SELECT memory_id, related FROM memories_crossrefs")
        crossrefs = cursor.fetchall()
        print(f"Found {len(crossrefs)} crossrefs")

        # Generate SQL file
        print("Generating SQL...")
        sql_lines = [
            "-- Auto-generated sync from memora to D1",
            "-- WARNING: This will replace all data in D1",
            "",
            "-- Clear existing data",
            "DELETE FROM memories_crossrefs;",
            "DELETE FROM memories;",
            "",
            "-- Insert memories",
        ]

        for row in memories:
            mem_id, content, metadata, tags, created_at, updated_at = row

            # Ensure JSON fields are valid
            if metadata:
                try:
                    json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = "{}"
            else:
                metadata = "{}"

            if tags:
                try:
                    json.loads(tags)
                except json.JSONDecodeError:
                    tags = "[]"
            else:
                tags = "[]"

            sql_lines.append(
                f"INSERT INTO memories (id, content, metadata, tags, created_at, updated_at) "
                f"VALUES ({mem_id}, {escape_sql_string(content)}, {escape_sql_string(metadata)}, "
                f"{escape_sql_string(tags)}, {escape_sql_string(created_at)}, "
                f"{escape_sql_string(updated_at) if updated_at else 'NULL'});"
            )

        sql_lines.append("")
        sql_lines.append("-- Insert crossrefs")

        for row in crossrefs:
            memory_id, related = row

            # Ensure related is valid JSON
            if related:
                try:
                    json.loads(related)
                except json.JSONDecodeError:
                    related = "[]"
            else:
                related = "[]"

            sql_lines.append(
                f"INSERT INTO memories_crossrefs (memory_id, related) "
                f"VALUES ({memory_id}, {escape_sql_string(related)});"
            )

        # Write SQL to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".sql", delete=False
        ) as f:
            f.write("\n".join(sql_lines))
            sql_file = f.name

        print(f"Generated {len(sql_lines)} SQL statements")
        print(f"SQL file: {sql_file}")

        # Apply to D1 using wrangler
        print(f"\nApplying to D1 ({'remote' if remote else 'local'})...")
        cmd = ["npx", "wrangler", "d1", "execute", database, f"--file={sql_file}"]
        if remote:
            cmd.append("--remote")
        else:
            cmd.append("--local")

        # Run from the memora-graph directory
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR.parent,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"Error: wrangler command failed")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            sys.exit(1)

        print(result.stdout)
        print("\nSync complete!")
        print(f"  Memories: {len(memories)}")
        print(f"  Crossrefs: {len(crossrefs)}")

        # Clean up temp file
        os.unlink(sql_file)

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Sync memora data to Cloudflare D1"
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Sync to remote D1 (production). Default is local D1.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source backend URI (e.g., s3://memora/memories.db). Default: auto-detect from environment.",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="memora-graph",
        help="Target D1 database name. Default: memora-graph.",
    )
    args = parser.parse_args()

    export_to_d1(remote=args.remote, source_uri=args.source, database=args.database)


if __name__ == "__main__":
    main()
