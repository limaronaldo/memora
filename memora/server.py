"""MCP-compatible memory server backed by SQLite."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .cloud_sync import schedule_sync as _schedule_cloud_graph_sync
from .storage import (
    DEFAULT_WORKSPACE,
    _redact_secrets,
    add_identity_alias,
    add_link,
    add_memories,
    add_memory,
    boost_memory,
    cleanup_expired_memories,
    clear_events,
    collect_all_tags,
    connect,
    content_preview,
    create_identity,
    delete_identity,
    delete_memories,
    delete_memory,
    delete_session,
    delete_workspace,
    detect_clusters,
    export_memories,
    find_invalid_tag_entries,
    get_crossrefs,
    get_identities_in_memory,
    get_identity,
    get_memories_by_identity,
    get_memory,
    get_session,
    get_statistics,
    get_workspace_stats,
    hybrid_search,
    import_memories,
    index_conversation,
    index_conversation_delta,
    link_memory_to_identity,
    list_identities,
    list_memories,
    list_sessions,
    list_workspaces,
    move_memories_to_workspace,
    poll_events,
    rebuild_crossrefs,
    rebuild_embeddings,
    remove_link,
    search_identities,
    search_sessions,
    semantic_search,
    soft_trim,
    sync_to_cloud,
    unlink_memory_from_identity,
    update_crossrefs,
    update_identity,
    update_memory,
)

# Content type inference patterns
TYPE_PATTERNS: List[tuple[str, str]] = [
    (r"^(?:TODO|TASK)[:>\s]", "todo"),
    (r"^(?:BUG|ISSUE|FIX|ERROR)[:>\s]", "issue"),
    (r"^(?:NOTE|TIP|INFO)[:>\s]", "note"),
    (r"^(?:IDEA|FEATURE|ENHANCEMENT)[:>\s]", "idea"),
    (r"^(?:QUESTION|\?)[:>\s]", "question"),
    (r"^(?:WARN|WARNING|CAUTION)[:>\s]", "warning"),
]

# Duplicate detection threshold
DUPLICATE_THRESHOLD = 0.85


def _infer_type(content: str) -> Optional[str]:
    """Infer memory type from content prefix patterns."""
    for pattern, type_name in TYPE_PATTERNS:
        if re.match(pattern, content, re.IGNORECASE):
            return type_name
    return None


def _suggest_tags(content: str, inferred_type: Optional[str]) -> List[str]:
    """Suggest tags based on content and inferred type."""
    suggestions = []

    # Type-based suggestions
    if inferred_type == "todo":
        suggestions.append("memora/todos")
    elif inferred_type == "issue":
        suggestions.append("memora/issues")
    elif inferred_type in ("note", "idea", "question"):
        suggestions.append("memora/knowledge")

    return suggestions


from .graph import (
    export_graph_html,
    register_graph_routes,
    start_graph_server,
)


def _read_int_env(var_name: str, fallback: int) -> int:
    try:
        return int(os.getenv(var_name, fallback))
    except (TypeError, ValueError):
        return fallback


VALID_TRANSPORTS = {"stdio", "sse", "streamable-http"}

_env_transport = os.getenv("MEMORA_TRANSPORT", "stdio")
DEFAULT_TRANSPORT = _env_transport if _env_transport in VALID_TRANSPORTS else "stdio"
DEFAULT_HOST = os.getenv("MEMORA_HOST", "127.0.0.1")
DEFAULT_PORT = _read_int_env("MEMORA_PORT", 8000)
DEFAULT_GRAPH_PORT = _read_int_env("MEMORA_GRAPH_PORT", 8765)

mcp = FastMCP("Memory MCP Server", host=DEFAULT_HOST, port=DEFAULT_PORT)

# Register graph visualization routes
register_graph_routes(mcp)


def _with_connection(func=None, *, writes=False):
    """Decorator that manages database connections and cloud sync.

    Opens a connection, runs the function, closes the connection,
    and syncs to cloud storage only after write operations.

    Args:
        writes: If True, syncs to cloud after operation. If False, skips sync (read-only).
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            conn = connect()
            try:
                result = func(conn, *args, **kwargs)
                # Only sync to cloud after write operations
                if writes:
                    sync_to_cloud()
                    # Broadcast update to connected clients
                    from .cloud_sync import sync_now

                    sync_now()
                return result
            finally:
                conn.close()

        return wrapper

    # Allow using as @_with_connection or @_with_connection(writes=True)
    if func is not None:
        # Called as @_with_connection (default: read-only, no sync)
        return decorator(func)
    else:
        # Called as @_with_connection(writes=True)
        return decorator


@_with_connection(writes=True)
def _create_memory(
    conn,
    content: str,
    metadata: Optional[Dict[str, Any]],
    tags: Optional[list[str]],
    tier: Optional[str] = None,
    expires_at: Optional[str] = None,
    workspace: Optional[str] = None,
):
    return add_memory(
        conn,
        content=content.strip(),
        metadata=metadata,
        tags=tags or [],
        tier=tier,
        expires_at=expires_at,
        workspace=workspace,
    )


@_with_connection
def _get_memory(conn, memory_id: int):
    return get_memory(conn, memory_id)


@_with_connection(writes=True)
def _update_memory(
    conn,
    memory_id: int,
    content: Optional[str],
    metadata: Optional[Dict[str, Any]],
    tags: Optional[list[str]],
    tier: Optional[str] = None,
    expires_at: Optional[str] = None,
):
    return update_memory(
        conn,
        memory_id,
        content=content,
        metadata=metadata,
        tags=tags,
        tier=tier,
        expires_at=expires_at,
    )


@_with_connection(writes=True)
def _delete_memory(conn, memory_id: int):
    return delete_memory(conn, memory_id)


@_with_connection
def _list_memories(
    conn,
    query: Optional[str],
    metadata_filters: Optional[Dict[str, Any]],
    limit: Optional[int],
    offset: Optional[int],
    date_from: Optional[str],
    date_to: Optional[str],
    tags_any: Optional[List[str]],
    tags_all: Optional[List[str]],
    tags_none: Optional[List[str]],
    sort_by_importance: bool = False,
    workspace: Optional[str] = None,
    workspaces: Optional[List[str]] = None,
):
    return list_memories(
        conn,
        query,
        metadata_filters,
        limit,
        offset,
        date_from,
        date_to,
        tags_any,
        tags_all,
        tags_none,
        sort_by_importance=sort_by_importance,
        workspace=workspace,
        workspaces=workspaces,
    )


@_with_connection(writes=True)
def _boost_memory(conn, memory_id: int, boost_amount: float):
    return boost_memory(conn, memory_id, boost_amount)


@_with_connection(writes=True)
def _create_memories(conn, entries: List[Dict[str, Any]]):
    return add_memories(conn, entries)


@_with_connection(writes=True)
def _delete_memories(conn, ids: List[int]):
    return delete_memories(conn, ids)


@_with_connection
def _collect_tags(conn):
    return collect_all_tags(conn)


@_with_connection
def _find_invalid_tags(conn):
    from . import TAG_WHITELIST

    return find_invalid_tag_entries(conn, TAG_WHITELIST)


@_with_connection(writes=True)  # May write crossrefs if refresh=True
def _get_related(conn, memory_id: int, refresh: bool) -> List[Dict[str, Any]]:
    if refresh:
        update_crossrefs(conn, memory_id)
    refs = get_crossrefs(conn, memory_id)
    if not refs and not refresh:
        update_crossrefs(conn, memory_id)
        refs = get_crossrefs(conn, memory_id)
    return refs


@_with_connection(writes=True)
def _rebuild_crossrefs(conn):
    return rebuild_crossrefs(conn)


@_with_connection
def _semantic_search(
    conn,
    query: str,
    metadata_filters: Optional[Dict[str, Any]],
    top_k: Optional[int],
    min_score: Optional[float],
):
    return semantic_search(
        conn,
        query,
        metadata_filters=metadata_filters,
        top_k=top_k,
        min_score=min_score,
    )


@_with_connection
def _hybrid_search(
    conn,
    query: str,
    semantic_weight: float,
    fusion_method: str,
    top_k: int,
    min_score: float,
    metadata_filters: Optional[Dict[str, Any]],
    date_from: Optional[str],
    date_to: Optional[str],
    tags_any: Optional[List[str]],
    tags_all: Optional[List[str]],
    tags_none: Optional[List[str]],
):
    return hybrid_search(
        conn,
        query,
        semantic_weight=semantic_weight,
        fusion_method=fusion_method,
        top_k=top_k,
        min_score=min_score,
        metadata_filters=metadata_filters,
        date_from=date_from,
        date_to=date_to,
        tags_any=tags_any,
        tags_all=tags_all,
        tags_none=tags_none,
    )


@_with_connection(writes=True)
def _rebuild_embeddings(conn):
    return rebuild_embeddings(conn)


@_with_connection
def _get_statistics(conn):
    return get_statistics(conn)


@_with_connection
def _export_memories(conn):
    return export_memories(conn)


@_with_connection(writes=True)
def _import_memories(conn, data: List[Dict[str, Any]], strategy: str):
    return import_memories(conn, data, strategy)


def _build_tag_hierarchy(tags):
    root = {"name": "root", "path": [], "children": {}, "tags": []}
    for tag in tags:
        parts = tag.split(".")
        node = root
        if not parts:
            continue
        for idx, part in enumerate(parts):
            children = node.setdefault("children", {})
            if part not in children:
                children[part] = {
                    "name": part,
                    "path": node["path"] + [part],
                    "children": {},
                    "tags": [],
                }
            node = children[part]
        node.setdefault("tags", []).append(tag)
    return _collapse_tag_tree(root)


def _collapse_tag_tree(node):
    children_map = node.get("children", {})
    children_list = [_collapse_tag_tree(child) for child in children_map.values()]
    node["children"] = children_list
    node["count"] = len(node.get("tags", [])) + sum(
        child["count"] for child in children_list
    )
    return {key: value for key, value in node.items() if key != "children" or value}


def _extract_hierarchy_path(metadata: Optional[Any]) -> List[str]:
    if not isinstance(metadata, Mapping):
        return []

    hierarchy = metadata.get("hierarchy")
    if isinstance(hierarchy, Mapping):
        raw_path = hierarchy.get("path")
        if isinstance(raw_path, Sequence) and not isinstance(raw_path, (str, bytes)):
            return [str(part) for part in raw_path if part is not None]

    path: List[str] = []
    section = metadata.get("section")
    if section is not None:
        path.append(str(section))
        subsection = metadata.get("subsection")
        if subsection is not None:
            path.append(str(subsection))
    return path


def _suggest_hierarchy_from_similar(
    similar_memories: List[Dict[str, Any]],
    max_suggestions: int = 3,
) -> List[Dict[str, Any]]:
    """Suggest hierarchy placement based on where similar memories are organized.

    Args:
        similar_memories: List of similar memory results with scores
        max_suggestions: Maximum number of hierarchy suggestions

    Returns:
        List of hierarchy suggestions with paths, scores, and example memory IDs
    """
    # Count hierarchy paths from similar memories, weighted by similarity score
    path_scores: Dict[tuple, float] = {}
    path_examples: Dict[tuple, List[int]] = {}

    for item in similar_memories:
        if not item:
            continue
        memory_id = item.get("id")
        score = item.get("score", 0)

        # Get full memory to extract hierarchy
        full_memory = _get_memory(memory_id)
        if not full_memory:
            continue

        path = _extract_hierarchy_path(full_memory.get("metadata"))
        if not path:
            continue

        path_tuple = tuple(path)
        path_scores[path_tuple] = path_scores.get(path_tuple, 0) + score
        if path_tuple not in path_examples:
            path_examples[path_tuple] = []
        path_examples[path_tuple].append(memory_id)

    if not path_scores:
        return []

    # Sort by weighted score
    sorted_paths = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)

    suggestions = []
    for path_tuple, total_score in sorted_paths[:max_suggestions]:
        path_list = list(path_tuple)
        suggestions.append(
            {
                "path": path_list,
                "section": path_list[0] if path_list else None,
                "subsection": "/".join(path_list[1:]) if len(path_list) > 1 else None,
                "confidence": round(total_score / len(similar_memories), 2),
                "similar_memory_ids": path_examples[path_tuple][:3],
            }
        )

    return suggestions


def _compact_memory(
    memory: Optional[Dict[str, Any]], preview_length: int = 200
) -> Optional[Dict[str, Any]]:
    """Return a compact representation of a memory (id, preview, tags, content_length).

    Args:
        memory: The memory dict to compact
        preview_length: Max length for content preview (default: 200 chars)

    Returns:
        Compact dict with id, preview, content_length, tags, created_at
    """
    if memory is None:
        return None
    content = memory.get("content", "")
    return {
        "id": memory.get("id"),
        "preview": content_preview(content, preview_length),
        "content_length": len(content),
        "tags": memory.get("tags", []),
        "created_at": memory.get("created_at"),
    }


def _get_existing_hierarchy_paths() -> List[List[str]]:
    """Get all unique hierarchy paths from existing memories."""
    items = _list_memories(None, None, None, 0, None, None, None, None, None)
    paths_set: set = set()
    for memory in items:
        if memory is None:
            continue
        path = _extract_hierarchy_path(memory.get("metadata"))
        if path:
            # Add the full path and all parent paths
            for i in range(1, len(path) + 1):
                paths_set.add(tuple(path[:i]))
    return sorted([list(p) for p in paths_set], key=lambda x: (len(x), x))


def _find_similar_paths(
    new_path: List[str], existing_paths: List[List[str]]
) -> List[List[str]]:
    """Find existing paths similar to new_path - siblings or paths with similar names."""
    if not new_path or not existing_paths:
        return []

    suggestions = []
    new_path_lower = [p.lower() for p in new_path]
    new_parent = new_path[:-1] if len(new_path) > 1 else []
    new_leaf = new_path_lower[-1] if new_path else ""

    for existing in existing_paths:
        existing_lower = [p.lower() for p in existing]
        existing_parent = existing[:-1] if len(existing) > 1 else []
        existing_leaf = existing_lower[-1] if existing else ""

        # Priority 1: Same parent, different leaf (siblings)
        if existing_parent == new_parent and existing_lower != new_path_lower:
            # Check if leaf names are similar (substring match)
            if new_leaf in existing_leaf or existing_leaf in new_leaf:
                if existing not in suggestions:
                    suggestions.insert(0, existing)  # High priority
                continue

        # Priority 2: Same parent (show siblings)
        if existing_parent == new_parent and existing not in suggestions:
            suggestions.append(existing)
            continue

        # Priority 3: Substring match in leaf (e.g., "background" matches "2. background")
        if new_leaf and existing_leaf:
            if new_leaf in existing_leaf or existing_leaf in new_leaf:
                if existing not in suggestions:
                    suggestions.append(existing)

    return suggestions[:5]  # Limit to 5 suggestions


def _build_hierarchy_tree(
    memories: List[Dict[str, Any]], include_root: bool = False, compact: bool = True
) -> Any:
    root: Dict[str, Any] = {
        "name": "root",
        "path": [],
        "memories": [],
        "children": {},
    }

    for memory in memories:
        path = _extract_hierarchy_path(memory.get("metadata"))
        node = root
        if not path:
            mem_data = _compact_memory(memory) if compact else dict(memory)
            if not compact:
                mem_data["hierarchy_path"] = node["path"]
            node["memories"].append(mem_data)
            continue

        for part in path:
            children: Dict[str, Any] = node.setdefault("children", {})
            if part not in children:
                children[part] = {
                    "name": part,
                    "path": node["path"] + [part],
                    "memories": [],
                    "children": {},
                }
            node = children[part]
        mem_data = _compact_memory(memory) if compact else dict(memory)
        if not compact:
            mem_data["hierarchy_path"] = node["path"]
        node["memories"].append(mem_data)

    def collapse(node: Dict[str, Any]) -> Dict[str, Any]:
        children_map: Dict[str, Any] = node.get("children", {})
        children_list = [collapse(child) for child in children_map.values()]
        node["children"] = children_list
        node["count"] = len(node.get("memories", [])) + sum(
            child["count"] for child in children_list
        )
        return node

    collapsed = collapse(root)
    if include_root:
        return collapsed
    return collapsed["children"]


@mcp.tool()
async def memory_create(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    suggest_similar: bool = True,
    similarity_threshold: float = 0.2,
    tier: Optional[str] = None,
    expires_at: Optional[str] = None,
    workspace: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new memory entry.

    Args:
        content: The memory content text
        metadata: Optional metadata dictionary
        tags: Optional list of tags
        suggest_similar: If True, find similar memories and suggest consolidation (default: True)
        similarity_threshold: Minimum similarity score for suggestions (default: 0.2)
        tier: Memory tier - "daily" (auto-expires) or "permanent" (default)
        expires_at: Expiration datetime for daily memories (ISO format)
        workspace: Workspace to create memory in (default: "default")
    """
    # Check hierarchy path BEFORE creating to detect new paths
    new_path = _extract_hierarchy_path(metadata)
    existing_paths = _get_existing_hierarchy_paths() if new_path else []
    path_is_new = bool(new_path) and (new_path not in existing_paths)

    # Initialize warnings dict
    warnings: Dict[str, Any] = {}

    # Auto-redact secrets/PII from content BEFORE saving
    redacted_content = content.strip()
    try:
        redacted_content, secrets_redacted = _redact_secrets(redacted_content)
        if secrets_redacted:
            warnings["secrets_redacted"] = secrets_redacted
    except Exception:
        pass  # Don't fail on redaction errors

    try:
        record = _create_memory(
            content=redacted_content,
            metadata=metadata,
            tags=tags or [],
            tier=tier,
            expires_at=expires_at,
            workspace=workspace,
        )
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}

    result: Dict[str, Any] = {"memory": record}

    # Warn if a new hierarchy path was created and suggest similar existing paths
    if path_is_new:
        similar = _find_similar_paths(new_path, existing_paths)
        if similar:
            warnings["new_hierarchy_path"] = f"New hierarchy path created: {new_path}"
            result["existing_similar_paths"] = similar
            result["hint"] = (
                "Did you mean to use one of these existing paths? Use memory_update to change if needed."
            )

    # Use cross-refs (related memories) for consolidation hints and duplicate detection
    # Cross-refs use full embedding context (content + metadata + tags) so are more accurate
    related_memories = record.get("related", []) if record else []
    if suggest_similar and related_memories:
        # Filter by threshold
        above_threshold = [
            m
            for m in related_memories
            if m and m.get("score", 0) >= similarity_threshold
        ]
        if above_threshold:
            result["similar_memories"] = above_threshold
            result["consolidation_hint"] = (
                f"Found {len(above_threshold)} similar memories. "
                "Consider: (1) merge content with memory_update, or (2) delete redundant ones with memory_delete."
            )
            # Check for potential duplicates (>0.85 similarity)
            duplicates = [
                m for m in above_threshold if m.get("score", 0) >= DUPLICATE_THRESHOLD
            ]
            if duplicates:
                warnings["duplicate_warning"] = (
                    f"Very similar memory exists (>={int(DUPLICATE_THRESHOLD * 100)}% match). "
                    f"Memory #{duplicates[0]['id']} has {int(duplicates[0]['score'] * 100)}% similarity."
                )

    # Add warnings to result if any
    if warnings:
        result["warnings"] = warnings

    # Infer type and suggest tags (only if user didn't provide tags)
    try:
        suggestions: Dict[str, Any] = {}
        inferred_type = _infer_type(redacted_content)
        if inferred_type:
            suggestions["type"] = inferred_type

        suggested_tags = _suggest_tags(redacted_content, inferred_type)
        # Only suggest tags not already applied
        existing_tags = set(tags or [])
        new_suggestions = [t for t in suggested_tags if t not in existing_tags]
        if new_suggestions:
            suggestions["tags"] = new_suggestions

        # Suggest hierarchy placement based on related memories (cross-refs)
        # (only if user didn't provide a hierarchy path)
        if not new_path and related_memories:
            hierarchy_suggestions = _suggest_hierarchy_from_similar(related_memories)
            if hierarchy_suggestions:
                suggestions["hierarchy"] = hierarchy_suggestions
                suggestions["hierarchy_hint"] = (
                    "Similar memories are organized under these paths. "
                    "Use memory_update to add section/subsection metadata."
                )

        if suggestions:
            result["suggestions"] = suggestions
    except Exception:
        pass  # Don't fail on type inference errors

    _schedule_cloud_graph_sync()
    return result


@mcp.tool()
async def memory_create_issue(
    content: str,
    status: str = "open",
    closed_reason: Optional[str] = None,
    severity: str = "minor",
    component: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new issue/bug memory.

    Args:
        content: Description of the issue
        status: Issue status - "open" (default) or "closed"
        closed_reason: If closed, the reason - "complete" or "not_planned"
        severity: Issue severity - "critical", "major", "minor" (default)
        component: Component/area affected (e.g., "graph", "storage", "api")
        category: Issue category (e.g., "bug", "enhancement", "performance")

    Returns:
        Created issue memory with auto-assigned tag "memora/issues"
    """
    # Validate status
    valid_statuses = {"open", "closed"}
    if status not in valid_statuses:
        return {
            "error": "invalid_status",
            "message": f"Status must be one of: {', '.join(valid_statuses)}",
        }

    # Validate closed_reason if status is closed
    if status == "closed":
        valid_reasons = {"complete", "not_planned"}
        if not closed_reason:
            return {
                "error": "missing_closed_reason",
                "message": "closed_reason required when status is 'closed'",
            }
        if closed_reason not in valid_reasons:
            return {
                "error": "invalid_closed_reason",
                "message": f"closed_reason must be one of: {', '.join(valid_reasons)}",
            }

    # Validate severity
    valid_severities = {"critical", "major", "minor"}
    if severity not in valid_severities:
        return {
            "error": "invalid_severity",
            "message": f"Severity must be one of: {', '.join(valid_severities)}",
        }

    # Build metadata
    metadata: Dict[str, Any] = {
        "type": "issue",
        "status": status,
        "severity": severity,
    }
    if closed_reason:
        metadata["closed_reason"] = closed_reason
    if component:
        metadata["component"] = component
    if category:
        metadata["category"] = category

    # Create with auto-tag
    tags = ["memora/issues"]

    try:
        record = _create_memory(content.strip(), metadata, tags)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}

    _schedule_cloud_graph_sync()
    return {"memory": record}


@mcp.tool()
async def memory_create_todo(
    content: str,
    status: str = "open",
    closed_reason: Optional[str] = None,
    priority: str = "medium",
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new TODO/task memory.

    Args:
        content: Description of the task
        status: Task status - "open" (default) or "closed"
        closed_reason: If closed, the reason - "complete" or "not_planned"
        priority: Task priority - "high", "medium" (default), "low"
        category: Task category (e.g., "cloud-backend", "graph-visualization", "docs")

    Returns:
        Created TODO memory with auto-assigned tag "memora/todos"
    """
    # Validate status
    valid_statuses = {"open", "closed"}
    if status not in valid_statuses:
        return {
            "error": "invalid_status",
            "message": f"Status must be one of: {', '.join(valid_statuses)}",
        }

    # Validate closed_reason if status is closed
    if status == "closed":
        valid_reasons = {"complete", "not_planned"}
        if not closed_reason:
            return {
                "error": "missing_closed_reason",
                "message": "closed_reason required when status is 'closed'",
            }
        if closed_reason not in valid_reasons:
            return {
                "error": "invalid_closed_reason",
                "message": f"closed_reason must be one of: {', '.join(valid_reasons)}",
            }

    # Validate priority
    valid_priorities = {"high", "medium", "low"}
    if priority not in valid_priorities:
        return {
            "error": "invalid_priority",
            "message": f"Priority must be one of: {', '.join(valid_priorities)}",
        }

    # Build metadata
    metadata: Dict[str, Any] = {
        "type": "todo",
        "status": status,
        "priority": priority,
    }
    if closed_reason:
        metadata["closed_reason"] = closed_reason
    if category:
        metadata["category"] = category

    # Create with auto-tag
    tags = ["memora/todos"]

    try:
        record = _create_memory(content.strip(), metadata, tags)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}

    _schedule_cloud_graph_sync()
    return {"memory": record}


@mcp.tool()
async def memory_create_section(
    content: str,
    section: Optional[str] = None,
    subsection: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new section/subsection header memory.

    Section memories are organizational placeholders that:
    - Are NOT visible in the graph visualization
    - Are NOT included in duplicate detection
    - Do NOT compute embeddings or cross-references

    Args:
        content: Title/description of the section
        section: Parent section name (e.g., "Architecture", "API")
        subsection: Subsection path (e.g., "endpoints/auth")

    Returns:
        Created section memory with auto-assigned tag "memora/sections"
    """
    # Build metadata
    metadata: Dict[str, Any] = {
        "type": "section",
    }
    if section:
        metadata["section"] = section
    if subsection:
        metadata["subsection"] = subsection

    # Create with auto-tag
    tags = ["memora/sections"]

    try:
        record = _create_memory(content.strip(), metadata, tags)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}

    _schedule_cloud_graph_sync()
    return {"memory": record}


@mcp.tool()
async def memory_create_daily(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    ttl_hours: int = 24,
) -> Dict[str, Any]:
    """Create a daily/ephemeral memory that auto-expires.

    Daily memories are ideal for:
    - Session context and scratch notes
    - Temporary task state
    - Working memory that shouldn't persist long-term

    Args:
        content: The memory content text
        metadata: Optional metadata dictionary
        tags: Optional list of tags
        ttl_hours: Hours until expiration (default: 24). Set to 0 for no expiration.

    Returns:
        Created memory with tier="daily" and expires_at set

    Raises:
        ValueError: If ttl_hours is negative
    """
    from datetime import datetime, timedelta

    # Validate ttl_hours
    if ttl_hours < 0:
        raise ValueError(f"ttl_hours must be >= 0, got {ttl_hours}")

    # Calculate expiration time
    if ttl_hours > 0:
        expires_at = (datetime.utcnow() + timedelta(hours=ttl_hours)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
    else:
        expires_at = None

    # Add daily-specific metadata
    daily_metadata = dict(metadata) if metadata else {}
    daily_metadata["type"] = "daily"

    try:
        record = _create_memory(
            content=content.strip(),
            metadata=daily_metadata,
            tags=tags or [],
            tier="daily",
            expires_at=expires_at,
        )
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}

    result: Dict[str, Any] = {"memory": record}
    if expires_at:
        result["expires_in_hours"] = ttl_hours
        result["hint"] = (
            f"This memory will expire at {expires_at}. "
            "Use memory_promote to make it permanent if needed."
        )

    _schedule_cloud_graph_sync()
    return result


@mcp.tool()
async def memory_checkpoint(
    summary: str,
    key_facts: Optional[List[str]] = None,
    context_source: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create checkpoint memories to preserve important context before compaction.

    Use this tool proactively when:
    - Context is approaching capacity limits
    - Before ending a long session
    - To preserve key decisions, facts, or action items
    - Before switching to a different task

    Args:
        summary: High-level summary of current context/session
        key_facts: List of specific facts, decisions, or action items to preserve
        context_source: Source identifier (e.g., "conversation", "session", "task")
        session_id: Optional session identifier for grouping related checkpoints

    Returns:
        Dictionary with created checkpoint memories
    """
    from datetime import date, datetime

    created_memories: List[Dict[str, Any]] = []
    checkpoint_date = date.today().isoformat()
    checkpoint_time = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # Build base metadata for all checkpoint memories
    base_metadata: Dict[str, Any] = {
        "type": "checkpoint",
        "checkpoint_at": checkpoint_time,
    }
    if context_source:
        base_metadata["source"] = context_source
    if session_id:
        base_metadata["session_id"] = session_id

    # Create summary memory
    summary_metadata = dict(base_metadata)
    summary_metadata["checkpoint_type"] = "summary"

    try:
        summary_record = _create_memory(
            content=summary.strip(),
            metadata=summary_metadata,
            tags=["memora/checkpoints", f"checkpoint/{checkpoint_date}"],
        )
        created_memories.append(summary_record)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc), "field": "summary"}

    # Create individual fact memories
    if key_facts:
        for i, fact in enumerate(key_facts):
            if not fact or not fact.strip():
                continue

            fact_metadata = dict(base_metadata)
            fact_metadata["checkpoint_type"] = "fact"
            fact_metadata["fact_index"] = i

            try:
                fact_record = _create_memory(
                    content=fact.strip(),
                    metadata=fact_metadata,
                    tags=["memora/checkpoints", f"checkpoint/{checkpoint_date}"],
                )
                created_memories.append(fact_record)
            except ValueError:
                # Skip invalid facts but continue with others
                continue

    if created_memories:
        _schedule_cloud_graph_sync()

    return {
        "checkpoint_created": True,
        "checkpoint_date": checkpoint_date,
        "checkpoint_time": checkpoint_time,
        "memories_created": len(created_memories),
        "memories": created_memories,
        "hint": (
            f"Created {len(created_memories)} checkpoint memories. "
            f"Find them later with: memory_list(tags_any=['checkpoint/{checkpoint_date}'])"
        ),
    }


@mcp.tool()
async def memory_list(
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
    sort_by_importance: bool = False,
    workspace: Optional[str] = None,
    workspaces: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """List memories, optionally filtering by substring query or metadata.

    Args:
        query: Optional text search query
        metadata_filters: Optional metadata filters
        limit: Maximum number of results to return (default: unlimited)
        offset: Number of results to skip (default: 0)
        date_from: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        date_to: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        tags_any: Match memories with ANY of these tags (OR logic)
        tags_all: Match memories with ALL of these tags (AND logic)
        tags_none: Exclude memories with ANY of these tags (NOT logic)
        sort_by_importance: Sort results by importance score (default: False, sorts by date)
        workspace: Filter to a single workspace (None = all workspaces)
        workspaces: Filter to multiple workspaces (None = all workspaces)
    """
    try:
        items = _list_memories(
            query,
            metadata_filters,
            limit,
            offset,
            date_from,
            date_to,
            tags_any,
            tags_all,
            tags_none,
            sort_by_importance,
            workspace,
            workspaces,
        )
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}
    return {"count": len(items), "memories": items}


@mcp.tool()
async def memory_list_compact(
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """List memories in compact format (id, preview, tags only) to reduce context usage.

    Returns minimal fields: id, content preview (first 200 chars), content_length, tags, and created_at.
    This tool is useful for browsing memories without loading full content and metadata.

    The content_length field helps identify verbose memories that may benefit from
    soft-trimming or summarization.

    Args:
        query: Optional text search query
        metadata_filters: Optional metadata filters
        limit: Maximum number of results to return (default: unlimited)
        offset: Number of results to skip (default: 0)
        date_from: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        date_to: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        tags_any: Match memories with ANY of these tags (OR logic)
        tags_all: Match memories with ALL of these tags (AND logic)
        tags_none: Exclude memories with ANY of these tags (NOT logic)
    """
    try:
        items = _list_memories(
            query,
            metadata_filters,
            limit,
            offset,
            date_from,
            date_to,
            tags_any,
            tags_all,
            tags_none,
        )
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}

    # Convert to compact format using _compact_memory helper
    compact_items = [_compact_memory(item) for item in items]

    return {"count": len(compact_items), "memories": compact_items}


@mcp.tool()
async def memory_create_batch(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create multiple memories in one call."""
    try:
        records = _create_memories(entries)
    except ValueError as exc:
        return {"error": "invalid_batch", "message": str(exc)}
    _schedule_cloud_graph_sync()
    return {"count": len(records), "memories": records}


@mcp.tool()
async def memory_delete_batch(ids: List[int]) -> Dict[str, Any]:
    """Delete multiple memories by id."""
    deleted = _delete_memories(ids)
    _schedule_cloud_graph_sync()
    return {"deleted": deleted}


@mcp.tool()
async def memory_get(memory_id: int, include_images: bool = False) -> Dict[str, Any]:
    """Retrieve a single memory by id.

    Args:
        memory_id: ID of the memory to retrieve
        include_images: If False, strip image data from metadata to reduce response size
    """
    record = _get_memory(memory_id)
    if not record:
        return {"error": "not_found", "id": memory_id}

    metadata = record.get("metadata") or {}
    if not include_images and metadata.get("images"):
        record["metadata"]["images"] = [
            {"caption": img.get("caption", "")} for img in metadata["images"]
        ]

    return {"memory": record}


@mcp.tool()
async def memory_soft_trim(
    memory_id: int,
    max_length: int = 500,
    head_ratio: float = 0.6,
    tail_ratio: float = 0.3,
) -> Dict[str, Any]:
    """Get a soft-trimmed view of a memory's content.

    Soft-trim preserves the beginning (head) and end (tail) of the content
    with an ellipsis in the middle showing how many characters were truncated.
    This is useful for viewing verbose memories without loading the full content.

    Note: This does NOT modify the stored memory. It only returns a trimmed view.
    To permanently shorten a memory, use memory_update with new content.

    Args:
        memory_id: ID of the memory to trim
        max_length: Maximum output length (default: 500 chars)
        head_ratio: Proportion of max_length for head (default: 0.6 = 60%)
        tail_ratio: Proportion of max_length for tail (default: 0.3 = 30%)

    Returns:
        Memory with trimmed_content, original_length, and was_trimmed fields.

    Example:
        For a 2000 char memory with max_length=500:
        - First 300 chars (60%) preserved
        - Last 150 chars (30%) preserved
        - Middle shows: "...[1550 chars truncated]..."
    """
    record = _get_memory(memory_id)
    if not record:
        return {"error": "not_found", "id": memory_id}

    content = record.get("content", "")
    original_length = len(content)
    trimmed_content = soft_trim(content, max_length, head_ratio, tail_ratio)

    return {
        "id": memory_id,
        "trimmed_content": trimmed_content,
        "original_length": original_length,
        "trimmed_length": len(trimmed_content),
        "was_trimmed": original_length > max_length,
        "tags": record.get("tags", []),
        "created_at": record.get("created_at"),
    }


@mcp.tool()
async def memory_update(
    memory_id: int,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    tier: Optional[str] = None,
    expires_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Update an existing memory. Only provided fields are updated.

    Args:
        memory_id: ID of the memory to update
        content: New content (optional)
        metadata: New metadata dict (optional)
        tags: New tags list (optional)
        tier: Memory tier - "daily" or "permanent" (optional)
        expires_at: Expiration datetime for daily memories (ISO format, optional)
    """
    try:
        record = _update_memory(memory_id, content, metadata, tags, tier, expires_at)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}
    if not record:
        return {"error": "not_found", "id": memory_id}
    _schedule_cloud_graph_sync()
    return {"memory": record}


@mcp.tool()
async def memory_delete(memory_id: int) -> Dict[str, Any]:
    """Delete a memory by id."""
    if _delete_memory(memory_id):
        _schedule_cloud_graph_sync()
        return {"status": "deleted", "id": memory_id}
    return {"error": "not_found", "id": memory_id}


@mcp.tool()
async def memory_tags() -> Dict[str, Any]:
    """Return the allowlisted tags."""
    from . import list_allowed_tags

    return {"allowed": list_allowed_tags()}


@mcp.tool()
async def memory_tag_hierarchy(include_root: bool = False) -> Dict[str, Any]:
    """Return stored tags organised as a namespace hierarchy."""

    tags = _collect_tags()
    tree = _build_tag_hierarchy(tags)
    if not include_root and isinstance(tree, dict):
        tree = tree.get("children", [])
    return {"count": len(tags), "hierarchy": tree}


@mcp.tool()
async def memory_validate_tags(include_memories: bool = True) -> Dict[str, Any]:
    """Validate stored tags against the allowlist and report invalid entries."""
    from . import list_allowed_tags

    invalid_full = _find_invalid_tags()
    allowed = list_allowed_tags()
    existing = _collect_tags()
    response: Dict[str, Any] = {
        "allowed": allowed,
        "existing": existing,
        "invalid_count": len(invalid_full),
    }
    if include_memories:
        response["invalid"] = invalid_full
    return response


@mcp.tool()
async def memory_hierarchy(
    query: Optional[str] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    include_root: bool = False,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
    compact: bool = True,
) -> Dict[str, Any]:
    """Return memories organised into a hierarchy derived from their metadata.

    Args:
        compact: If True (default), return only id, preview (first 80 chars), and tags
                 per memory to reduce response size. Set to False for full memory data.
    """
    try:
        items = _list_memories(
            query,
            metadata_filters,
            None,
            0,
            date_from,
            date_to,
            tags_any,
            tags_all,
            tags_none,
        )
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}

    hierarchy = _build_hierarchy_tree(items, include_root=include_root, compact=compact)
    return {"count": len(items), "hierarchy": hierarchy}


@mcp.tool()
async def memory_semantic_search(
    query: str,
    top_k: int = 5,
    metadata_filters: Optional[Dict[str, Any]] = None,
    min_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Perform a semantic search using vector embeddings."""

    try:
        results = _semantic_search(
            query,
            metadata_filters,
            top_k,
            min_score,
        )
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}
    return {"count": len(results), "results": results}


@mcp.tool()
async def memory_hybrid_search(
    query: str,
    semantic_weight: float = 0.6,
    fusion_method: str = "rrf",
    top_k: int = 10,
    min_score: float = 0.0,
    metadata_filters: Optional[Dict[str, Any]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    tags_any: Optional[List[str]] = None,
    tags_all: Optional[List[str]] = None,
    tags_none: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Perform a hybrid search combining keyword (FTS) and semantic (vector) search.

    Args:
        query: Search query text
        semantic_weight: Weight for semantic results (0-1). Higher values favor semantic similarity.
                        Keyword weight = 1 - semantic_weight. Default: 0.6 (60% semantic, 40% keyword)
        fusion_method: How to combine keyword and semantic results:
            - "rrf" (default): Reciprocal Rank Fusion - position-based merging, robust to score scale differences
            - "weighted": Direct score weighting - semantic_weight * vector_score + keyword_weight * text_score
        top_k: Maximum number of results to return (default: 10)
        min_score: Minimum combined score threshold (default: 0.0)
        metadata_filters: Optional metadata filters
        date_from: Optional date filter (ISO format or relative like "7d", "1m", "1y")
        date_to: Optional date filter (ISO format or relative)
        tags_any: Match memories with ANY of these tags (OR logic)
        tags_all: Match memories with ALL of these tags (AND logic)
        tags_none: Exclude memories with ANY of these tags (NOT logic)

    Returns:
        Dictionary with count and list of results, each containing score and memory
    """
    try:
        results = _hybrid_search(
            query,
            semantic_weight,
            fusion_method,
            top_k,
            min_score,
            metadata_filters,
            date_from,
            date_to,
            tags_any,
            tags_all,
            tags_none,
        )
    except ValueError as exc:
        return {"error": "invalid_filters", "message": str(exc)}
    return {"count": len(results), "results": results}


@mcp.tool()
async def memory_rebuild_embeddings() -> Dict[str, Any]:
    """Recompute embeddings for all memories."""

    updated = _rebuild_embeddings()
    return {"updated": updated}


@mcp.tool()
async def memory_related(memory_id: int, refresh: bool = False) -> Dict[str, Any]:
    """Return cross-referenced memories for a given entry."""

    related = _get_related(memory_id, refresh)
    return {"id": memory_id, "related": related}


@mcp.tool()
async def memory_rebuild_crossrefs() -> Dict[str, Any]:
    """Recompute cross-reference links for all memories."""

    updated = _rebuild_crossrefs()
    return {"updated": updated}


@mcp.tool()
async def memory_stats() -> Dict[str, Any]:
    """Get statistics and analytics about stored memories."""

    return _get_statistics()


@mcp.tool()
async def memory_embedding_cache_stats() -> Dict[str, Any]:
    """Get embedding cache statistics.

    Returns cache hit/miss rates, entry count, and storage size.
    Only available when MEMORA_EMBEDDING_CACHE is enabled (default: true).

    Returns:
        Dictionary with cache statistics including hits, misses, hit_rate,
        entries, max_entries, and enabled status.
    """
    from . import storage

    if not storage.EMBEDDING_CACHE_ENABLED:
        return {"enabled": False, "message": "Embedding cache is disabled"}

    with _get_conn() as conn:
        return storage.get_embedding_cache_stats(conn)


@mcp.tool()
async def memory_embedding_cache_clear() -> Dict[str, Any]:
    """Clear the embedding cache.

    Removes all cached embeddings. Useful when changing embedding models
    or when cache has become stale.

    Returns:
        Dictionary with count of cleared entries.
    """
    from . import storage

    if not storage.EMBEDDING_CACHE_ENABLED:
        return {
            "enabled": False,
            "message": "Embedding cache is disabled",
            "cleared": 0,
        }

    with _get_conn() as conn:
        cleared = storage.clear_embedding_cache(conn)
        conn.commit()
        return {"cleared": cleared, "enabled": True}


@mcp.tool()
async def memory_sync_version() -> Dict[str, Any]:
    """Get the current global sync version.

    Use this to check the current version before calling memory_sync_delta.

    Returns:
        Dictionary with current_version number.
    """
    from . import storage

    with _get_conn() as conn:
        version = storage.get_current_sync_version(conn)
        return {"current_version": version}


@mcp.tool()
async def memory_sync_delta(
    since_version: int,
    include_deleted: bool = True,
    agent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Get all memory changes since a version number for delta sync.

    This enables efficient incremental synchronization between agents/sessions.
    Each create, update, or delete operation increments the global sync version.

    Args:
        since_version: Get changes after this version (exclusive). Use 0 for full sync.
        include_deleted: Include deleted memory records (default: True)
        agent_id: Optional agent ID to track sync state. If provided, the agent's
                  last_sync_version will be updated after this call.

    Returns:
        Dictionary with:
        - current_version: The current global sync version
        - since_version: The version you requested changes from
        - change_count: Number of changes returned
        - changes: List of changes, each containing:
            - action: "create", "update", or "delete"
            - sync_version: Version when this change occurred
            - memory: Full memory object (for create/update)
            - memory_id, content_preview, deleted_at (for delete)
    """
    from . import storage

    with _get_conn() as conn:
        return storage.sync_delta(
            conn,
            since_version=since_version,
            include_deleted=include_deleted,
            agent_id=agent_id,
        )


@mcp.tool()
async def memory_sync_state(agent_id: str) -> Dict[str, Any]:
    """Get the sync state for a specific agent.

    Args:
        agent_id: The agent identifier to check

    Returns:
        Dictionary with agent_id, last_sync_version, and last_sync_at
    """
    from . import storage

    with _get_conn() as conn:
        return storage.get_agent_sync_state(conn, agent_id)


@mcp.tool()
async def memory_sync_cleanup(older_than_days: int = 30) -> Dict[str, Any]:
    """Clean up old deleted memory records.

    Deleted memories are tracked for sync purposes. This removes old
    deletion records that are no longer needed.

    Args:
        older_than_days: Remove records older than this (default: 30)

    Returns:
        Dictionary with count of removed records
    """
    from . import storage

    with _get_conn() as conn:
        removed = storage.cleanup_deleted_memories(conn, older_than_days)
        return {"removed": removed, "older_than_days": older_than_days}


@mcp.tool()
async def memory_share(
    memory_id: int,
    source_agent: Optional[str] = None,
    target_agents: Optional[List[str]] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """Share a memory with other agents/sessions.

    Creates a share event that other agents can poll for. Automatically
    adds the 'shared-cache' tag to the memory.

    Args:
        memory_id: ID of the memory to share
        source_agent: Your agent ID (for tracking who shared)
        target_agents: List of specific agent IDs to share with.
                      If None, broadcasts to all agents.
        message: Optional message to include with the share

    Returns:
        Dictionary with share event details including event_id
    """
    from . import storage

    with _get_conn() as conn:
        try:
            return storage.share_memory(
                conn,
                memory_id=memory_id,
                source_agent=source_agent,
                target_agents=target_agents,
                message=message,
            )
        except ValueError as e:
            return {"error": str(e)}


@mcp.tool()
async def memory_shared_poll(
    agent_id: Optional[str] = None,
    since_timestamp: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Poll for memories shared with you.

    Retrieves share events, optionally filtered by agent and time.

    Args:
        agent_id: Your agent ID to filter shares targeted at you.
                 If None, returns all shares (including broadcasts).
        since_timestamp: Only get shares after this timestamp (ISO format)
        limit: Maximum number of results (default: 50)

    Returns:
        Dictionary with count and list of share events, each containing
        the shared memory and share metadata.
    """
    from . import storage

    with _get_conn() as conn:
        return storage.get_shared_memories(
            conn,
            agent_id=agent_id,
            since_timestamp=since_timestamp,
            limit=limit,
        )


@mcp.tool()
async def memory_share_ack(
    event_id: int,
    agent_id: str,
) -> Dict[str, Any]:
    """Acknowledge receipt of a shared memory.

    Marks that your agent has received and processed a share event.

    Args:
        event_id: The share event ID to acknowledge
        agent_id: Your agent ID

    Returns:
        Dictionary with acknowledgment status
    """
    from . import storage

    with _get_conn() as conn:
        return storage.acknowledge_share(conn, event_id, agent_id)


@mcp.tool()
async def memory_boost(
    memory_id: int,
    boost_amount: float = 0.5,
) -> Dict[str, Any]:
    """Boost a memory's importance score.

    Manually increase a memory's base importance to make it rank higher in
    importance-sorted searches. The boost is permanent and cumulative.

    Args:
        memory_id: ID of the memory to boost
        boost_amount: Amount to add to base importance (default: 0.5)
                      Common values: 0.25 (small), 0.5 (medium), 1.0 (large)

    Returns:
        Updated memory with new importance score, or error if not found
    """
    record = _boost_memory(memory_id, boost_amount)
    if not record:
        return {"error": "not_found", "id": memory_id}
    _schedule_cloud_graph_sync()
    return {"memory": record, "boosted_by": boost_amount}


@mcp.tool()
async def memory_promote(
    memory_id: int,
    clear_expiration: bool = True,
) -> Dict[str, Any]:
    """Promote a daily memory to permanent tier.

    Converts an ephemeral daily memory into a permanent one,
    optionally clearing the expiration time.

    Args:
        memory_id: ID of the memory to promote
        clear_expiration: If True (default), also clear the expires_at field

    Returns:
        Updated memory with tier="permanent", or error if not found
    """
    # Get current memory to check its tier
    current = _get_memory(memory_id)
    if not current:
        return {"error": "not_found", "id": memory_id}

    current_tier = current.get("tier", "permanent")
    if current_tier == "permanent":
        return {
            "memory": current,
            "message": "Memory is already permanent",
            "changed": False,
        }

    # Promote to permanent
    expires_at = None if clear_expiration else current.get("expires_at")
    try:
        record = _update_memory(
            memory_id,
            content=None,
            metadata=None,
            tags=None,
            tier="permanent",
            expires_at=expires_at,
        )
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}

    if not record:
        return {"error": "not_found", "id": memory_id}

    return {
        "memory": record,
        "message": f"Memory promoted from '{current_tier}' to 'permanent'",
        "changed": True,
        "previous_tier": current_tier,
    }


@_with_connection(writes=True)
def _cleanup_expired(conn, dry_run: bool = False):
    return cleanup_expired_memories(conn, dry_run=dry_run)


@mcp.tool()
async def memory_cleanup_expired(
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Delete expired daily tier memories.

    Removes memories where tier='daily' and expires_at has passed.
    Run this periodically to clean up ephemeral memories.

    Args:
        dry_run: If True, only report what would be deleted without actually deleting

    Returns:
        Dictionary with count of deleted memories and their details
    """
    result = _cleanup_expired(dry_run=dry_run)
    if not dry_run and result.get("deleted", 0) > 0:
        _schedule_cloud_graph_sync()
    return result


@_with_connection(writes=True)
def _add_link(conn, from_id: int, to_id: int, edge_type: str, bidirectional: bool):
    return add_link(conn, from_id, to_id, edge_type, bidirectional)


@_with_connection(writes=True)
def _remove_link(conn, from_id: int, to_id: int, bidirectional: bool):
    return remove_link(conn, from_id, to_id, bidirectional)


@_with_connection
def _detect_clusters(conn, min_cluster_size: int, min_score: float):
    return detect_clusters(conn, min_cluster_size, min_score)


@mcp.tool()
async def memory_link(
    from_id: int,
    to_id: int,
    edge_type: str = "references",
    bidirectional: bool = True,
) -> Dict[str, Any]:
    """Create an explicit typed link between two memories.

    Args:
        from_id: Source memory ID
        to_id: Target memory ID
        edge_type: Type of relationship. Options:
            - "references" (default): General reference
            - "implements": Source implements/realizes target
            - "supersedes": Source replaces/updates target
            - "extends": Source builds upon target
            - "contradicts": Source conflicts with target
            - "related_to": Generic relationship
        bidirectional: If True, also create reverse link (default: True)

    Returns:
        Dict with created links and their types
    """
    try:
        result = _add_link(from_id, to_id, edge_type, bidirectional)
        _schedule_cloud_graph_sync()
        return result
    except ValueError as e:
        return {"error": "invalid_input", "message": str(e)}


@mcp.tool()
async def memory_unlink(
    from_id: int,
    to_id: int,
    bidirectional: bool = True,
) -> Dict[str, Any]:
    """Remove a link between two memories.

    Args:
        from_id: Source memory ID
        to_id: Target memory ID
        bidirectional: If True, also remove reverse link (default: True)

    Returns:
        Dict with removed links
    """
    result = _remove_link(from_id, to_id, bidirectional)
    _schedule_cloud_graph_sync()
    return result


@mcp.tool()
async def memory_clusters(
    min_cluster_size: int = 2,
    min_score: float = 0.3,
) -> Dict[str, Any]:
    """Detect clusters of related memories.

    Uses connected components algorithm to find groups of memories
    that are linked together through cross-references.

    Args:
        min_cluster_size: Minimum memories to form a cluster (default: 2)
        min_score: Minimum similarity score to consider connected (default: 0.3)

    Returns:
        List of clusters with member IDs, sizes, and common tags
    """
    clusters = _detect_clusters(min_cluster_size, min_score)
    return {
        "count": len(clusters),
        "clusters": clusters,
    }


@mcp.tool()
async def memory_find_duplicates(
    min_similarity: float = 0.7,
    max_similarity: float = 0.95,
    limit: int = 10,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Find potential duplicate memory pairs with optional LLM-powered comparison.

    Scans cross-references to find memory pairs with similarity scores in the
    specified range, then optionally uses LLM to semantically compare them.

    Args:
        min_similarity: Minimum similarity score to consider (default: 0.7)
        max_similarity: Maximum similarity score to consider (default: 0.95)
        limit: Maximum pairs to analyze (default: 10)
        use_llm: Whether to use LLM for semantic comparison (default: True)

    Returns:
        Dictionary with:
        - pairs: List of potential duplicate pairs with analysis
        - total_candidates: Total pairs found in range
        - analyzed: Number of pairs analyzed with LLM
        - llm_available: Whether LLM comparison was available
    """
    from .storage import compare_memories_llm, connect, find_duplicate_candidates

    with connect() as conn:
        candidates = find_duplicate_candidates(
            conn, min_similarity, max_similarity, limit * 2
        )

    total_candidates = len(candidates)
    pairs = []
    llm_available = False

    for candidate in candidates[:limit]:
        mem_a = _get_memory(candidate["memory_a_id"])
        mem_b = _get_memory(candidate["memory_b_id"])

        if not mem_a or not mem_b:
            continue

        pair_result = {
            "memory_a": {
                "id": mem_a["id"],
                "preview": mem_a["content"][:150] + "..."
                if len(mem_a["content"]) > 150
                else mem_a["content"],
                "tags": mem_a.get("tags", []),
            },
            "memory_b": {
                "id": mem_b["id"],
                "preview": mem_b["content"][:150] + "..."
                if len(mem_b["content"]) > 150
                else mem_b["content"],
                "tags": mem_b.get("tags", []),
            },
            "similarity_score": round(candidate["similarity_score"], 3),
        }

        # Run LLM comparison if enabled
        if use_llm:
            llm_result = compare_memories_llm(
                mem_a["content"],
                mem_b["content"],
                mem_a.get("metadata"),
                mem_b.get("metadata"),
            )
            if llm_result:
                llm_available = True
                pair_result["llm_verdict"] = llm_result.get("verdict", "review")
                pair_result["llm_confidence"] = llm_result.get("confidence", 0)
                pair_result["llm_reasoning"] = llm_result.get("reasoning", "")
                pair_result["suggested_action"] = llm_result.get(
                    "suggested_action", "review"
                )
                if llm_result.get("merge_suggestion"):
                    pair_result["merge_suggestion"] = llm_result["merge_suggestion"]

        pairs.append(pair_result)

    return {
        "pairs": pairs,
        "total_candidates": total_candidates,
        "analyzed": len(pairs),
        "llm_available": llm_available,
    }


@mcp.tool()
async def memory_merge(
    source_id: int,
    target_id: int,
    merge_strategy: str = "append",
) -> Dict[str, Any]:
    """Merge source memory into target, then delete source.

    Combines two memories into one, preserving content and metadata.

    Args:
        source_id: Memory ID to merge from (will be deleted)
        target_id: Memory ID to merge into (will be updated)
        merge_strategy: How to combine content:
            - "append": Append source content to target (default)
            - "prepend": Prepend source content to target
            - "replace": Replace target content with source

    Returns:
        Updated target memory and deletion confirmation
    """
    from .storage import connect, delete_memory, update_memory

    source = _get_memory(source_id)
    target = _get_memory(target_id)

    if not source:
        return {
            "error": "not_found",
            "message": f"Source memory #{source_id} not found",
        }
    if not target:
        return {
            "error": "not_found",
            "message": f"Target memory #{target_id} not found",
        }

    # Combine content based on strategy
    if merge_strategy == "prepend":
        new_content = source["content"] + "\n\n---\n\n" + target["content"]
    elif merge_strategy == "replace":
        new_content = source["content"]
    else:  # append (default)
        new_content = target["content"] + "\n\n---\n\n" + source["content"]

    # Merge metadata (target takes precedence, but add source-specific fields)
    merged_metadata = dict(source.get("metadata") or {})
    merged_metadata.update(target.get("metadata") or {})
    merged_metadata["merged_from"] = source_id

    # Union tags
    source_tags = set(source.get("tags") or [])
    target_tags = set(target.get("tags") or [])
    merged_tags = list(source_tags | target_tags)

    # Update target memory
    with connect() as conn:
        updated = update_memory(
            conn,
            target_id,
            content=new_content,
            metadata=merged_metadata,
            tags=merged_tags,
        )
        conn.commit()

        # Delete source memory
        delete_memory(conn, source_id)
        conn.commit()

    _schedule_cloud_graph_sync()
    return {
        "merged": True,
        "target_id": target_id,
        "source_id": source_id,
        "updated_memory": updated,
        "message": f"Memory #{source_id} merged into #{target_id} and deleted",
    }


@mcp.tool()
async def memory_export() -> Dict[str, Any]:
    """Export all memories to JSON format for backup or transfer."""

    memories = _export_memories()
    return {"count": len(memories), "memories": memories}


@mcp.tool()
async def memory_upload_image(
    file_path: str,
    memory_id: int,
    image_index: int = 0,
    caption: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload an image file directly to R2 storage.

    Uploads a local image file to R2 and returns the r2:// reference URL
    that can be used in memory metadata.

    Args:
        file_path: Absolute path to the image file to upload
        memory_id: Memory ID this image belongs to (used for organizing in R2)
        image_index: Index of image within the memory (default: 0)
        caption: Optional caption for the image

    Returns:
        Dictionary with r2_url (the r2:// reference) and image object ready for metadata
    """
    import mimetypes
    import os

    from .image_storage import get_image_storage_instance

    image_storage = get_image_storage_instance()
    if not image_storage:
        return {
            "error": "r2_not_configured",
            "message": "R2 storage is not configured. Set MEMORA_STORAGE_URI to s3:// and configure AWS credentials.",
        }

    # Validate file exists
    if not os.path.isfile(file_path):
        return {"error": "file_not_found", "message": f"File not found: {file_path}"}

    # Determine content type
    content_type, _ = mimetypes.guess_type(file_path)
    if not content_type or not content_type.startswith("image/"):
        content_type = "image/png"  # Default to PNG for unknown types

    try:
        # Read file and upload
        with open(file_path, "rb") as f:
            image_data = f.read()

        r2_url = image_storage.upload_image(
            image_data=image_data,
            content_type=content_type,
            memory_id=memory_id,
            image_index=image_index,
        )

        # Build image object for metadata
        image_obj = {"src": r2_url}
        if caption:
            image_obj["caption"] = caption

        return {
            "r2_url": r2_url,
            "image": image_obj,
            "file_path": file_path,
            "content_type": content_type,
            "size_bytes": len(image_data),
        }

    except Exception as e:
        return {"error": "upload_failed", "message": str(e)}


@mcp.tool()
async def memory_migrate_images(dry_run: bool = False) -> Dict[str, Any]:
    """Migrate existing base64 images to R2 storage.

    Scans all memories and uploads any base64-encoded images to R2,
    replacing the data URIs with R2 URLs.

    Args:
        dry_run: If True, only report what would be migrated without making changes

    Returns:
        Dictionary with migration results including count of migrated images
    """
    return _migrate_images_to_r2(dry_run=dry_run)


@_with_connection(writes=True)
def _migrate_images_to_r2(conn, dry_run: bool = False) -> Dict[str, Any]:
    """Migrate all base64 images to R2 storage."""
    import json as json_lib

    from .image_storage import get_image_storage_instance, parse_data_uri
    from .storage import update_memory

    image_storage = get_image_storage_instance()
    if not image_storage:
        return {
            "error": "r2_not_configured",
            "message": "R2 storage is not configured. Set MEMORA_STORAGE_URI to s3:// and configure AWS credentials.",
        }

    # Find memories with base64 images
    rows = conn.execute(
        "SELECT id, metadata FROM memories WHERE metadata LIKE '%data:image%'"
    ).fetchall()

    if not rows:
        return {
            "migrated_memories": 0,
            "migrated_images": 0,
            "message": "No base64 images found",
        }

    results = {
        "dry_run": dry_run,
        "memories_scanned": len(rows),
        "migrated_memories": 0,
        "migrated_images": 0,
        "errors": [],
    }

    for row in rows:
        memory_id = row["id"]
        try:
            metadata = json_lib.loads(row["metadata"]) if row["metadata"] else {}
        except json_lib.JSONDecodeError:
            continue

        images = metadata.get("images", [])
        if not isinstance(images, list):
            continue

        updated = False
        for idx, img in enumerate(images):
            if not isinstance(img, dict):
                continue
            src = img.get("src", "")
            if not src.startswith("data:image"):
                continue

            if dry_run:
                results["migrated_images"] += 1
                updated = True
                continue

            # Upload to R2
            try:
                image_bytes, content_type = parse_data_uri(src)
                new_url = image_storage.upload_image(
                    image_data=image_bytes,
                    content_type=content_type,
                    memory_id=memory_id,
                    image_index=idx,
                )
                img["src"] = new_url
                results["migrated_images"] += 1
                updated = True
            except Exception as e:
                results["errors"].append(
                    {
                        "memory_id": memory_id,
                        "image_index": idx,
                        "error": str(e),
                    }
                )

        if updated:
            results["migrated_memories"] += 1
            if not dry_run:
                # Update the memory with new URLs
                update_memory(conn, memory_id, metadata=metadata)

    if dry_run:
        results["message"] = (
            f"Would migrate {results['migrated_images']} images from {results['migrated_memories']} memories"
        )
    else:
        results["message"] = (
            f"Migrated {results['migrated_images']} images from {results['migrated_memories']} memories"
        )

    return results


# NOTE: Graph visualization functions moved to memora/graph/ module
# See: graph/data.py, graph/templates.py, graph/issues.py, graph/server.py


@mcp.tool()
async def memory_export_graph(
    output_path: Optional[str] = None,
    min_score: float = 0.25,
) -> Dict[str, Any]:
    """Export memories as interactive HTML knowledge graph.

    Args:
        output_path: Path to save HTML file (default: ~/memories_graph.html)
        min_score: Minimum similarity score for edges (default: 0.25)

    Returns:
        Dictionary with path, node count, edge count, and tags
    """
    import os

    if output_path is None:
        output_path = os.path.expanduser("~/memories_graph.html")

    return export_graph_html(output_path, min_score)


# Removed ~400 lines of old _export_graph_html code - now in graph/data.py


@mcp.tool()
async def memory_import(
    data: List[Dict[str, Any]],
    strategy: str = "append",
) -> Dict[str, Any]:
    """Import memories from JSON format.

    Args:
        data: List of memory dictionaries with content, metadata, tags, created_at
        strategy: "replace" (clear all first), "merge" (skip duplicates), or "append" (add all)
    """
    try:
        result = _import_memories(data, strategy)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}
    _schedule_cloud_graph_sync()
    return result


@_with_connection
def _poll_events(
    conn,
    since_timestamp: Optional[str],
    tags_filter: Optional[List[str]],
    unconsumed_only: bool,
):
    return poll_events(conn, since_timestamp, tags_filter, unconsumed_only)


@_with_connection(writes=True)
def _clear_events(conn, event_ids: List[int]):
    return clear_events(conn, event_ids)


@mcp.tool()
async def memory_events_poll(
    since_timestamp: Optional[str] = None,
    tags_filter: Optional[List[str]] = None,
    unconsumed_only: bool = True,
) -> Dict[str, Any]:
    """Poll for memory events (e.g., shared-cache notifications).

    Args:
        since_timestamp: Only return events after this timestamp (ISO format)
        tags_filter: Only return events with these tags (e.g., ["shared-cache"])
        unconsumed_only: Only return unconsumed events (default: True)

    Returns:
        Dictionary with count and list of events
    """
    events = _poll_events(since_timestamp, tags_filter, unconsumed_only)
    return {"count": len(events), "events": events}


@mcp.tool()
async def memory_events_clear(event_ids: List[int]) -> Dict[str, Any]:
    """Mark events as consumed.

    Args:
        event_ids: List of event IDs to mark as consumed

    Returns:
        Dictionary with count of cleared events
    """
    cleared = _clear_events(event_ids)
    return {"cleared": cleared}


# Graph functions moved to memora/graph/ module


# =============================================================================
# Project Context Tools - AI Instruction File Discovery & Ingestion
# =============================================================================

from .project_context import (
    CORE_INSTRUCTION_FILES,
    ProjectContextConfig,
    discover_instruction_files,
    find_existing_context_memories,
    scan_project_context,
    update_or_create_context_memory,
)


@_with_connection(writes=True)
def _scan_project(
    conn,
    path: str,
    extract_sections: bool,
    scan_parents: bool,
    force_rescan: bool,
) -> Dict[str, Any]:
    """Internal function to scan project and create/update memories."""
    from pathlib import Path

    config = ProjectContextConfig(
        extract_sections=extract_sections,
        scan_parents=scan_parents,
    )

    # Get memories to create
    memories = scan_project_context(path, config)

    if not memories:
        return {
            "status": "no_files_found",
            "path": path,
            "scanned_patterns": list(CORE_INSTRUCTION_FILES.keys()),
        }

    # Find existing context memories for this project
    existing = find_existing_context_memories(conn, path)

    results = {
        "created": 0,
        "updated": 0,
        "unchanged": 0,
        "moved": 0,
        "errors": [],
        "memories": [],
    }

    for memory_data in memories:
        try:
            action, memory_id = update_or_create_context_memory(
                conn, memory_data, existing
            )
            results[action] = results.get(action, 0) + 1
            if memory_id and action in ("created", "updated"):
                results["memories"].append(
                    {
                        "id": memory_id,
                        "action": action,
                        "source": memory_data.get("metadata", {}).get(
                            "source_file", ""
                        ),
                        "section": memory_data.get("metadata", {}).get(
                            "section_path", ""
                        ),
                    }
                )
        except Exception as e:
            results["errors"].append(
                {
                    "source": memory_data.get("metadata", {}).get("source_file", ""),
                    "error": str(e),
                }
            )

    results["path"] = path
    results["total_processed"] = len(memories)

    return results


@mcp.tool()
async def memory_scan_project(
    path: Optional[str] = None,
    extract_sections: bool = True,
    scan_parents: bool = False,
    force_rescan: bool = False,
) -> Dict[str, Any]:
    """Scan current directory for AI instruction files and ingest them as memories.

    Discovers and parses AI instruction files (CLAUDE.md, .cursorrules, AGENTS.md, etc.)
    and creates searchable memories for each file and section.

    Supported files:
    - CLAUDE.md - Claude Code instructions
    - AGENTS.md - Multi-agent system instructions
    - .cursorrules - Cursor IDE rules
    - .github/copilot-instructions.md - GitHub Copilot instructions
    - GEMINI.md - Gemini tools instructions
    - .aider.conf.yml - Aider configuration
    - CONVENTIONS.md, CODING_GUIDELINES.md - General conventions
    - .windsurfrules - Windsurf IDE rules

    Args:
        path: Directory to scan (default: current working directory)
        extract_sections: Parse markdown into separate section memories (default: True)
        scan_parents: Also scan parent directories (default: False for security)
        force_rescan: Force update even if file hasn't changed (default: False)

    Returns:
        Dictionary with counts of created/updated/unchanged memories and details
    """
    import os

    if path is None:
        path = os.getcwd()

    return _scan_project(path, extract_sections, scan_parents, force_rescan)


@_with_connection
def _get_project_context(
    conn,
    path: str,
    include_sections: bool,
) -> Dict[str, Any]:
    """Internal function to get project context memories."""
    existing = find_existing_context_memories(conn, path)

    if not existing:
        return {
            "count": 0,
            "memories": [],
            "path": path,
            "hint": "No project context found. Run memory_scan_project to ingest instruction files.",
        }

    # Filter by type if needed
    if not include_sections:
        existing = [
            m for m in existing if m.get("metadata", {}).get("is_parent", False)
        ]

    # Return compact format
    compact_memories = []
    for mem in existing:
        metadata = mem.get("metadata", {})
        content = mem.get("content", "")
        preview = content[:200] + "..." if len(content) > 200 else content

        compact_memories.append(
            {
                "id": mem["id"],
                "preview": preview,
                "tags": mem.get("tags", []),
                "source_file": metadata.get("source_file", ""),
                "file_type": metadata.get("file_type", ""),
                "section_path": metadata.get("section_path", ""),
                "is_section": metadata.get("is_section", False),
            }
        )

    return {
        "count": len(compact_memories),
        "memories": compact_memories,
        "path": path,
    }


@mcp.tool()
async def memory_get_project_context(
    path: Optional[str] = None,
    include_sections: bool = True,
) -> Dict[str, Any]:
    """Get all project context memories for a directory.

    Retrieves previously ingested AI instruction file memories for a project.
    Use memory_scan_project first to ingest files.

    Args:
        path: Directory path (default: current working directory)
        include_sections: Include section-level memories (default: True)

    Returns:
        Dictionary with count and list of project context memories
    """
    import os

    if path is None:
        path = os.getcwd()

    return _get_project_context(path, include_sections)


@mcp.tool()
async def memory_list_instruction_files(
    path: Optional[str] = None,
) -> Dict[str, Any]:
    """List AI instruction files discovered in a directory (without ingesting).

    Scans for known instruction file patterns without creating memories.
    Useful for previewing what would be ingested.

    Args:
        path: Directory to scan (default: current working directory)

    Returns:
        Dictionary with list of discovered files and their types
    """
    import os
    from pathlib import Path

    if path is None:
        path = os.getcwd()

    files = discover_instruction_files(path)

    file_info = []
    for file_path, file_type, file_format in files:
        try:
            stat = file_path.stat()
            file_info.append(
                {
                    "path": str(file_path),
                    "name": file_path.name,
                    "type": file_type.value,
                    "format": file_format.value,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
            )
        except Exception:
            continue

    return {
        "count": len(file_info),
        "files": file_info,
        "path": path,
        "supported_patterns": list(CORE_INSTRUCTION_FILES.keys()),
    }


# ---------------------------------------------------------------------------
# Session Transcript Indexing Tools
# ---------------------------------------------------------------------------


@_with_connection(writes=True)
def _index_conversation(
    conn,
    messages: List[Dict[str, Any]],
    session_id: Optional[str],
    title: Optional[str],
    chunk_size: int,
    overlap: int,
    tags: Optional[List[str]],
    create_memories: bool,
):
    return index_conversation(
        conn, messages, session_id, title, chunk_size, overlap, tags, create_memories
    )


@_with_connection(writes=True)
def _index_conversation_delta(
    conn,
    session_id: str,
    new_messages: List[Dict[str, Any]],
    chunk_size: int,
    overlap: int,
    tags: Optional[List[str]],
):
    return index_conversation_delta(
        conn, session_id, new_messages, chunk_size, overlap, tags
    )


@_with_connection
def _get_session(conn, session_id: str):
    return get_session(conn, session_id)


@_with_connection
def _list_sessions(
    conn,
    limit: Optional[int],
    offset: int,
    date_from: Optional[str],
    date_to: Optional[str],
):
    return list_sessions(conn, limit, offset, date_from, date_to)


@_with_connection
def _search_sessions(
    conn,
    query: str,
    session_ids: Optional[List[str]],
    top_k: int,
    min_score: float,
):
    return search_sessions(conn, query, session_ids, top_k, min_score)


@_with_connection(writes=True)
def _delete_session(conn, session_id: str):
    return delete_session(conn, session_id)


@mcp.tool()
async def memory_index_conversation(
    messages: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    title: Optional[str] = None,
    chunk_size: int = 10,
    overlap: int = 2,
    tags: Optional[List[str]] = None,
    create_memories: bool = True,
) -> Dict[str, Any]:
    """Index a conversation for semantic search.

    Chunks the conversation into overlapping segments and creates searchable
    memory entries for each chunk. This enables semantic search across
    past conversations.

    Args:
        messages: List of message dicts with 'role', 'content', and optional 'timestamp'.
                  Example: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
        session_id: Unique session identifier (auto-generated if not provided)
        title: Optional descriptive title for the session
        chunk_size: Number of messages per chunk (default: 10)
        overlap: Number of messages to overlap between chunks for context continuity (default: 2)
        tags: Additional tags to apply to created memories
        create_memories: If True, create searchable memory entries for each chunk (default: True)

    Returns:
        Dict with session_id, message_count, chunks_created, memories_created

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "How do I implement auth?"},
        ...     {"role": "assistant", "content": "You can use JWT tokens..."},
        ...     {"role": "user", "content": "Show me an example"},
        ...     {"role": "assistant", "content": "Here's the code..."}
        ... ]
        >>> result = await memory_index_conversation(messages, title="Auth discussion")
        >>> result
        {"session_id": "session-20260128-143022-abc12345", "chunks_created": 1, ...}
    """
    if not messages:
        return {"error": "no_messages", "message": "No messages provided"}

    result = _index_conversation(
        messages, session_id, title, chunk_size, overlap, tags, create_memories
    )
    if result.get("memories_created"):
        _schedule_cloud_graph_sync()
    return result


@mcp.tool()
async def memory_index_conversation_delta(
    session_id: str,
    new_messages: List[Dict[str, Any]],
    chunk_size: int = 10,
    overlap: int = 2,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Incrementally index new messages for an existing session.

    Use this to add new messages to a previously indexed session without
    re-indexing the entire conversation. Only the new messages are processed.

    Args:
        session_id: Existing session identifier
        new_messages: New messages to add and index
        chunk_size: Messages per chunk (default: 10)
        overlap: Overlap between chunks (default: 2)
        tags: Additional tags for new memories

    Returns:
        Dict with new_chunks_created, new_memories_created, total_message_count
    """
    if not new_messages:
        return {"error": "no_messages", "message": "No new messages provided"}

    result = _index_conversation_delta(
        session_id, new_messages, chunk_size, overlap, tags
    )
    if result.get("new_memories_created"):
        _schedule_cloud_graph_sync()
    return result


@mcp.tool()
async def memory_session_get(session_id: str) -> Dict[str, Any]:
    """Get metadata for an indexed session.

    Args:
        session_id: Session identifier

    Returns:
        Session metadata including title, message_count, chunk_count, timestamps
    """
    session = _get_session(session_id)
    if not session:
        return {"error": "not_found", "session_id": session_id}
    return {"session": session}


@mcp.tool()
async def memory_session_list(
    limit: Optional[int] = None,
    offset: int = 0,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> Dict[str, Any]:
    """List all indexed sessions.

    Args:
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip (for pagination)
        date_from: Filter sessions created after this date (ISO format)
        date_to: Filter sessions created before this date (ISO format)

    Returns:
        Dict with count and list of sessions
    """
    sessions = _list_sessions(limit, offset, date_from, date_to)
    return {"count": len(sessions), "sessions": sessions}


@mcp.tool()
async def memory_session_search(
    query: str,
    session_ids: Optional[List[str]] = None,
    top_k: int = 10,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """Search across indexed session transcripts using semantic search.

    Finds relevant conversation chunks matching your query. Results include
    session context so you know which conversation the match came from.

    Args:
        query: Search query text
        session_ids: Optional list of session IDs to search within (None = all sessions)
        top_k: Maximum number of results to return (default: 10)
        min_score: Minimum similarity score threshold (default: 0.0)

    Returns:
        Dict with results including score, content, session_id, session_title, chunk_index

    Example:
        >>> result = await memory_session_search("authentication implementation")
        >>> result["results"][0]
        {"score": 0.87, "session_title": "Auth discussion", "content": "...", ...}
    """
    results = _search_sessions(query, session_ids, top_k, min_score)
    return {"count": len(results), "results": results}


@mcp.tool()
async def memory_session_delete(session_id: str) -> Dict[str, Any]:
    """Delete an indexed session and all its associated chunks and memories.

    This permanently removes the session, all its chunks, and any memories
    created from those chunks.

    Args:
        session_id: Session to delete

    Returns:
        Dict with deletion counts (session_deleted, chunks_deleted, memories_deleted)
    """
    result = _delete_session(session_id)
    if result.get("memories_deleted", 0) > 0:
        _schedule_cloud_graph_sync()
    return result


# ---------------------------------------------------------------------------
# Identity Links (Entity Unification) Tools
# ---------------------------------------------------------------------------


@_with_connection(writes=True)
def _create_identity(
    conn,
    canonical_id: str,
    display_name: str,
    entity_type: str,
    aliases: Optional[List[str]],
    metadata: Optional[Dict[str, Any]],
):
    return create_identity(
        conn, canonical_id, display_name, entity_type, aliases, metadata
    )


@_with_connection
def _get_identity(conn, canonical_id: str):
    return get_identity(conn, canonical_id)


@_with_connection(writes=True)
def _update_identity(
    conn,
    canonical_id: str,
    display_name: Optional[str],
    entity_type: Optional[str],
    metadata: Optional[Dict[str, Any]],
):
    return update_identity(conn, canonical_id, display_name, entity_type, metadata)


@_with_connection(writes=True)
def _delete_identity(conn, canonical_id: str):
    return delete_identity(conn, canonical_id)


@_with_connection
def _list_identities(
    conn,
    entity_type: Optional[str],
    limit: Optional[int],
    offset: int,
):
    return list_identities(conn, entity_type, limit, offset)


@_with_connection
def _search_identities(conn, query: str, entity_type: Optional[str], limit: int):
    return search_identities(conn, query, entity_type, limit)


@_with_connection(writes=True)
def _add_identity_alias(conn, canonical_id: str, alias: str, source: str):
    return add_identity_alias(conn, canonical_id, alias, source)


@_with_connection(writes=True)
def _link_memory_to_identity(
    conn, memory_id: int, identity_id: str, mention_text: Optional[str]
):
    return link_memory_to_identity(conn, memory_id, identity_id, mention_text)


@_with_connection(writes=True)
def _unlink_memory_from_identity(conn, memory_id: int, identity_id: str):
    return unlink_memory_from_identity(conn, memory_id, identity_id)


@_with_connection
def _get_memories_by_identity(
    conn, identity_id: str, include_aliases: bool, limit: Optional[int]
):
    return get_memories_by_identity(conn, identity_id, include_aliases, limit)


@_with_connection
def _get_identities_in_memory(conn, memory_id: int):
    return get_identities_in_memory(conn, memory_id)


# Workspace management wrappers
@_with_connection
def _list_workspaces(conn):
    return list_workspaces(conn)


@_with_connection
def _get_workspace_stats(conn, workspace: str):
    return get_workspace_stats(conn, workspace)


@_with_connection(writes=True)
def _move_memories_to_workspace(conn, memory_ids: List[int], target_workspace: str):
    return move_memories_to_workspace(conn, memory_ids, target_workspace)


@_with_connection(writes=True)
def _delete_workspace(conn, workspace: str, delete_memories: bool):
    return delete_workspace(conn, workspace, delete_memories)


@mcp.tool()
async def memory_identity_create(
    canonical_id: str,
    display_name: str,
    entity_type: str = "person",
    aliases: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a canonical identity with optional aliases.

    An identity represents a unique entity (person, organization, project, etc.)
    that can be referenced across multiple memories. Use identities to unify
    references to the same entity across different contexts.

    Args:
        canonical_id: Unique identifier (e.g., "user:ronaldo", "org:acme", "project:memora")
        display_name: Human-readable name
        entity_type: Type of entity - person, organization, project, tool, concept, other
        aliases: Alternative names/IDs (e.g., ["@ronaldo", "ronaldo@email.com"])
        metadata: Additional structured data about the identity

    Returns:
        Created identity with canonical_id, display_name, entity_type, aliases

    Example:
        >>> await memory_identity_create(
        ...     canonical_id="user:ronaldo",
        ...     display_name="Ronaldo Lima",
        ...     entity_type="person",
        ...     aliases=["@ronaldo", "limaronaldo"]
        ... )
    """
    try:
        return _create_identity(
            canonical_id, display_name, entity_type, aliases, metadata
        )
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}


@mcp.tool()
async def memory_identity_get(canonical_id: str) -> Dict[str, Any]:
    """Get an identity by its canonical ID.

    Args:
        canonical_id: The identity's unique identifier

    Returns:
        Identity with display_name, entity_type, aliases, metadata, timestamps
    """
    identity = _get_identity(canonical_id)
    if not identity:
        return {"error": "not_found", "canonical_id": canonical_id}
    return {"identity": identity}


@mcp.tool()
async def memory_identity_update(
    canonical_id: str,
    display_name: Optional[str] = None,
    entity_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Update an identity's properties.

    Args:
        canonical_id: The identity to update
        display_name: New display name (optional)
        entity_type: New entity type (optional)
        metadata: New metadata (optional, replaces existing)

    Returns:
        Updated identity
    """
    try:
        identity = _update_identity(canonical_id, display_name, entity_type, metadata)
        if not identity:
            return {"error": "not_found", "canonical_id": canonical_id}
        return {"identity": identity}
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}


@mcp.tool()
async def memory_identity_delete(canonical_id: str) -> Dict[str, Any]:
    """Delete an identity and all its links to memories.

    This removes the identity, its aliases, and all links to memories.
    The memories themselves are NOT deleted.

    Args:
        canonical_id: The identity to delete

    Returns:
        Dict with deletion counts (deleted, links_removed, aliases_removed)
    """
    return _delete_identity(canonical_id)


@mcp.tool()
async def memory_identity_list(
    entity_type: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    """List all identities.

    Args:
        entity_type: Filter by entity type (person, organization, project, etc.)
        limit: Maximum identities to return
        offset: Offset for pagination

    Returns:
        Dict with count and list of identities (includes memory_count for each)
    """
    identities = _list_identities(entity_type, limit, offset)
    return {"count": len(identities), "identities": identities}


@mcp.tool()
async def memory_identity_search(
    query: str,
    entity_type: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Search identities by name or alias.

    Args:
        query: Search query (matches display_name, canonical_id, or aliases)
        entity_type: Optional filter by entity type
        limit: Maximum results (default: 10)

    Returns:
        Dict with matching identities
    """
    results = _search_identities(query, entity_type, limit)
    return {"count": len(results), "identities": results}


@mcp.tool()
async def memory_identity_add_alias(
    canonical_id: str,
    alias: str,
    source: str = "manual",
) -> Dict[str, Any]:
    """Add an alias to an existing identity.

    Aliases allow the same identity to be found by different names.

    Args:
        canonical_id: The identity to add the alias to
        alias: The new alias (e.g., "@ronaldo", "ronaldo@email.com")
        source: Source of the alias (e.g., "github", "email", "manual")

    Returns:
        Dict with the added alias info
    """
    try:
        return _add_identity_alias(canonical_id, alias, source)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}


@mcp.tool()
async def memory_identity_link(
    memory_id: int,
    identity_id: str,
    mention_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Link a memory to an identity.

    Creates a bidirectional relationship - you can find memories by identity
    and identities mentioned in a memory.

    Args:
        memory_id: The memory to link
        identity_id: The identity (canonical_id or alias) to link to
        mention_text: Optional text that triggered the link (e.g., "@ronaldo")

    Returns:
        Dict with the link info (memory_id, identity_id, mention_text)
    """
    try:
        return _link_memory_to_identity(memory_id, identity_id, mention_text)
    except ValueError as exc:
        return {"error": "invalid_input", "message": str(exc)}


@mcp.tool()
async def memory_identity_unlink(
    memory_id: int,
    identity_id: str,
) -> Dict[str, Any]:
    """Remove a link between a memory and an identity.

    Args:
        memory_id: The memory to unlink
        identity_id: The identity to unlink from

    Returns:
        Dict with removed status
    """
    return _unlink_memory_from_identity(memory_id, identity_id)


@mcp.tool()
async def memory_search_by_identity(
    identity_id: str,
    include_aliases: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Find all memories linked to an identity.

    Args:
        identity_id: The identity (canonical_id or alias) to search for
        include_aliases: If True, also search by aliases (default: True)
        limit: Maximum memories to return

    Returns:
        Dict with memories that mention/are linked to this identity
    """
    memories = _get_memories_by_identity(identity_id, include_aliases, limit)
    return {"count": len(memories), "memories": memories}


@mcp.tool()
async def memory_get_identities(memory_id: int) -> Dict[str, Any]:
    """Get all identities linked to a memory.

    Args:
        memory_id: The memory ID

    Returns:
        Dict with identities mentioned in this memory
    """
    identities = _get_identities_in_memory(memory_id)
    return {"count": len(identities), "identities": identities}


# ============================================================================
# Workspace Management Tools
# ============================================================================


@mcp.tool()
async def memory_workspace_list() -> Dict[str, Any]:
    """List all workspaces with memory counts.

    Returns a list of all workspaces that have memories, ordered by memory count.
    Each workspace includes first and last memory timestamps.

    Returns:
        Dict with count and list of workspaces
    """
    workspaces = _list_workspaces()
    return {"count": len(workspaces), "workspaces": workspaces}


@mcp.tool()
async def memory_workspace_stats(workspace: str) -> Dict[str, Any]:
    """Get detailed statistics for a workspace.

    Args:
        workspace: Workspace name to get stats for

    Returns:
        Dict with workspace statistics including:
        - total_memories, daily_memories, permanent_memories
        - first_memory, last_memory timestamps
        - avg_importance score
        - top_tags with counts
    """
    return _get_workspace_stats(workspace)


@mcp.tool()
async def memory_workspace_move(
    memory_ids: List[int],
    target_workspace: str,
) -> Dict[str, Any]:
    """Move memories to a different workspace.

    Args:
        memory_ids: List of memory IDs to move
        target_workspace: Destination workspace name

    Returns:
        Dict with moved count and any not_found IDs
    """
    if not memory_ids:
        return {"error": "invalid_input", "message": "memory_ids cannot be empty"}
    result = _move_memories_to_workspace(memory_ids, target_workspace)
    if result.get("moved", 0) > 0:
        _schedule_cloud_graph_sync()
    return result


@mcp.tool()
async def memory_workspace_delete(
    workspace: str,
    delete_memories: bool = False,
) -> Dict[str, Any]:
    """Delete a workspace.

    If delete_memories is False (default), memories are moved to the default workspace.
    If delete_memories is True, all memories in the workspace are permanently deleted.

    Cannot delete the "default" workspace.

    Args:
        workspace: Workspace to delete
        delete_memories: If True, delete all memories. If False, move to default.

    Returns:
        Dict with deletion results
    """
    if workspace == DEFAULT_WORKSPACE:
        return {
            "error": "cannot_delete_default",
            "message": "Cannot delete the default workspace",
        }
    result = _delete_workspace(workspace, delete_memories)
    if result.get("memories_deleted", 0) > 0 or result.get(
        "memories_moved_to_default", 0
    ) > 0:
        _schedule_cloud_graph_sync()
    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Memory MCP Server")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Default: start server (make it the default if no subcommand)
    parser.add_argument(
        "--transport",
        choices=sorted(VALID_TRANSPORTS),
        default=DEFAULT_TRANSPORT,
        help="MCP transport to use (defaults to env MEMORA_TRANSPORT or 'stdio')",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host interface for HTTP transports (defaults to env MEMORA_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port for HTTP transports (defaults to env MEMORA_PORT or 8000)",
    )
    parser.add_argument(
        "--graph-port",
        type=int,
        default=DEFAULT_GRAPH_PORT,
        help="Port for graph visualization server (defaults to env MEMORA_GRAPH_PORT or 8765)",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable the graph visualization server",
    )

    # Subcommand: sync-pull
    sync_pull_parser = subparsers.add_parser(
        "sync-pull", help="Force pull database from cloud storage (ignore local cache)"
    )

    # Subcommand: sync-push
    sync_push_parser = subparsers.add_parser(
        "sync-push", help="Force push database to cloud storage"
    )

    # Subcommand: sync-status
    sync_status_parser = subparsers.add_parser(
        "sync-status", help="Show sync status and backend information"
    )

    # Subcommand: info
    info_parser = subparsers.add_parser("info", help="Show storage backend information")

    # Subcommand: migrate-images
    migrate_parser = subparsers.add_parser(
        "migrate-images", help="Migrate base64 images to R2 storage"
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )

    args = parser.parse_args(argv)

    # Handle subcommands
    if args.command == "sync-pull":
        _handle_sync_pull()
    elif args.command == "sync-push":
        _handle_sync_push()
    elif args.command == "sync-status":
        _handle_sync_status()
    elif args.command == "info":
        _handle_info()
    elif args.command == "migrate-images":
        _handle_migrate_images(dry_run=args.dry_run)
    else:
        # Default: start server
        mcp.settings.host = args.host
        mcp.settings.port = args.port

        # Pre-warm database connection (triggers cloud sync if needed)
        # This prevents "connection failed" on first MCP connection
        try:
            import sys

            print("Initializing database...", file=sys.stderr)
            conn = connect()
            conn.close()
            print("Database ready.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Database pre-warm failed: {e}", file=sys.stderr)

        # Start graph visualization server unless disabled
        if not args.no_graph:
            start_graph_server(args.host, args.graph_port)

        mcp.run(transport=args.transport)


def _handle_sync_pull() -> None:
    """Handle sync-pull command."""
    import json

    from .backends import CloudSQLiteBackend
    from .storage import STORAGE_BACKEND

    if not isinstance(STORAGE_BACKEND, CloudSQLiteBackend):
        print("Error: sync-pull only works with cloud storage backends")
        print(f"Current backend: {STORAGE_BACKEND.__class__.__name__}")
        exit(1)

    print(f"Pulling database from {STORAGE_BACKEND.cloud_url}...")
    try:
        STORAGE_BACKEND.force_sync_pull()
        info = STORAGE_BACKEND.get_info()
        print("✓ Sync completed successfully")
        print(f"  Cache path: {info['cache_path']}")
        print(f"  Size: {info['cache_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Last sync: {info.get('last_sync', 'N/A')}")
    except Exception as e:
        print(f"✗ Sync failed: {e}")
        exit(1)


def _handle_sync_push() -> None:
    """Handle sync-push command."""
    import json

    from .backends import CloudSQLiteBackend
    from .storage import STORAGE_BACKEND

    if not isinstance(STORAGE_BACKEND, CloudSQLiteBackend):
        print("Error: sync-push only works with cloud storage backends")
        print(f"Current backend: {STORAGE_BACKEND.__class__.__name__}")
        exit(1)

    print(f"Pushing database to {STORAGE_BACKEND.cloud_url}...")
    try:
        STORAGE_BACKEND.force_sync_push()
        info = STORAGE_BACKEND.get_info()
        print("✓ Push completed successfully")
        print(f"  Cloud URL: {info['cloud_url']}")
        print(f"  Size: {info['cache_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"  Last sync: {info.get('last_sync', 'N/A')}")
    except Exception as e:
        print(f"✗ Push failed: {e}")
        exit(1)


def _handle_sync_status() -> None:
    """Handle sync-status command."""
    import json

    from .backends import CloudSQLiteBackend
    from .storage import STORAGE_BACKEND

    info = STORAGE_BACKEND.get_info()
    backend_type = info.get("backend_type", "unknown")

    print(f"Storage Backend: {backend_type}")
    print()

    if backend_type == "cloud_sqlite":
        print(f"Cloud URL: {info.get('cloud_url', 'N/A')}")
        print(f"Bucket: {info.get('bucket', 'N/A')}")
        print(f"Key: {info.get('key', 'N/A')}")
        print()
        print(f"Cache Path: {info.get('cache_path', 'N/A')}")
        print(f"Cache Exists: {info.get('cache_exists', False)}")
        print(f"Cache Size: {info.get('cache_size_bytes', 0) / 1024 / 1024:.2f} MB")
        print()
        print(f"Is Dirty: {info.get('is_dirty', False)}")
        print(f"Last ETag: {info.get('last_etag', 'N/A')}")
        print(f"Last Sync: {info.get('last_sync', 'N/A')}")
        print(f"Auto Sync: {info.get('auto_sync', True)}")
        print(f"Encryption: {info.get('encrypt', False)}")
    elif backend_type == "local_sqlite":
        print(f"Database Path: {info.get('db_path', 'N/A')}")
        print(f"Exists: {info.get('exists', False)}")
        print(f"Size: {info.get('size_bytes', 0) / 1024 / 1024:.2f} MB")
    else:
        print(json.dumps(info, indent=2))


def _handle_info() -> None:
    """Handle info command."""
    import json

    from .storage import STORAGE_BACKEND

    info = STORAGE_BACKEND.get_info()
    print(json.dumps(info, indent=2, default=str))


def _handle_migrate_images(dry_run: bool = False) -> None:
    """Handle migrate-images command."""
    import json

    print(f"{'[DRY RUN] ' if dry_run else ''}Migrating base64 images to R2 storage...")

    result = _migrate_images_to_r2(dry_run=dry_run)

    if "error" in result:
        print(f"Error: {result['message']}")
        return

    print(json.dumps(result, indent=2))

    if result.get("errors"):
        print(f"\nWarning: {len(result['errors'])} errors occurred during migration")


if __name__ == "__main__":
    main()
