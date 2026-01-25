/**
 * GET /api/graph - Returns graph nodes, edges, and metadata for visualization
 * Supports ?db=memora or ?db=ob1 parameter to select database
 */

interface Env {
  DB_MEMORA: D1Database;
  DB_OB1: D1Database;
  MIN_EDGE_SCORE?: string;
  DEFAULT_DB?: string;
  DB_CONFIG?: string;
}

function getDatabase(env: Env, dbName: string | null): D1Database {
  const name = dbName || env.DEFAULT_DB || "memora";
  if (name === "ob1") return env.DB_OB1;
  return env.DB_MEMORA;
}

interface Memory {
  id: number;
  content: string;
  metadata: string;
  tags: string;
  created_at: string;
  updated_at: string | null;
}

interface CrossRef {
  memory_id: number;
  related: string;
}

interface GraphNode {
  id: number;
  label: string;
  title: string;
  color: string | { background: string; border: string };
  size: number;
  mass: number;
  borderWidth?: number;
  shape?: string;
}

interface GraphEdge {
  id: number;
  from: number;
  to: number;
}

// Tag colors (purple palette)
const TAG_COLORS = [
  "#a855f7", "#c084fc", "#d8b4fe", "#9333ea",
  "#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd"
];

// Status colors for issues
const ISSUE_STATUS_COLORS: Record<string, string> = {
  "open": "#ff7b72",
  "closed:complete": "#7ee787",
  "closed:not_planned": "#8b949e",
};

// Status colors for TODOs
const TODO_STATUS_COLORS: Record<string, string> = {
  "open": "#58a6ff",
  "closed:complete": "#7ee787",
  "closed:not_planned": "#8b949e",
};

const DUPLICATE_THRESHOLD = 0.85;

function parseJson<T>(str: string | null, defaultValue: T): T {
  if (!str) return defaultValue;
  try {
    return JSON.parse(str);
  } catch {
    return defaultValue;
  }
}

function isSection(metadata: Record<string, unknown> | null): boolean {
  return metadata?.type === "section";
}

function isIssue(metadata: Record<string, unknown> | null): boolean {
  return metadata?.type === "issue";
}

function isTodo(metadata: Record<string, unknown> | null): boolean {
  return metadata?.type === "todo";
}

function getIssueStatus(metadata: Record<string, unknown>): string {
  const status = (metadata.status as string) || "open";
  if (status === "resolved") return "closed:complete";
  if (status === "wontfix") return "closed:not_planned";
  if (status === "in_progress") return "open";
  if (status === "closed") {
    const reason = (metadata.closed_reason as string) || "complete";
    return `closed:${reason}`;
  }
  return status;
}

function getTodoStatus(metadata: Record<string, unknown>): string {
  const status = (metadata.status as string) || "open";
  if (status === "completed") return "closed:complete";
  if (status === "blocked") return "closed:not_planned";
  if (status === "in_progress") return "open";
  if (status === "closed") {
    const reason = (metadata.closed_reason as string) || "complete";
    return `closed:${reason}`;
  }
  return status;
}

export const onRequestGet: PagesFunction<Env> = async ({ env, request }) => {
  const url = new URL(request.url);
  const dbName = url.searchParams.get("db");
  const db = getDatabase(env, dbName);
  const minScore = parseFloat(env.MIN_EDGE_SCORE || "0.40");

  // Fetch all memories
  const memoriesResult = await db.prepare(
    "SELECT id, content, metadata, tags, created_at, updated_at FROM memories"
  ).all<Memory>();

  if (!memoriesResult.results || memoriesResult.results.length === 0) {
    return Response.json({ error: "no_memories", message: "No memories to visualize" });
  }

  const memories = memoriesResult.results;

  // Fetch all crossrefs
  const crossrefsResult = await db.prepare(
    "SELECT memory_id, related FROM memories_crossrefs"
  ).all<CrossRef>();

  const crossrefsMap = new Map<number, Array<{ id: number; score: number }>>();
  for (const cr of crossrefsResult.results || []) {
    const related = parseJson<Array<{ id: number; score: number }>>(cr.related, []);
    crossrefsMap.set(cr.memory_id, related);
  }

  // Build edges
  const edges: GraphEdge[] = [];
  const seen = new Set<string>();
  let edgeId = 0;

  for (const m of memories) {
    const refs = crossrefsMap.get(m.id) || [];
    for (const ref of refs) {
      if (ref.score <= minScore) continue;
      const edgeKey = [Math.min(m.id, ref.id), Math.max(m.id, ref.id)].join("-");
      if (!seen.has(edgeKey)) {
        seen.add(edgeKey);
        edges.push({ id: edgeId++, from: m.id, to: ref.id });
      }
    }
  }

  // Count connections per node
  const connectionCounts = new Map<number, number>();
  for (const edge of edges) {
    connectionCounts.set(edge.from, (connectionCounts.get(edge.from) || 0) + 1);
    connectionCounts.set(edge.to, (connectionCounts.get(edge.to) || 0) + 1);
  }

  // Find duplicates
  const memoryIds = new Set(memories.filter(m => !isSection(parseJson(m.metadata, null))).map(m => m.id));
  const duplicateIds = new Set<number>();

  for (const m of memories) {
    const meta = parseJson<Record<string, unknown>>(m.metadata, {});
    if (isSection(meta)) continue;

    const refs = crossrefsMap.get(m.id) || [];
    for (const ref of refs) {
      if (ref.score >= DUPLICATE_THRESHOLD && memoryIds.has(ref.id)) {
        duplicateIds.add(m.id);
        duplicateIds.add(ref.id);
      }
    }
  }

  // Build tag colors
  const tagColors: Record<string, string> = {};
  for (const m of memories) {
    const tags = parseJson<string[]>(m.tags, []);
    const primaryTag = tags[0] || "untagged";
    if (!(primaryTag in tagColors)) {
      tagColors[primaryTag] = TAG_COLORS[Object.keys(tagColors).length % TAG_COLORS.length];
    }
  }

  // Build nodes
  const nodes: GraphNode[] = [];
  for (const m of memories) {
    const meta = parseJson<Record<string, unknown>>(m.metadata, {});

    // Skip section memories
    if (isSection(meta)) continue;

    const tags = parseJson<string[]>(m.tags, []);
    const primaryTag = tags[0] || "untagged";
    const content = m.content;

    const firstLine = content.split("\n")[0].replace(/^#+\s*/, "").trim().slice(0, 60);
    const headline = firstLine.replace(/"/g, "'").replace(/\\/g, "");
    const label = content.slice(0, 35).replace(/[\n#*_`[\]]/g, " ").trim().replace(/"/g, "'").replace(/\\/g, "");

    // Calculate node size based on connections
    const connections = connectionCounts.get(m.id) || 0;
    const nodeSize = 12 + Math.min(28, Math.floor(Math.log1p(connections) * 8));
    const nodeMass = 0.5 + Math.min(2.5, Math.log1p(connections) * 0.8);

    // Build title with type indicator
    let typeLabel = "";
    if (isIssue(meta)) typeLabel = " - Issue";
    else if (isTodo(meta)) typeLabel = " - TODO";

    const node: GraphNode = {
      id: m.id,
      label: label.length > 35 ? label + "..." : label,
      title: `#${m.id}${typeLabel}\n${headline}`,
      color: tagColors[primaryTag],
      size: nodeSize,
      mass: nodeMass,
    };

    // Apply issue-specific styling
    if (isIssue(meta)) {
      const status = getIssueStatus(meta);
      node.shape = "dot";
      node.color = ISSUE_STATUS_COLORS[status] || ISSUE_STATUS_COLORS["open"];
      if (meta.severity === "critical") {
        node.borderWidth = 4;
      }
    }

    // Apply TODO-specific styling
    if (isTodo(meta)) {
      const status = getTodoStatus(meta);
      node.shape = "dot";
      node.color = TODO_STATUS_COLORS[status] || TODO_STATUS_COLORS["open"];
      if (meta.priority === "high") {
        node.borderWidth = 4;
      }
    }

    // Apply duplicate indicator
    if (duplicateIds.has(m.id)) {
      node.color = {
        background: typeof node.color === "string" ? node.color : "#a855f7",
        border: "#f85149",
      };
      node.borderWidth = 3;
    }

    nodes.push(node);
  }

  // Build mappings
  const tagToNodes: Record<string, number[]> = {};
  const sectionToNodes: Record<string, number[]> = {};
  const subsectionToNodes: Record<string, number[]> = {};
  const statusToNodes: Record<string, number[]> = {};
  const issueCategoryToNodes: Record<string, number[]> = {};
  const todoStatusToNodes: Record<string, number[]> = {};
  const todoCategoryToNodes: Record<string, number[]> = {};
  const nodeTimestamps: Record<number, string> = {};

  let minDate = "";
  let maxDate = "";
  const dates: string[] = [];

  for (const m of memories) {
    const meta = parseJson<Record<string, unknown>>(m.metadata, {});
    const tags = parseJson<string[]>(m.tags, []);

    // Skip sections for mappings
    if (isSection(meta)) continue;

    // Tags mapping
    for (const tag of tags) {
      if (!tagToNodes[tag]) tagToNodes[tag] = [];
      tagToNodes[tag].push(m.id);
    }

    // Issue mappings
    if (isIssue(meta)) {
      const status = getIssueStatus(meta);
      if (!statusToNodes[status]) statusToNodes[status] = [];
      statusToNodes[status].push(m.id);

      const component = (meta.component as string) || "uncategorized";
      if (!issueCategoryToNodes[component]) issueCategoryToNodes[component] = [];
      issueCategoryToNodes[component].push(m.id);
    }

    // TODO mappings
    if (isTodo(meta)) {
      const status = getTodoStatus(meta);
      if (!todoStatusToNodes[status]) todoStatusToNodes[status] = [];
      todoStatusToNodes[status].push(m.id);

      const category = (meta.category as string) || "uncategorized";
      if (!todoCategoryToNodes[category]) todoCategoryToNodes[category] = [];
      todoCategoryToNodes[category].push(m.id);
    }

    // Section mappings (skip issues and TODOs)
    if (!isIssue(meta) && !isTodo(meta)) {
      const hierarchy = meta.hierarchy as { path?: string[] } | undefined;
      let section = "Uncategorized";
      let parts: string[] = [];

      if (hierarchy?.path?.length) {
        section = hierarchy.path[0];
        parts = hierarchy.path.slice(1);
      } else {
        section = (meta.section as string) || "Uncategorized";
        const subsection = meta.subsection as string;
        if (subsection) parts = subsection.split("/");
      }

      if (!sectionToNodes[section]) sectionToNodes[section] = [];
      sectionToNodes[section].push(m.id);

      if (parts.length) {
        for (let i = 0; i < parts.length; i++) {
          const partialPath = parts.slice(0, i + 1).join("/");
          const fullKey = `${section}/${partialPath}`;
          if (!subsectionToNodes[fullKey]) subsectionToNodes[fullKey] = [];
          subsectionToNodes[fullKey].push(m.id);
        }
      }
    }

    // Timeline data
    if (m.created_at) {
      nodeTimestamps[m.id] = m.created_at;
      dates.push(m.created_at);
    }
  }

  if (dates.length) {
    dates.sort();
    minDate = dates[0];
    maxDate = dates[dates.length - 1];
  }

  return Response.json({
    nodes,
    edges,
    tagColors,
    tagToNodes,
    sectionToNodes,
    subsectionToNodes,
    statusToNodes,
    issueCategoryToNodes,
    todoStatusToNodes,
    todoCategoryToNodes,
    duplicateIds: Array.from(duplicateIds),
    nodeTimestamps,
    minDate,
    maxDate,
  });
};
