#!/bin/bash
# Sync memora data to D1, loading environment from .mcp.json
#
# Usage:
#   ./scripts/sync.sh           # Local D1
#   ./scripts/sync.sh --remote  # Production D1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MEMORA_ROOT="$(dirname "$PROJECT_ROOT")"
MCP_CONFIG="$MEMORA_ROOT/.mcp.json"

if [ ! -f "$MCP_CONFIG" ]; then
    echo "Error: .mcp.json not found at $MCP_CONFIG"
    exit 1
fi

# Extract environment variables from .mcp.json using python
ENV_VARS=$(python3 -c "
import json
import sys

with open('$MCP_CONFIG') as f:
    config = json.load(f)

env = config.get('mcpServers', {}).get('memora', {}).get('env', {})
for key, value in env.items():
    # Only export storage-related vars
    if key.startswith(('AWS_', 'MEMORA_STORAGE', 'MEMORA_CLOUD')):
        print(f'export {key}=\"{value}\"')
")

if [ -z "$ENV_VARS" ]; then
    echo "Error: Could not extract environment from .mcp.json"
    exit 1
fi

# Export the variables
eval "$ENV_VARS"

echo "Loaded environment from .mcp.json:"
echo "  MEMORA_STORAGE_URI=$MEMORA_STORAGE_URI"
echo "  AWS_PROFILE=$AWS_PROFILE"
echo ""

# Run the sync script
cd "$PROJECT_ROOT"
python scripts/sync-to-d1.py "$@"

# Notify connected clients via WebSocket worker (only for remote sync)
if [[ "$*" == *"--remote"* ]]; then
    echo ""
    echo "Notifying connected clients..."
    curl -s -X POST "https://memora-graph-sync.cloudflare-strategic612.workers.dev/broadcast" \
        -H "Content-Type: application/json" \
        -d '{}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"  Broadcast sent to {d.get('sent',0)} clients\")" 2>/dev/null || echo "  (webhook notification skipped)"
fi
