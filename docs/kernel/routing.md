# Routing and Gossip (Current Behavior)

HolographiX currently uses explicit peer lists and a minimal gossip loop. There
is no global routing system or resolver chain in this repository.

## MeshNode
`holo.net.mesh.MeshNode`:
- Sends and receives chunks to explicit UDP peers.
- Supports INV/WANT control-plane datagrams.
- Optionally joins IPv4 multicast groups.

## Default Gossip Loop
`src/examples/holo_mesh_node.py` provides the canonical loop:
- Receives datagrams, stores completed chunks.
- Forwards a fraction of newly completed chunks (`--forward-prob`).
- Periodically re-radiates a random stored chunk (`--rate-hz`).
- Periodically sends inventory to peers (default: 1s).

This loop is intentionally simple and intended to be replaced or extended by
higher-level routing logic outside this repository.

## What Is Not Implemented
- Resolver chains or content routing policies.
- Structured overlays, DHTs, or gain-based global routing.
- Session tracking or congestion control beyond simple rate limits.
