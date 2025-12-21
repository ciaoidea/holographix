# Mesh Control Plane (INV/WANT)

This document defines the control-plane datagrams used by `holo.net.mesh`.
They advertise which chunk IDs a peer has and request missing chunks.

## Control Datagram Format (HOCT)
Header struct: `>4sB16sH` (big-endian)

```
0   4  magic      = "HOCT"
4   1  ctrl_type  = 1 (INV) or 2 (WANT)
5   16 content_id (16 bytes)
21  2  count      = number of chunk IDs (u16)
```

Body:
- `count` entries of `u16` chunk IDs (`>H`), sorted and deduplicated.
- Maximum entries per datagram: 64 (`MAX_CTRL_CHUNKS`).

## Behavior in MeshNode
- `send_inventory(content_id)` sends INV to peers with the chunk IDs present
  under the local store path for that content.
- On INV receipt: the node stores the peer inventory and sends WANT for the
  first missing chunk IDs (up to 64).
- On WANT receipt: the node sends the requested chunks (if present).

No session state, retransmissions, or ACKs are implemented. INV/WANT is
opportunistic.

## Limitations
- Chunk IDs in INV/WANT are 16-bit values. IDs above 65535 cannot be requested.
- Recovery chunks are sent with IDs >= 1,000,000 in the mesh; INV/WANT cannot
  represent them, so they must be sent proactively.
- Chunk ID is a transport identifier. The `block_id` used for reconstruction is
  stored inside each chunk header and may not equal the transport chunk ID
  (notably for olonomic audio v3).

## Related Code
- `src/holo/net/transport.py`
- `src/holo/net/mesh.py`
