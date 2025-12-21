# Privacy and Integrity (Current Behavior)

The current codebase provides only optional per-datagram protection:

- Integrity/auth: HMAC-SHA256 over each datagram when `auth_key` is set.
- Confidentiality: AES-GCM per datagram when `enc_key` is set.

These are purely local configuration knobs. There is no key exchange, no ACL,
no identity layer, and no privacy budget system in this repository.

## What Exists
- `holo.net.transport.iter_chunk_datagrams` supports `auth_key` and `enc_key`.
- `holo.net.mesh.MeshNode` accepts `auth_key` and `enc_key` for send/receive.
- AES-GCM requires the `cryptography` package at runtime.

## What Does Not Exist (Yet)
- Access control or authorization decisions.
- Per-peer scopes or policy enforcement.
- Resolver chains or privacy budgets.

If you need those features, design them above the current transport and keep
moving designs in the Wiki until they are stable enough to version here.
