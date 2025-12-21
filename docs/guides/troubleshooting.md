# Troubleshooting

## No chunks received
Cause:
- UDP blocked by firewall/NAT, or wrong `--peer`/`--bind`.
Fix:
- Verify ports and reachability; start with loopback tests.
- Use matching ports on both sides.

## Receiver shows `mac_fail` > 0
Cause:
- HMAC keys do not match.
Fix:
- Use the same `--auth-key` on all peers.

## Encrypted datagrams never decode
Cause:
- AES-GCM keys or key IDs do not match, or `cryptography` missing.
Fix:
- Use the same `--enc-key` and `--enc-key-id` on all peers.
- Install `cryptography` if encryption is enabled.

## Decode fails with "No chunk_*.holo found"
Cause:
- Wrong directory or empty store.
Fix:
- Confirm `.holo` directory exists and contains `chunk_*.holo`.
- Ensure content is seeded or received before decoding.

## Chunks fragment too much / high loss
Cause:
- `--max-payload` too large for the path MTU, or chunks too large.
Fix:
- Lower `--max-payload` (e.g., 1024-1136).
- For images, set `--packet-bytes` when encoding.
- For audio, lower `TARGET_KB` or increase `--blocks`.

## INV/WANT does not fetch recovery chunks
Cause:
- INV/WANT uses 16-bit chunk IDs; recovery IDs are >= 1,000,000.
Fix:
- Send `recovery_*.holo` proactively (sender or mesh-broadcast).

## TNC decode yields no chunks
Cause:
- Wrong WAV format, sample rate, or baud/preamble mismatch.
Fix:
- Convert with `python -m holo tnc-wav-fix`.
- Match `--baud` and `--preamble-len` on TX/RX.
- Use `--raw-s16le` with correct `--raw-fs` for raw audio.
