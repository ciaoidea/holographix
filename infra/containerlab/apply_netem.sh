#!/usr/bin/env bash
set -euo pipefail

# Apply a netem profile to a given node interface.
# Example: ./apply_netem.sh r1 eth2 "loss 10% 30% 25"

node=$1
iface=$2
shift 2
profile=$*

if [[ -z "$profile" ]]; then
  echo "Usage: $0 <node> <iface> <netem-args>" >&2
  exit 1
fi

cname="clab-${CLAB_LAB_NAME:-holo-lab}-${node}"

docker exec "$cname" tc qdisc del dev "$iface" root 2>/dev/null || true
docker exec "$cname" tc qdisc add dev "$iface" root netem $profile

echo "Applied netem '$profile' to ${cname}:${iface}"
