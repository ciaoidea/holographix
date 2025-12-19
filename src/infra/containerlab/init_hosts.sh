#!/usr/bin/env bash
set -euo pipefail

# Initialize host IPs/routes inside the containerlab lab.
# Assumes lab name "holo-lab" from holo-lab.clab.yml.

lab=${CLAB_LAB_NAME:-holo-lab}

declare -A host_if
host_if[h1]="eth1 10.10.1.2/24 10.10.1.1"
host_if[h2]="eth1 10.10.3.2/24 10.10.3.1"

for host in "${!host_if[@]}"; do
  iface=$(awk '{print $1}' <<<"${host_if[$host]}")
  ipcidr=$(awk '{print $2}' <<<"${host_if[$host]}")
  gw=$(awk '{print $3}' <<<"${host_if[$host]}")

  cname="clab-${lab}-${host}"
  docker exec "$cname" ip addr add "$ipcidr" dev "$iface" || true
  docker exec "$cname" ip link set "$iface" up
  docker exec "$cname" ip route replace default via "$gw"
done

echo "Hosts initialized for lab ${lab}."
