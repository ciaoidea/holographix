# Containerlab baseline lab

Minimal reproducible lab to test HolographiX vs a baseline over impaired links.

Topology (`holo-lab.clab.yml`):
```
h1 (python host) -- r1 (FRR) -- r2 (FRR) -- h2 (python host)
```
- h1/h2 run the HolographiX sender/receiver and baseline app.
- r1/r2 route via OSPF; impairments applied with `tc netem` on interfaces (e.g., r1 eth2).

## Bring up the lab
```
containerlab deploy -t infra/containerlab/holo-lab.clab.yml
infra/containerlab/init_hosts.sh
```

## Apply impairments
Examples (loss/jitter/burst applied inside containers):
```
# 10% loss with correlation
infra/containerlab/apply_netem.sh r1 eth2 "loss 10% 30%"

# 80ms delay with 20ms jitter
infra/containerlab/apply_netem.sh r1 eth2 "delay 80ms 20ms"

# bursty loss (GE-like)
infra/containerlab/apply_netem.sh r1 eth2 "loss 5% 50% 25"
```
Remove impairments:
```
infra/containerlab/apply_netem.sh r1 eth2 "delay 0ms"
docker exec clab-holo-lab-r1 tc qdisc del dev eth2 root
```

## Run HolographiX vs baseline
- Inside `h1`/`h2` containers (python:3.11-slim), install deps: `pip install /workspace` or bind-mount the repo when deploying.
- Generate chunks once (from `src/`): `python3 -m holo flower.jpg 1 --packet-bytes 1136 --coarse-side 16` (creates `flower.jpg.holo/`, generated locally).
- HolographiX send: `python3 examples/holo_mesh_sender.py --uri holo://demo/flower --chunk-dir flower.jpg.holo --peer 10.10.3.2:5000`
- HolographiX receive/decode: `python3 examples/holo_mesh_receiver.py --listen 0.0.0.0:5000 --out-dir cortex_rx --decode recon.png`
- Baseline UDP send: `python3 examples/baseline_udp_sender.py --file flower.jpg --peer 10.10.3.2:6000`
- Baseline UDP receive: `python3 examples/baseline_udp_receiver.py --listen 0.0.0.0:6000 --out baseline.jpg --duration 5`
- Capture pcaps on r1/r2 with `tcpdump` for analysis.

Key metrics to extract from captures/logs:
- Time to first usable percept (Holo vs baseline).
- PSNR/SNR vs delivered bits (use `examples/psnr_benchmark.py` / `examples/snr_benchmark_audio.py` on reconstructions).
- Degradation curve when dropping X% packets; recovery after reroute.

## Teardown
```
containerlab destroy -t infra/containerlab/holo-lab.clab.yml --cleanup
```

## Next step: ns-3 microscope
Once the containerlab scenarios are solid, port a single wireless/mobility case to ns-3 (simple UDP app carrying Holo chunks) to stress fading/handover. Keep it separate; containerlab remains the main home.
