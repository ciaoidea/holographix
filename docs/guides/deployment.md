# Deployment (systemd)

Sample systemd units live in `src/systemd/`.
They assume the repository is installed at `/opt/holographix` and run the example
scripts from `src/examples/`.

## 1) Install the Repo
Either:
- clone/copy to `/opt/holographix`, or
- edit the unit files to point to your path.

## 2) Configure Environment Files
Copy and edit the env files:

```bash
sudo cp src/systemd/holo_mesh_node.env /etc/default/holo_mesh_node
sudo cp src/systemd/holo_mesh_sender.env /etc/default/holo_mesh_sender
sudo cp src/systemd/holo_mesh_receiver.env /etc/default/holo_mesh_receiver
```

Set bind addresses, peers, store paths, and optional `--auth-key` or `--enc-key`.

## 3) Install Unit Files

```bash
sudo cp src/systemd/holo_mesh_node.service /etc/systemd/system/
sudo cp src/systemd/holo_mesh_sender.service /etc/systemd/system/
sudo cp src/systemd/holo_mesh_receiver.service /etc/systemd/system/
```

## 4) Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now holo_mesh_node
# optionally:
sudo systemctl enable --now holo_mesh_sender
sudo systemctl enable --now holo_mesh_receiver
```

## Notes
- Units set `PYTHONPATH=/opt/holographix/src`. If you install via pip, update
  the unit files to run the installed module instead.
- `--max-payload` controls UDP datagram size. Keep it below your path MTU.
- The receiver unit (`holo_mesh_receiver`) decodes on exit; set `HXR_DURATION`
  to control how long it listens before writing the output.
