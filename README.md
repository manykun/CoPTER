# Dependence

```shell
sudo apt update
sudo apt install libzmq5 libzmq3-dev
sudo apt install protobuf-compiler==3.20.3 libprotobuf-dev==3.20.3
```

```shell
./waf configure --enable-examples --enable-mpi
./waf build
pip3 install --user ./contrib/opengym/model/ns3gym
```

## Code study guide

For a detailed Chinese walkthrough of the project architecture, ACC/CoPTER,
M3-related assets, SOR, ns-3 integration, experiment scripts, traffic generators,
and analysis tools, see [docs/code-study/README.md](docs/code-study/README.md).

For the reproducible three-scenario ACC effectiveness workflow, fixed-action
sanity checks, ablations, commands, and decision gates, see
[docs/acc-validation-experiment.md](docs/acc-validation-experiment.md).

For the Chinese research-progress report covering paper baselines, ACC
effectiveness, the negative continual-learning result, reward diagnosis,
tail-safe reward redesign, registered gates, and next objectives, see
[docs/research-progress-acc-sor.md](docs/research-progress-acc-sor.md).

For the controlled ACC multiscale-action and same-path steady-to-burst
catastrophic-forgetting experiment, see
[docs/acc-forgetting-multiscale-experiment.md](docs/acc-forgetting-multiscale-experiment.md).
