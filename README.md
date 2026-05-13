# CoPTER

## Dependence

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
