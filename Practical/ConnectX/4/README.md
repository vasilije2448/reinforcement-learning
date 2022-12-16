# AlphaZero

Work in progress. TODO:

1) Currently I run episodes independently on each CPU. That's slow because GPU is making predictions 1 by 1, and each process consumes a lot of RAM. It's much better to have 1 GPU worker and run everything in parallel like [here](https://github.com/bhansconnect/alphazero-pybind11).

2) Improve neural net training.

3) Better exploration.

4) Rewrite in C++?

#### Known issues:

```
RuntimeError: CUDA error: unknown error
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below
might be incorrect.
```

Not sure what's causing this bug, but stopping the program and running it again solves the
issue (might take a few tries). Lowering num_cpu in TrainArgs also seems to help.

#### Useful:

1) [AlphaGo Zero paper](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)

2) [AlphaZero paper (chess)](https://arxiv.org/pdf/1712.01815.pdf)

3) [AlphaZero Simple](https://joshvarty.github.io/AlphaZero/) - Step by step tutorial for Connect2.

4) [AlphaZero Pybind11](https://github.com/bhansconnect/alphazero-pybind11) - Probably the best general open source Alpha Zero implementation. There's [another one](https://jonathan-laurent.github.io/AlphaZero.jl/stable/) written in Julia that looks promising but I haven't tested it.
