# Nichess

https://www.nichess.org

Requires C++17 and CMake 3.14 or newer.

### How to run

After cloning the repository, inside nichess-cpp run:

```
git submodule update --init
```

Build:

```
cmake -S . -B build
cmake --build build
```

Play:

```
./build/play_vs_agent
```
