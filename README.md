# unified-memory-microbench

## Building on Power9 with gcc 4.8.5

```
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=`readlink -f ../toolchains/clang.toolchain`
```

## Building with a gcc that has std::regex

```
mkdir build && cd build
cmake ..
```
