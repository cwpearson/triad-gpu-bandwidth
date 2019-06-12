# unified-memory-microbench

## Building on Power9 with gcc 4.8.5

```
cmake .. -DCMAKE_TOOLCHAIN_FILE=`readlink -f ../toolchains/clang.toolchain`
```

## Building with a gcc that has std::regex

cmake ..
