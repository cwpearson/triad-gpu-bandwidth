# unified-memory-microbench

### For help

`./triad -h`

### To skip the system allocator (on systems without ATS)

`./triad --no-system`

### To do multiple sizes

`./triad -g 1 -g 2`

### To do multiple iterations

`./triad -n 6`

### To cause tee to output right away

`stdbuf -o 0 ./triad | tee triad.csv`

### To throw away stderr

`./triad 2>/dev/null`