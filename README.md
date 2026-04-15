```
gcc -O2 -Iinclude -o app main_host.c input_tensor.c src/features_0_weight.c src/features_0_bias.c runner.c ops.c tensor.c -lm
./app
```
channel 0
output shape = [1, 1, 111, 111]
output memory = 49284 bytes (48.13 KB, 0.05 MB)
referenced flash input = 599424 bytes (585.38 KB, 0.57 MB)
RAM working set ~= 49284 bytes (48.13 KB, 0.05 MB)
execution time = 8.745 ms


channel 0
output shape = [1, 1, 111, 111]
output memory = 49284 bytes (48.13 KB, 0.05 MB)
referenced flash input = 599424 bytes (585.38 KB, 0.57 MB)
RAM working set ~= 49284 bytes (48.13 KB, 0.05 MB)
execution time = 290.000 ms
