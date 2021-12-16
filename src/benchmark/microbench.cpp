#include "./benchmark.h"

int main(int argc, char **argv) {
    auto flags = parse_flags(argc, argv);
    auto data_type = get_with_default(flags, "data_type", "uint64_t");
    if(data_type == "uint64_t") {
        Benchmark <uint64_t, uint64_t> bench;
        bench.parse_args(argc, argv);
        bench.run_benchmark();
    } else if(data_type == "uint32_t") {
        Benchmark <uint32_t, uint64_t> bench;
        bench.parse_args(argc, argv);
        bench.run_benchmark();
    }
//    bench.correctness_test();
}

