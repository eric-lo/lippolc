#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <thread>
#include <ctime>

#include "omp.h"
#include "tbb/parallel_sort.h"
#include "flags.h"
#include "utils.h"

#include"../core/lipp.h"

template<typename KEY_TYPE, typename PAYLOAD_TYPE>
class Benchmark {

    LIPP <KEY_TYPE, PAYLOAD_TYPE> index;

    enum Operation {
        READ = 0, INSERT, DELETE, SCAN, UPDATE
    };

    // parameters
    double read_ratio = 1;
    double insert_ratio = 0;
    double delete_ratio = 0;
    double update_ratio = 0;
    double scan_ratio = 0;
    size_t scan_num = 100;
    size_t operations_num;
    long long table_size = -1;
    size_t init_table_size;
    double init_table_ratio;
    size_t thread_num;
    std::string index_type;
    std::string keys_file_path;
    std::string keys_file_type;
    std::string sample_distribution;
    bool latency_sample = false;
    double latency_sample_ratio = 0.001;
    int model_num;
    int error_bound;
    std::string output_path;
    double pgm_metric;
    size_t random_seed;
    double variance;
    double mse;
    bool memory_record;
    bool dataset_statistic;
    bool total_shuffle;
    bool workload_shift_test;
    double write_ratio_begin;
    double write_ratio_end;
    bool data_shift = false;
    bool hyper_thread = false;

    std::vector <KEY_TYPE> init_keys;
    KEY_TYPE *keys;
    std::pair <KEY_TYPE, PAYLOAD_TYPE> *init_key_values;
    std::vector <std::pair<Operation, KEY_TYPE>> operations;
    volatile bool running = false;
    std::atomic <size_t> ready_threads = 0;
    std::mt19937 gen;

    struct Stat {
        std::vector<double> latency;
        uint64_t throughput = 0;
        size_t fitness_of_dataset = 0;
        double mse = 0;
        long long memory_consumption = 0;
        uint64_t success_insert = 0;
        uint64_t success_read = 0;
        uint64_t success_update = 0;
        uint64_t success_erase = 0;
        uint64_t scan_not_enough = 0;

        void clear() {
            latency.clear();
            throughput = 0;
            fitness_of_dataset = 0;
            mse = 0;
            memory_consumption = 0;
            success_insert = 0;
            success_read = 0;
            success_update = 0;
            success_erase = 0;
            scan_not_enough = 0;
        }
    } stat;

    struct alignas(CACHELINE_SIZE)
    ThreadParam {
        std::vector<double> latency;
        uint64_t success_insert = 0;
        uint64_t success_read = 0;
        uint64_t success_update = 0;
        uint64_t success_erase = 0;
        uint64_t scan_not_enough = 0;
    };
    typedef ThreadParam param_t;
public:
    Benchmark() {
    }

    KEY_TYPE *load_keys() {
        // Read keys from file
        COUT_THIS("Reading data from file.");

        if (table_size > 0) keys = new KEY_TYPE[table_size];


        if (keys_file_type == "binary") {
            table_size = load_binary_data(keys, table_size, keys_file_path);
            if (table_size <= 0) {
                COUT_THIS("Could not open key file, please check the path of key file.");
                exit(0);
            }
        } else if (keys_file_type == "text") {
            table_size = load_text_data(keys, table_size, keys_file_path);
            if (table_size <= 0) {
                COUT_THIS("Could not open key file, please check the path of key file.");
                exit(0);
            }
        } else {
            COUT_THIS("Could not open key file, please check the path of key file.");
            exit(0);
        }

        if (!data_shift) {
            tbb::parallel_sort(keys, keys + table_size);
            auto last = std::unique(keys, keys + table_size);
            table_size = last - keys;
            std::shuffle(keys, keys + table_size, gen);
        }

        init_table_size = init_table_ratio * table_size;
        std::cout << "Table size is " << table_size << ", Init table size is " << init_table_size << std::endl;

        for (auto j = 0; j < 10; j++) {
            std::cout << keys[j] << " ";
        }
        std::cout << std::endl;

        // prepare data
        COUT_THIS("prepare init keys.");
        init_keys.resize(init_table_size);
#pragma omp parallel for num_threads(thread_num)
        for (size_t i = 0; i < init_table_size; ++i) {
            init_keys[i] = (keys[i]);
        }
        tbb::parallel_sort(init_keys.begin(), init_keys.end());

        init_key_values = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[init_keys.size()];
#pragma omp parallel for num_threads(thread_num)
        for (int i = 0; i < init_keys.size(); i++) {
            init_key_values[i].first = init_keys[i];
            init_key_values[i].second = init_keys[i];
        }
        COUT_VAR(table_size);
        COUT_VAR(init_keys.size());

        return keys;
    }

    /*
   * keys_file_path:      the path where keys file at
   * keys_file_type:      binary or text
   * read_ratio:          the ratio of read operation
   * insert_ratio         the ratio of insert operation
   * delete_ratio         the ratio of delete operation
   * update_ratio         the ratio of update operation
   * scan_ratio           the ratio of scan operation
   * scan_num             the number of keys that every scan operation need to scan
   * operations_num      the number of operations(read, insert, delete, update, scan)
   * table_size           the total number of keys in key file
   * init_table_size      the number of keys that will be used in bulk loading
   * thread_num           the number of worker thread
   * index_type           the type of index(xindex, hot, alex...). Detail could be refered to src/competitor
   * sample_distribution  the distribution of
   * latency_sample_ratio the ratio of latency sampling
   * error_bound          the error bound of PGM metric
   * output_path          the path to store result
  */
    inline void parse_args(int argc, char **argv) {
        auto flags = parse_flags(argc, argv);
        keys_file_path = get_required(flags, "keys_file"); // required
        keys_file_type = get_with_default(flags, "keys_file_type", "binary");
        read_ratio = stod(get_required(flags, "read")); // required
        insert_ratio = stod(get_with_default(flags, "insert", "0")); // required
        delete_ratio = stod(get_with_default(flags, "delete", "0"));
        update_ratio = stod(get_with_default(flags, "update", "0"));
        scan_ratio = stod(get_with_default(flags, "scan", "0"));
        scan_num = stoi(get_with_default(flags, "scan_num", "100"));
        operations_num = stoi(get_with_default(flags, "operations_num", "1000000000")); // required
        table_size = stoi(get_with_default(flags, "table_size", "-1"));
        init_table_ratio = stod(get_with_default(flags, "init_table_ratio", "0.5"));
        init_table_size = 0;
        thread_num = stoi(get_with_default(flags, "thread_num", "1")); // required
        sample_distribution = get_with_default(flags, "sample_distribution", "uniform");
        latency_sample = get_boolean_flag(flags, "latency_sample");
        latency_sample_ratio = stod(get_with_default(flags, "latency_sample_ratio", "0.001"));
        error_bound = stoi(get_with_default(flags, "error_bound", "64"));
        output_path = get_with_default(flags, "output_path", "./result");
        pgm_metric = stod(get_with_default(flags, "pgm_metric", "500000"));
        random_seed = stoul(get_with_default(flags, "seed", "1866"));
        gen.seed(random_seed);
        variance = stod(get_with_default(flags, "variance", "20"));
        memory_record = get_boolean_flag(flags, "memory");
        dataset_statistic = get_boolean_flag(flags, "dataset_statistic");
        workload_shift_test = get_boolean_flag(flags, "workload_shift_test");
        data_shift = get_boolean_flag(flags, "data_shift");
        hyper_thread = get_boolean_flag(flags, "hyper_thread");
        if (workload_shift_test) {
            write_ratio_begin = stod(get_with_default(flags, "write_ratio_begin", "0"));
            write_ratio_end = stod(get_with_default(flags, "write_ratio_end", "0"));
            read_ratio = 1 - write_ratio_end;
            insert_ratio = write_ratio_end;
        }

        COUT_THIS("[micro] Read:Insert:Update:Scan= " << read_ratio << ":" << insert_ratio << ":" << update_ratio << ":"
                                                      << scan_ratio);
        double ratio_sum = read_ratio + insert_ratio + delete_ratio + update_ratio + scan_ratio;
        INVARIANT(ratio_sum > 0.9999 && ratio_sum < 1.0001);  // avoid precision lost
        INVARIANT(sample_distribution == "zipf" || sample_distribution == "uniform");
    }


    void generate_operations(KEY_TYPE *keys) {
        // prepare operations
        operations.reserve(operations_num);
        COUT_THIS("sample keys.");
        KEY_TYPE *sample_ptr = nullptr;
        if (sample_distribution == "uniform") {
            sample_ptr = get_search_keys(&init_keys[0], init_table_size, operations_num, &random_seed);
        } else if (sample_distribution == "zipf") {
            sample_ptr = get_search_keys_zipf(&init_keys[0], init_table_size, operations_num, &random_seed);
        }

        // generate operations(read, insert, update, scan)
        COUT_THIS("generate operations.");
        std::uniform_real_distribution<> ratio_dis(0, 1);
        size_t sample_counter = 0, insert_counter = init_table_size;

        size_t rest_key_num = table_size - init_table_size;
        auto unique_init_table_size = init_table_size;
        auto tail = unique_data(&keys[0], unique_init_table_size, &keys[init_table_size], rest_key_num);
        table_size = tail - keys;

        if (workload_shift_test) {
            read_ratio = 1 - write_ratio_end;
            insert_ratio = write_ratio_end;
            insert_counter = init_table_size + rest_key_num / 2;
        }

        for (size_t i = 0; i < operations_num; ++i) {
            auto prob = ratio_dis(gen);
            if (prob < read_ratio) {
                operations.push_back(std::pair<Operation, KEY_TYPE>(READ, sample_ptr[sample_counter++]));
            } else if (prob < read_ratio + insert_ratio) {
                if (insert_counter >= table_size) {
                    operations_num = i;
                    break;
                }
                operations.push_back(std::pair<Operation, KEY_TYPE>(INSERT, keys[insert_counter++]));
            } else if (prob < read_ratio + insert_ratio + update_ratio) {
                operations.push_back(std::pair<Operation, KEY_TYPE>(UPDATE, sample_ptr[sample_counter++]));
            } else {
                operations.push_back(std::pair<Operation, KEY_TYPE>(SCAN, sample_ptr[sample_counter++]));
            }
        }

        COUT_VAR(operations.size());

        delete[] sample_ptr;
    }

    void run() {
        std::thread *thread_array = new std::thread[thread_num];
        param_t params[thread_num];

        printf("Begin running\n");
        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time = std::chrono::high_resolution_clock::now();
//        System::profile("perf.data", [&]() {
#pragma omp parallel num_threads(thread_num)
        {
            // thread specifier
            auto thread_id = omp_get_thread_num();
            // Latency Sample Variable
            int latency_sample_interval = operations_num / (operations_num * latency_sample_ratio);
            auto latency_sample_start_time = std::chrono::high_resolution_clock::now();
            auto latency_sample_end_time = std::chrono::high_resolution_clock::now();
            param_t &thread_param = params[thread_id];
            std::vector<double> &latency = thread_param.latency;
            // Operation Parameter
            PAYLOAD_TYPE val;
            std::pair <KEY_TYPE, PAYLOAD_TYPE> *scan_result = new std::pair<KEY_TYPE, PAYLOAD_TYPE>[scan_num];
            // waiting all thread ready
#pragma omp barrier
#pragma omp master
            start_time = std::chrono::high_resolution_clock::now();
// running benchmark
#pragma omp for schedule(dynamic, 10000)
            for (auto i = 0; i < operations_num; i++) {
                auto op = operations[i].first;
                auto key = operations[i].second;

                if (latency_sample && i % latency_sample_interval == 0)
                    latency_sample_start_time = std::chrono::high_resolution_clock::now();

                if (op == READ) {  // get
                    val = index.at(key, false);
                    if(val != key) {
                        printf("read wrong payload, Key %lu, val %llu\n",key, val);
                        exit(1);
                    }
                    thread_param.success_read += 1;
                } else if (op == INSERT) {  // insert
                    index.insert(key, key);
                    thread_param.success_insert += 1;
                } else if (op == SCAN) { // scan
                    // TODO
                }

                if (latency_sample && i % latency_sample_interval == 0) {
                    latency_sample_end_time = std::chrono::high_resolution_clock::now();
                    latency.push_back(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    (latency_sample_end_time - latency_sample_start_time)).count());
                }
            } // omp for loop
#pragma omp master
            end_time = std::chrono::high_resolution_clock::now();
        } // all thread join here

//        });

        std::chrono::duration<double> diff = end_time - start_time;
        printf("Finish running\n");

        // gather thread local variable
        for (auto &p: params) {
            if (latency_sample)stat.latency.insert(stat.latency.end(), p.latency.begin(), p.latency.end());
            stat.success_read += p.success_read;
            stat.success_insert += p.success_insert;
            stat.success_update += p.success_update;
            stat.success_erase += p.success_erase;
            stat.scan_not_enough += p.scan_not_enough;
        }
        // calculate throughput
        stat.throughput = static_cast<uint64_t>(operations_num / diff.count());

        print_stat();

        delete[] thread_array;
    }

    void print_stat(bool header = false, bool clear_flag = true) {
        double avg_latency = 0;
        // average latency
        for (auto t : stat.latency) {
            avg_latency += t;
        }
        avg_latency /= stat.latency.size();

        // latency variance
        double latency_variance = 0;
        if (latency_sample) {
            for (auto t : stat.latency) {
                latency_variance += (t - avg_latency) * (t - avg_latency);
            }
            latency_variance /= stat.latency.size();
            std::sort(stat.latency.begin(), stat.latency.end());
        }

        printf(
                "Throughput\tmemory\tPGM\tMSE\tsuccess_read\tsuccess_insert\tsuccess_update\tscan_not_enough\n");
        printf("%llu\t", stat.throughput);
        printf("%u\t", stat.memory_consumption);
        printf("%u\t", stat.fitness_of_dataset);
        printf("%f\t", stat.mse);
        printf("%llu\t", stat.success_read);
        printf("%llu\t", stat.success_insert);
        printf("%llu\t", stat.success_update);
        printf("%llu\n", stat.scan_not_enough);

        // time id
        std::time_t t = std::time(nullptr);
        char time_str[100];

        if (!file_exists(output_path)) {
            std::ofstream ofile;
            ofile.open(output_path, std::ios::app);
            ofile << "id" << ",";
            ofile << "read_ratio" << "," << "insert_ratio" << "," << "update_ratio" << "," << "scan_ratio" << ",";
            ofile << "key_path" << ",";
            ofile << "index_type" << ",";
            ofile << "throughput" << ",";
            ofile << "init_table_size" << ",";
            ofile << "memory_consumption" << ",";
            ofile << "thread_num" << ",";
            ofile << "min" << ",";
            ofile << "50 percentile" << ",";
            ofile << "90 percentile" << ",";
            ofile << "99 percentile" << ",";
            ofile << "99.9 percentile" << ",";
            ofile << "99.99 percentile" << ",";
            ofile << "max" << ",";
            ofile << "avg" << ",";
            ofile << "pgm" << ",";
            ofile << "variance" << ",";
            ofile << "seed" << ",";
            ofile << "scan_num" << ",";
            ofile << "write_ratio_begin" << ",";
            ofile << "write_ratio_end" << ",";
            ofile << "latency_variance" << ",";
            ofile << "latency_sample" << ",";
            ofile << "workload_shift" << ",";
            ofile << "data_shift" << ",";
            ofile << "hyper_thread" << ",";
            ofile << "mse" << ",";
            ofile << "error_bound" ",";
            ofile << "table_size" << std::endl;
        }

        std::ofstream ofile;
        ofile.open(output_path, std::ios::app);
        if (std::strftime(time_str, sizeof(time_str), "%Y%m%d%H%M%S", std::localtime(&t))) {
            ofile << time_str << ',';
        }
        ofile << read_ratio << "," << insert_ratio << "," << update_ratio << "," << scan_ratio << ",";

        ofile << keys_file_path << ",";
        ofile << index_type << ",";
        ofile << stat.throughput << ",";
        ofile << init_table_size << ",";
        ofile << stat.memory_consumption << ",";
        ofile << thread_num << ",";
        if (latency_sample) {
            ofile << stat.latency[0] << ",";
            ofile << stat.latency[0.5 * stat.latency.size()] << ",";
            ofile << stat.latency[0.9 * stat.latency.size()] << ",";
            ofile << stat.latency[0.99 * stat.latency.size()] << ",";
            ofile << stat.latency[0.999 * stat.latency.size()] << ",";
            ofile << stat.latency[0.9999 * stat.latency.size()] << ",";
            ofile << stat.latency[stat.latency.size() - 1] << ",";
            ofile << avg_latency << ",";
        } else {
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
            ofile << 0 << ",";
        }
        ofile << stat.fitness_of_dataset << ",";
        ofile << variance << ",";
        ofile << random_seed << ",";
        ofile << scan_num << ",";
        ofile << write_ratio_begin << ",";
        ofile << write_ratio_end << ",";
        ofile << latency_variance << ",";
        ofile << latency_sample << ",";
        ofile << workload_shift_test << ",";
        ofile << data_shift << ",";
        ofile << hyper_thread << ",";
        ofile << stat.mse << ",";
        ofile << error_bound << ",";
        ofile << table_size << std::endl;
        ofile.close();

        if (clear_flag) stat.clear();

//        malloc_stats_print(NULL, NULL, NULL);
    }

    void run_benchmark() {
        load_keys();
        generate_operations(keys);

        COUT_THIS("bulk loading");
        index.bulk_load(init_key_values, static_cast<int>(init_keys.size()));
        run();
    }
};
