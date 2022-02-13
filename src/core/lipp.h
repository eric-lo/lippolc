#ifndef __LIPPOL_H__
#define __LIPPOL_H__

#include "concurrency.h"
#include "lipp_base.h"
#include "omp.h"
#include "tbb/combinable.h"
#include "tbb/enumerable_thread_specific.h"
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <list>
#include <math.h>
#include <sstream>
#include <stack>
#include <stdint.h>
#include <thread>
#include <vector>

typedef uint8_t bitmap_t;
#define BITMAP_WIDTH (sizeof(bitmap_t) * 8)
#define BITMAP_SIZE(num_items) (((num_items) + BITMAP_WIDTH - 1) / BITMAP_WIDTH)
#define BITMAP_GET(bitmap, pos)                                                \
  (((bitmap)[(pos) / BITMAP_WIDTH] >> ((pos) % BITMAP_WIDTH)) & 1)
#define BITMAP_SET(bitmap, pos)                                                \
  ((bitmap)[(pos) / BITMAP_WIDTH] |= 1 << ((pos) % BITMAP_WIDTH))
#define BITMAP_CLEAR(bitmap, pos)                                              \
  ((bitmap)[(pos) / BITMAP_WIDTH] &= ~bitmap_t(1 << ((pos) % BITMAP_WIDTH)))
#define BITMAP_NEXT_1(bitmap_item) __builtin_ctz((bitmap_item))

// runtime assert
#define RT_ASSERT(expr)                                                        \
  {                                                                            \
    if (!(expr)) {                                                             \
      fprintf(stderr, "Thread %d: RT_ASSERT Error at %s:%d, `%s` not hold!\n", \
              omp_get_thread_num(), __FILE__, __LINE__, #expr);                \
      exit(0);                                                                 \
    }                                                                          \
  }

typedef void (*dealloc_func)(void *ptr);

// runtime debug
#define DEBUG 0

#if DEBUG
#define RESET "\033[0m"
#define RED "\033[31m"     /* Red */
#define GREEN "\033[32m"   /* Green */
#define YELLOW "\033[33m"  /* Yellow */
#define BLUE "\033[34m"    /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m"    /* Cyan */
#define WHITE "\033[37m"   /* White */

#define RT_DEBUG(msg, ...)                                                     \
  if (omp_get_thread_num() == 0) {                                             \
    printf(GREEN "T%d: " msg RESET "\n", omp_get_thread_num(), __VA_ARGS__);   \
  } else if (omp_get_thread_num() == 1) {                                      \
    printf(YELLOW "\t\t\tT%d: " msg RESET "\n", omp_get_thread_num(),          \
           __VA_ARGS__);                                                       \
  } else {                                                                     \
    printf(BLUE "\t\t\t\t\t\tT%d: " msg RESET "\n", omp_get_thread_num(),      \
           __VA_ARGS__);                                                       \
  }
#else
#define RT_DEBUG(msg, ...)
#endif

#define COLLECT_TIME 0

#if COLLECT_TIME
#include <chrono>
#endif

namespace lippolc {

template <class T, class P, bool USE_FMCD = true> class LIPP {
  static_assert(std::is_arithmetic<T>::value, "LIPP key type must be numeric.");

  inline int compute_gap_count(int size) {
    if (size >= 1000000)
      return 1;
    if (size >= 100000)
      return 2;
    return 5;
  }

  struct Node;
  inline int PREDICT_POS(Node *node, T key) const {
    double v = node->model.predict_double(key);
    if (v > std::numeric_limits<int>::max() / 2) {
      return node->num_items - 1;
    }
    if (v < 0) {
      return 0;
    }
    return std::min(node->num_items - 1, static_cast<int>(v));
  }

  static void remove_last_bit(bitmap_t &bitmap_item) {
    bitmap_item -= 1 << BITMAP_NEXT_1(bitmap_item);
  }

  const double BUILD_LR_REMAIN;
  const bool QUIET;

  struct {
    long long fmcd_success_times = 0;
    long long fmcd_broken_times = 0;
#if COLLECT_TIME
    double time_scan_and_destory_tree = 0;
    double time_build_tree_bulk = 0;
#endif
  } stats;

public:
  // Epoch based Memory Reclaim
  class ThreadSpecificEpochBasedReclamationInformation {

    // std::array<std::vector<void *>, 3> mFreeLists;
    std::array<std::vector<std::pair<void *, dealloc_func>>, 3> mFreeLists;
    std::atomic<uint32_t> mLocalEpoch;
    uint32_t mPreviouslyAccessedEpoch;
    bool mThreadWantsToAdvance;

  public:
    ThreadSpecificEpochBasedReclamationInformation()
        : mFreeLists(), mLocalEpoch(3), mPreviouslyAccessedEpoch(3),
          mThreadWantsToAdvance(false) {}

    ThreadSpecificEpochBasedReclamationInformation(
        ThreadSpecificEpochBasedReclamationInformation const &other) = delete;

    ThreadSpecificEpochBasedReclamationInformation(
        ThreadSpecificEpochBasedReclamationInformation &&other) = delete;

    ~ThreadSpecificEpochBasedReclamationInformation() {
      for (uint32_t i = 0; i < 3; ++i) {
        freeForEpoch(i);
      }
    }

    void scheduleForDeletion(std::pair<void *, dealloc_func> func_pair) {
      assert(mLocalEpoch != 3);
      std::vector<std::pair<void *, dealloc_func>> &currentFreeList =
          mFreeLists[mLocalEpoch];
      currentFreeList.emplace_back(func_pair);
      mThreadWantsToAdvance = (currentFreeList.size() % 64u) == 0;
    }

    uint32_t getLocalEpoch() const {
      return mLocalEpoch.load(std::memory_order_acquire);
    }

    void enter(uint32_t newEpoch) {
      assert(mLocalEpoch == 3);
      if (mPreviouslyAccessedEpoch != newEpoch) {
        freeForEpoch(newEpoch);
        mThreadWantsToAdvance = false;
        mPreviouslyAccessedEpoch = newEpoch;
      }
      mLocalEpoch.store(newEpoch, std::memory_order_release);
    }

    void leave() { mLocalEpoch.store(3, std::memory_order_release); }

    bool doesThreadWantToAdvanceEpoch() { return (mThreadWantsToAdvance); }

  private:
    void freeForEpoch(uint32_t epoch) {
      std::vector<std::pair<void *, dealloc_func>> &previousFreeList =
          mFreeLists[epoch];
      // for (void *pointer : previousFreeList) {
      for (std::pair<void *, dealloc_func> func_pair : previousFreeList) {
        func_pair.second(func_pair.first);
        /*
        auto node = reinterpret_cast<Node *>(pointer);
        my_tree->delete_items(node->items, node->num_items);
        const int bitmap_size = BITMAP_SIZE(node->num_items);
        my_tree->delete_bitmap(node->none_bitmap, bitmap_size);
        my_tree->delete_bitmap(node->child_bitmap, bitmap_size);
        my_tree->delete_nodes(node, 1);
        */
      }
      previousFreeList.resize(0u);
    }
  };

  class EpochBasedMemoryReclamationStrategy {
  public:
    uint32_t NEXT_EPOCH[3] = {1, 2, 0};
    uint32_t PREVIOUS_EPOCH[3] = {2, 0, 1};

    std::atomic<uint32_t> mCurrentEpoch;
    tbb::enumerable_thread_specific<
        ThreadSpecificEpochBasedReclamationInformation,
        tbb::cache_aligned_allocator<
            ThreadSpecificEpochBasedReclamationInformation>,
        tbb::ets_key_per_instance>
        mThreadSpecificInformations;

  private:
    EpochBasedMemoryReclamationStrategy()
        : mCurrentEpoch(0), mThreadSpecificInformations() {}

  public:
    static EpochBasedMemoryReclamationStrategy *getInstance() {
      static EpochBasedMemoryReclamationStrategy instance;
      return &instance;
    }

    void enterCriticalSection() {
      ThreadSpecificEpochBasedReclamationInformation &currentMemoryInformation =
          mThreadSpecificInformations.local();
      uint32_t currentEpoch = mCurrentEpoch.load(std::memory_order_acquire);
      currentMemoryInformation.enter(currentEpoch);
      if (currentMemoryInformation.doesThreadWantToAdvanceEpoch() &&
          canAdvance(currentEpoch)) {
        mCurrentEpoch.compare_exchange_strong(currentEpoch,
                                              NEXT_EPOCH[currentEpoch]);
      }
    }

    bool canAdvance(uint32_t currentEpoch) {
      uint32_t previousEpoch = PREVIOUS_EPOCH[currentEpoch];
      return !std::any_of(
          mThreadSpecificInformations.begin(),
          mThreadSpecificInformations.end(),
          [previousEpoch](ThreadSpecificEpochBasedReclamationInformation const
                              &threadInformation) {
            return (threadInformation.getLocalEpoch() == previousEpoch);
          });
    }

    void leaveCriticialSection() {
      ThreadSpecificEpochBasedReclamationInformation &currentMemoryInformation =
          mThreadSpecificInformations.local();
      currentMemoryInformation.leave();
    }

    void scheduleForDeletion(std::pair<void *, dealloc_func> func_pair) {
      mThreadSpecificInformations.local().scheduleForDeletion(func_pair);
    }
  };

  class EpochGuard {
    EpochBasedMemoryReclamationStrategy *instance;

  public:
    EpochGuard() {
      instance = EpochBasedMemoryReclamationStrategy::getInstance();
      instance->enterCriticalSection();
    }

    ~EpochGuard() { instance->leaveCriticialSection(); }
  };

  EpochBasedMemoryReclamationStrategy *ebr;

  typedef std::pair<T, P> V;

  LIPP(double BUILD_LR_REMAIN = 0, bool QUIET = true)
      : BUILD_LR_REMAIN(BUILD_LR_REMAIN), QUIET(QUIET) {
    {
      std::vector<Node *> nodes;
      for (int _ = 0; _ < 1e7; _++) {
        Node *node = build_tree_two(T(0), P(), T(1), P());
        nodes.push_back(node);
      }
      for (auto node : nodes) {
        destroy_tree(node);
      }
      if (!QUIET) {
        printf("initial memory pool size = %lu\n",
               pending_two[omp_get_thread_num()].size());
      }
    }
    if (USE_FMCD && !QUIET) {
      printf("enable FMCD\n");
    }

    root = build_tree_none();
    ebr = EpochBasedMemoryReclamationStrategy::getInstance();
  }
  ~LIPP() {
    destroy_tree(root);
    root = NULL;
    destory_pending();
  }

  void contention() {
    std::stack<Node*> s;
    s.push(root);
    int max_level = 0;
    root->level = 0;

    while (!s.empty()) {
      Node* node = s.top(); s.pop();
      for (int i = 0; i < node->num_items; i ++) {
        if (BITMAP_GET(node->child_bitmap, i) == 1) {
          int this_level = node->level + 1;
          if(this_level > max_level)
            max_level = this_level;
          node->items[i].comp.child->level = this_level;
          s.push(node->items[i].comp.child);
        }
      }
    }
    int result[max_level+1];
    while (!s.empty()) {
      Node* node = s.top(); s.pop();
      for (int i = 0; i < node->num_items; i ++) {
        result[node->level] += node->restart_cnt;
        if (BITMAP_GET(node->child_bitmap, i) == 1) {
          s.push(node->items[i].comp.child);
        }
      }
    }
    printf("Contention\n");
    for(int i = 0; i <= max_level; i++){
      printf("level %d: %d\n", i, result[i]);
    }
  }

  void insert(const V &v) { insert(v.first, v.second); }
  void insert(const T &key, const P &value) {
    EpochGuard guard;
    // root = insert_tree(root, key, value);
    bool state = insert_tree(key, value);
    RT_DEBUG("Insert_tree(%d): success/fail? %d", key, state);
  }

  P at(const T &key, bool skip_existence_check = true) {
    EpochGuard guard;
    int restartCount = 0;
  restart:
    if (restartCount++)
      yield(restartCount);
    bool needRestart = false;

    // for lock coupling
    Node *parent = nullptr;
    uint64_t versionParent;

    for (Node *node = root;;) {
      uint64_t versionNode = node->readLockOrRestart(
          needRestart); // set versionNode be child's version; read "lock" the
                        // child
      if (needRestart) {
        node->restart_cnt++;
        goto restart;
      } // if child is locked by another thread, restart

      int pos = PREDICT_POS(node, key);
      if (BITMAP_GET(node->child_bitmap, pos) == 1) { // 1 means child
        // now ready for the tree traversal
        parent = node; // initialize new parent to be the (current) node---(2)
        versionParent = versionNode; // initialize versionparent to be the fetched
                                    // version number of the (new) parent
        node = node->items[pos].comp.child;           // now: node is the child

        parent->readUnlockOrRestart(versionParent, needRestart);
        if (needRestart) {
          parent->restart_cnt++;
          goto restart;
        } // if parent has changed: restart

      } else {          // the entry is a data
        if (skip_existence_check) {
          node->readUnlockOrRestart(
              versionNode, needRestart); // as this is the leaf node, unlock it
          if (needRestart) {
            node->restart_cnt++;
            goto restart;
          }

          return node->items[pos].comp.data.value;
        } else {
          if (BITMAP_GET(node->none_bitmap, pos) == 1) {
            RT_ASSERT(false);
          } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
            node->readUnlockOrRestart(
                versionNode,
                needRestart); // as this is the leaf node, unlock it
            if (needRestart) {
              node->restart_cnt++;
              goto restart;
            }

            RT_ASSERT(node->items[pos].comp.data.key == key);
            return node->items[pos].comp.data.value;
          }
        }
      }
    }
  }

  void yield(int count) {
    if (count > 3)
      sched_yield();
    else
      _mm_pause();
  }

  bool exists(const T &key) const {
    EpochGuard guard;
    Node *node = root;
    while (true) {
      int pos = PREDICT_POS(node, key);
      if (BITMAP_GET(node->none_bitmap, pos) == 1) {
        return false;
      } else if (BITMAP_GET(node->child_bitmap, pos) == 0) {
        return node->items[pos].comp.data.key == key;
      } else {
        node = node->items[pos].comp.child;
      }
    }
  }
  void bulk_load(const V *vs, int num_keys) {
    if (num_keys == 0) {
      destroy_tree(root);
      root = build_tree_none();
      return;
    }
    if (num_keys == 1) {
      destroy_tree(root);
      root = build_tree_none();
      insert(vs[0]);
      return;
    }
    if (num_keys == 2) {
      destroy_tree(root);
      root =
          build_tree_two(vs[0].first, vs[0].second, vs[1].first, vs[1].second);
      return;
    }

    RT_ASSERT(num_keys > 2);
    for (int i = 1; i < num_keys; i++) {
      RT_ASSERT(vs[i].first > vs[i - 1].first);
    }

    T *keys = new T[num_keys];
    P *values = new P[num_keys];
    for (int i = 0; i < num_keys; i++) {
      keys[i] = vs[i].first;
      values[i] = vs[i].second;
    }
    destroy_tree(root);
    root = build_tree_bulk(keys, values, num_keys);
    delete[] keys;
    delete[] values;
  }

  void show() const {
    printf("============= SHOW LIPP ================\n");

    std::stack<Node *> s;
    s.push(root);
    while (!s.empty()) {
      Node *node = s.top();
      s.pop();

      printf("Node(%p, a = %lf, b = %lf, num_items = %d)", node, node->model.a,
             node->model.b, node->num_items);
      printf("[");
      int first = 1;
      for (int i = 0; i < node->num_items; i++) {
        if (!first) {
          printf(", ");
        }
        first = 0;
        if (BITMAP_GET(node->none_bitmap, i) == 1) {
          printf("None");
        } else if (BITMAP_GET(node->child_bitmap, i) == 0) {
          std::stringstream s;
          s << node->items[i].comp.data.key;
          printf("Key(%s)", s.str().c_str());
        } else {
          printf("Child(%p)", node->items[i].comp.child);
          s.push(node->items[i].comp.child);
        }
      }
      printf("]\n");
    }
  }
  void print_depth() const {
    std::stack<Node *> s;
    std::stack<int> d;
    s.push(root);
    d.push(1);

    int max_depth = 1;
    int sum_depth = 0, sum_nodes = 0;
    while (!s.empty()) {
      Node *node = s.top();
      s.pop();
      int depth = d.top();
      d.pop();
      for (int i = 0; i < node->num_items; i++) {
        if (BITMAP_GET(node->child_bitmap, i) == 1) {
          s.push(node->items[i].comp.child);
          d.push(depth + 1);
        } else if (BITMAP_GET(node->none_bitmap, i) != 1) {
          max_depth = std::max(max_depth, depth);
          sum_depth += depth;
          sum_nodes++;
        }
      }
    }

    printf("max_depth = %d, avg_depth = %.2lf\n", max_depth,
           double(sum_depth) / double(sum_nodes));
  }
  void verify() const {
    std::stack<Node *> s;
    s.push(root);

    while (!s.empty()) {
      Node *node = s.top();
      s.pop();
      int sum_size = 0;
      for (int i = 0; i < node->num_items; i++) {
        if (BITMAP_GET(node->child_bitmap, i) == 1) {
          s.push(node->items[i].comp.child);
          sum_size += node->items[i].comp.child->size;
        } else if (BITMAP_GET(node->none_bitmap, i) != 1) {
          sum_size++;
        }
      }
      RT_ASSERT(sum_size == node->size);
    }
  }
  void print_stats() const {
    printf("======== Stats ===========\n");
    if (USE_FMCD) {
      printf("\t fmcd_success_times = %lld\n", stats.fmcd_success_times);
      printf("\t fmcd_broken_times = %lld\n", stats.fmcd_broken_times);
    }
#if COLLECT_TIME
    printf("\t time_scan_and_destory_tree = %lf\n",
           stats.time_scan_and_destory_tree);
    printf("\t time_build_tree_bulk = %lf\n", stats.time_build_tree_bulk);
#endif
  }
  size_t index_size() const {
    std::stack<Node *> s;
    s.push(root);

    size_t size = 0;
    while (!s.empty()) {
      Node *node = s.top();
      s.pop();
      bool has_child = false;
      for (int i = 0; i < node->num_items; i++) {
        if (BITMAP_GET(node->child_bitmap, i) == 1) {
          size += sizeof(Item);
          has_child = true;
          s.push(node->items[i].comp.child);
        }
      }
      if (has_child)
        size += sizeof(*node);
    }
    return size;
  }

private:
  struct Node;
  struct Item {
    union {
      struct {
        T key;
        P value;
      } data;
      Node *child;
    } comp;
  };
  struct Node : OptLock {
    int is_two;            // is special node for only two keys
    int build_size;        // tree size (include sub nodes) when node created
    std::atomic<int> size; // current subtree size
    // int size;
    int fixed; // fixed node will not trigger rebuild
    std::atomic<int> num_inserts, num_insert_to_data;
    std::atomic<int> num_items; // number of slots
    // int num_items;
    LinearModel<T> model;
    Item *items;
    bitmap_t *none_bitmap; // 1 means empty entry, 0 means Data or Child
    bitmap_t
        *child_bitmap; // 1 means Child. will always be 0 when none_bitmap is 1
    std::atomic<unsigned int> restart_cnt;
    int level;
  };

  Node *root;
  int adjustsuccess = 0;
  std::stack<Node *> pending_two[1024];

  std::allocator<Node> node_allocator;
  Node *new_nodes(int n) {
    Node *p = node_allocator.allocate(n);
    for (int i = 0; i < n; ++i) {
      p[i].typeVersionLockObsolete.store(0b100);
      // lock->typeVersionLockObsolete.store(0b100);
      p[i].restart_cnt = 0;
    }

    RT_ASSERT(p != NULL && p != (Node *)(-1));
    return p;
  }

  void delete_nodes(Node *p, int n) { node_allocator.deallocate(p, n); }

  void safe_delete_nodes(Node *p, int n) {
    auto callback = [](void *pointer) {
      std::pair<LIPP *, Node *> *ptr =
          reinterpret_cast<std::pair<LIPP *, Node *> *>(pointer);
      auto my_tree = ptr->first;
      auto node = ptr->second;
      if (node->is_two) {
        node->writeUnlock();
        my_tree->pending_two[omp_get_thread_num()].push(node);
      } else {
        my_tree->delete_items(node->items, node->num_items);
        const int bitmap_size = BITMAP_SIZE(node->num_items);
        my_tree->delete_bitmap(node->none_bitmap, bitmap_size);
        my_tree->delete_bitmap(node->child_bitmap, bitmap_size);
        my_tree->delete_nodes(node, 1);
      }
      delete ptr;
      return;
    };

    for (int i = 0; i < n; ++i) {
      auto ptr = new std::pair<LIPP *, Node *>(this, p);
      ebr->scheduleForDeletion(
          std::make_pair(reinterpret_cast<void *>(ptr), callback));
      p = p + 1;
    }
    // node_allocator.deallocate(p, n);
  }

  std::allocator<Item> item_allocator;
  Item *new_items(int n) {
    Item *p = item_allocator.allocate(n);
    RT_ASSERT(p != NULL && p != (Item *)(-1));
    return p;
  }
  void delete_items(Item *p, int n) { item_allocator.deallocate(p, n); }

  std::allocator<bitmap_t> bitmap_allocator;
  bitmap_t *new_bitmap(int n) {
    bitmap_t *p = bitmap_allocator.allocate(n);
    RT_ASSERT(p != NULL && p != (bitmap_t *)(-1));
    return p;
  }
  void delete_bitmap(bitmap_t *p, int n) { bitmap_allocator.deallocate(p, n); }

  /// build an empty tree
  Node *build_tree_none() {
    Node *node = new_nodes(1);
    node->is_two = 0;
    node->build_size = 0;
    node->size = 0;
    node->fixed = 0;
    node->num_inserts = node->num_insert_to_data = 0;
    node->num_items = 1;
    node->model.a = node->model.b = 0;
    node->items = new_items(1);
    node->none_bitmap = new_bitmap(1);
    node->none_bitmap[0] = 0;
    BITMAP_SET(node->none_bitmap, 0);
    node->child_bitmap = new_bitmap(1);
    node->child_bitmap[0] = 0;
    return node;
  }
  /// build a tree with two keys
  Node *build_tree_two(T key1, P value1, T key2, P value2) {
    if (key1 > key2) {
      std::swap(key1, key2);
      std::swap(value1, value2);
    }
    RT_ASSERT(key1 < key2);
    static_assert(BITMAP_WIDTH == 8);

    Node *node = NULL;
    if (pending_two[omp_get_thread_num()].empty()) {
      node = new_nodes(1);
      node->is_two = 1;
      node->build_size = 2;
      node->size = 2;
      node->fixed = 0;
      node->num_inserts = node->num_insert_to_data = 0;

      node->num_items = 8;
      node->items = new_items(node->num_items);
      node->none_bitmap = new_bitmap(1);
      node->child_bitmap = new_bitmap(1);
      node->none_bitmap[0] = 0xff;
      node->child_bitmap[0] = 0;
    } else {
      node = pending_two[omp_get_thread_num()].top();
      pending_two[omp_get_thread_num()].pop();
    }

    const long double mid1_key = key1;
    const long double mid2_key = key2;

    const double mid1_target = node->num_items / 3;
    const double mid2_target = node->num_items * 2 / 3;

    node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
    node->model.b = mid1_target - node->model.a * mid1_key;
    RT_ASSERT(isfinite(node->model.a));
    RT_ASSERT(isfinite(node->model.b));

    { // insert key1&value1
      int pos = PREDICT_POS(node, key1);
      RT_ASSERT(BITMAP_GET(node->none_bitmap, pos) == 1);
      BITMAP_CLEAR(node->none_bitmap, pos);
      node->items[pos].comp.data.key = key1;
      node->items[pos].comp.data.value = value1;
    }
    { // insert key2&value2
      int pos = PREDICT_POS(node, key2);
      RT_ASSERT(BITMAP_GET(node->none_bitmap, pos) == 1);
      BITMAP_CLEAR(node->none_bitmap, pos);
      node->items[pos].comp.data.key = key2;
      node->items[pos].comp.data.value = value2;
    }

    return node;
  }
  /// bulk build, _keys must be sorted in asc order.
  Node *build_tree_bulk(T *_keys, P *_values, int _size) {
    if (USE_FMCD) {
      return build_tree_bulk_fmcd(_keys, _values, _size);
    } else {
      return build_tree_bulk_fast(_keys, _values, _size);
    }
  }
  /// bulk build, _keys must be sorted in asc order.
  /// split keys into three parts at each node.
  Node *build_tree_bulk_fast(T *_keys, P *_values, int _size) {
    RT_ASSERT(_size > 1);

    typedef struct {
      int begin;
      int end;
      int level; // top level = 1
      Node *node;
    } Segment;
    std::stack<Segment> s;

    Node *ret = new_nodes(1);
    s.push((Segment){0, _size, 1, ret});

    while (!s.empty()) {
      const int begin = s.top().begin;
      const int end = s.top().end;
      const int level = s.top().level;
      Node *node = s.top().node;
      s.pop();

      RT_ASSERT(end - begin >= 2);
      if (end - begin == 2) {
        Node *_ = build_tree_two(_keys[begin], _values[begin], _keys[begin + 1],
                                 _values[begin + 1]);
        memcpy(node, _, sizeof(Node));
        delete_nodes(_, 1);
      } else {
        T *keys = _keys + begin;
        P *values = _values + begin;
        const int size = end - begin;
        const int BUILD_GAP_CNT = compute_gap_count(size);

        node->is_two = 0;
        node->build_size = size;
        node->size = size;
        node->fixed = 0;
        node->num_inserts = node->num_insert_to_data = 0;

        int mid1_pos = (size - 1) / 3;
        int mid2_pos = (size - 1) * 2 / 3;

        RT_ASSERT(0 <= mid1_pos);
        RT_ASSERT(mid1_pos < mid2_pos);
        RT_ASSERT(mid2_pos < size - 1);

        const long double mid1_key =
            (static_cast<long double>(keys[mid1_pos]) +
             static_cast<long double>(keys[mid1_pos + 1])) /
            2;
        const long double mid2_key =
            (static_cast<long double>(keys[mid2_pos]) +
             static_cast<long double>(keys[mid2_pos + 1])) /
            2;

        node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
        const double mid1_target =
            mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
            static_cast<int>(BUILD_GAP_CNT + 1) / 2;
        const double mid2_target =
            mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
            static_cast<int>(BUILD_GAP_CNT + 1) / 2;

        node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
        node->model.b = mid1_target - node->model.a * mid1_key;
        RT_ASSERT(isfinite(node->model.a));
        RT_ASSERT(isfinite(node->model.b));

        const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
        node->model.b += lr_remains;
        node->num_items += lr_remains * 2;

        if (size > 1e6) {
          node->fixed = 1;
        }

        node->items = new_items(node->num_items);
        const int bitmap_size = BITMAP_SIZE(node->num_items);
        node->none_bitmap = new_bitmap(bitmap_size);
        node->child_bitmap = new_bitmap(bitmap_size);
        memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
        memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

        for (int item_i = PREDICT_POS(node, keys[0]), offset = 0;
             offset < size;) {
          int next = offset + 1, next_i = -1;
          while (next < size) {
            next_i = PREDICT_POS(node, keys[next]);
            if (next_i == item_i) {
              next++;
            } else {
              break;
            }
          }
          if (next == offset + 1) {
            BITMAP_CLEAR(node->none_bitmap, item_i);
            node->items[item_i].comp.data.key = keys[offset];
            node->items[item_i].comp.data.value = values[offset];
          } else {
            // ASSERT(next - offset <= (size+2) / 3);
            BITMAP_CLEAR(node->none_bitmap, item_i);
            BITMAP_SET(node->child_bitmap, item_i);
            node->items[item_i].comp.child = new_nodes(1);
            s.push((Segment){begin + offset, begin + next, level + 1,
                             node->items[item_i].comp.child});
          }
          if (next >= size) {
            break;
          } else {
            item_i = next_i;
            offset = next;
          }
        }
      }
    }

    return ret;
  }
  /// bulk build, _keys must be sorted in asc order.
  /// FMCD method.
  Node *build_tree_bulk_fmcd(T *_keys, P *_values, int _size) {
    RT_ASSERT(_size > 1);

    typedef struct {
      int begin;
      int end;
      int level; // top level = 1
      Node *node;
    } Segment;
    std::stack<Segment> s;

    Node *ret = new_nodes(1);
    s.push((Segment){0, _size, 1, ret});

    while (!s.empty()) {
      const int begin = s.top().begin;
      const int end = s.top().end;
      const int level = s.top().level;
      Node *node = s.top().node;
      s.pop();

      RT_ASSERT(end - begin >= 2);
      if (end - begin == 2) {
        Node *_ = build_tree_two(_keys[begin], _values[begin], _keys[begin + 1],
                                 _values[begin + 1]);
        memcpy(node, _, sizeof(Node));
        delete_nodes(_, 1);
      } else {
        T *keys = _keys + begin;
        P *values = _values + begin;
        const int size = end - begin;
        const int BUILD_GAP_CNT = compute_gap_count(size);

        node->is_two = 0;
        node->build_size = size;
        node->size = size;
        node->fixed = 0;
        node->num_inserts = node->num_insert_to_data = 0;

        // FMCD method
        // Here the implementation is a little different with Algorithm 1 in our
        // paper. In Algorithm 1, U_T should be (keys[size-1-D] - keys[D]) / (L
        // - 2). But according to the derivation described in our paper, M.A
        // should be less than 1 / U_T. So we added a small number (1e-6) to
        // U_T. In fact, it has only a negligible impact of the performance.
        {
          const int L = size * static_cast<int>(BUILD_GAP_CNT + 1);
          int i = 0;
          int D = 1;
          RT_ASSERT(D <= size - 1 - D);
          double Ut = (static_cast<long double>(keys[size - 1 - D]) -
                       static_cast<long double>(keys[D])) /
                          (static_cast<double>(L - 2)) +
                      1e-6;
          while (i < size - 1 - D) {
            while (i + D < size && keys[i + D] - keys[i] >= Ut) {
              i++;
            }
            if (i + D >= size) {
              break;
            }
            D = D + 1;
            if (D * 3 > size)
              break;
            RT_ASSERT(D <= size - 1 - D);
            Ut = (static_cast<long double>(keys[size - 1 - D]) -
                  static_cast<long double>(keys[D])) /
                     (static_cast<double>(L - 2)) +
                 1e-6;
          }
          if (D * 3 <= size) {
            stats.fmcd_success_times++;

            node->model.a = 1.0 / Ut;
            node->model.b =
                (L -
                 node->model.a * (static_cast<long double>(keys[size - 1 - D]) +
                                  static_cast<long double>(keys[D]))) /
                2;
            RT_ASSERT(isfinite(node->model.a));
            RT_ASSERT(isfinite(node->model.b));
            node->num_items = L;
          } else {
            stats.fmcd_broken_times++;

            int mid1_pos = (size - 1) / 3;
            int mid2_pos = (size - 1) * 2 / 3;

            RT_ASSERT(0 <= mid1_pos);
            RT_ASSERT(mid1_pos < mid2_pos);
            RT_ASSERT(mid2_pos < size - 1);

            const long double mid1_key =
                (static_cast<long double>(keys[mid1_pos]) +
                 static_cast<long double>(keys[mid1_pos + 1])) /
                2;
            const long double mid2_key =
                (static_cast<long double>(keys[mid2_pos]) +
                 static_cast<long double>(keys[mid2_pos + 1])) /
                2;

            node->num_items = size * static_cast<int>(BUILD_GAP_CNT + 1);
            const double mid1_target =
                mid1_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
                static_cast<int>(BUILD_GAP_CNT + 1) / 2;
            const double mid2_target =
                mid2_pos * static_cast<int>(BUILD_GAP_CNT + 1) +
                static_cast<int>(BUILD_GAP_CNT + 1) / 2;

            node->model.a = (mid2_target - mid1_target) / (mid2_key - mid1_key);
            node->model.b = mid1_target - node->model.a * mid1_key;
            RT_ASSERT(isfinite(node->model.a));
            RT_ASSERT(isfinite(node->model.b));
          }
        }
        RT_ASSERT(node->model.a >= 0);
        const int lr_remains = static_cast<int>(size * BUILD_LR_REMAIN);
        node->model.b += lr_remains;
        node->num_items += lr_remains * 2;

        if (size > 1e6) {
          node->fixed = 1;
        }

        node->items = new_items(node->num_items);
        const int bitmap_size = BITMAP_SIZE(node->num_items);
        node->none_bitmap = new_bitmap(bitmap_size);
        node->child_bitmap = new_bitmap(bitmap_size);
        memset(node->none_bitmap, 0xff, sizeof(bitmap_t) * bitmap_size);
        memset(node->child_bitmap, 0, sizeof(bitmap_t) * bitmap_size);

        for (int item_i = PREDICT_POS(node, keys[0]), offset = 0;
             offset < size;) {
          int next = offset + 1, next_i = -1;
          while (next < size) {
            next_i = PREDICT_POS(node, keys[next]);
            if (next_i == item_i) {
              next++;
            } else {
              break;
            }
          }
          if (next == offset + 1) {
            BITMAP_CLEAR(node->none_bitmap, item_i);
            node->items[item_i].comp.data.key = keys[offset];
            node->items[item_i].comp.data.value = values[offset];
          } else {
            // ASSERT(next - offset <= (size+2) / 3);
            BITMAP_CLEAR(node->none_bitmap, item_i);
            BITMAP_SET(node->child_bitmap, item_i);
            node->items[item_i].comp.child = new_nodes(1);
            s.push((Segment){begin + offset, begin + next, level + 1,
                             node->items[item_i].comp.child});
          }
          if (next >= size) {
            break;
          } else {
            item_i = next_i;
            offset = next;
          }
        }
      }
    }

    return ret;
  }

  void destory_pending() {
    for (int i = 0; i < 1024; ++i) {
      while (!pending_two[i].empty()) {
        Node *node = pending_two[i].top();
        pending_two[i].pop();

        delete_items(node->items, node->num_items);
        const int bitmap_size = BITMAP_SIZE(node->num_items);
        delete_bitmap(node->none_bitmap, bitmap_size);
        delete_bitmap(node->child_bitmap, bitmap_size);
        delete_nodes(node, 1);
      }
    }
  }

  void destroy_tree(Node *root) {
    std::stack<Node *> s;
    s.push(root);
    while (!s.empty()) {
      Node *node = s.top();
      s.pop();

      for (int i = 0; i < node->num_items; i++) {
        if (BITMAP_GET(node->child_bitmap, i) == 1) {
          s.push(node->items[i].comp.child);
        }
      }

      if (node->is_two) {
        RT_ASSERT(node->build_size == 2);
        RT_ASSERT(node->num_items == 8);
        node->size = 2;
        node->num_inserts = node->num_insert_to_data = 0;
        node->none_bitmap[0] = 0xff;
        node->child_bitmap[0] = 0;
        pending_two[omp_get_thread_num()].push(node);
      } else {
        delete_items(node->items, node->num_items);
        const int bitmap_size = BITMAP_SIZE(node->num_items);
        delete_bitmap(node->none_bitmap, bitmap_size);
        delete_bitmap(node->child_bitmap, bitmap_size);
        delete_nodes(node, 1);
      }
    }
  }

  int scan_and_destory_tree(
      Node *_subroot, T **keys, P **values, // keys here is ptr to ptr
      bool destory = true) { 

    std::list<Node *> bfs;
    std::list<Node *> lockedNodes;

    RT_DEBUG(
        "ADJUST: PHASE 1: BFS all sub tree locks rooted at %p, with %d keys",
        _subroot, _subroot->size.load());

    bfs.push_back(_subroot);

    while (!bfs.empty()) {
      Node *node = bfs.front();
      bfs.pop_front();

      bool needRestart = false;
      node->writeLockOrRestart(needRestart);
      if (needRestart) {
        // release locks on all locked ancestors
        RT_DEBUG("ADJUST: Xlock %p fail, unlocking all locked", node);
        for (auto &n : lockedNodes) {
          n->writeUnlock();
        }

        return -1;
      }
      // x-lock this node SUCCESS
      lockedNodes.push_back(node);

      RT_DEBUG("ADJUST: Xlock OK on node=%p", node);

      for (int i = 0; i < node->num_items;
           i++) { // the i-th entry of the node now

        if (BITMAP_GET(node->none_bitmap, i) ==
            0) { // it has data/child; not empty entry
          if (BITMAP_GET(node->child_bitmap, i) == 1) { // means it is a child
            bfs.push_back(node->items[i].comp.child);
            RT_DEBUG("ADJUST: BFS pushed to stack %p",
                     node->items[i].comp.child);
          }
        }
      }
    } // end while

    RT_DEBUG("ADJUST: BFS locks all granted. **PHASE 2: now",
             0); // as it must contain at least 1 arg after comma
    typedef std::pair<int, Node *> Segment; // <begin, Node*>
    std::stack<Segment> s;
    s.push(Segment(0, _subroot));

    const int ESIZE = _subroot->size;
    *keys = new T[ESIZE];
    *values = new P[ESIZE];

    while (!s.empty()) {
      int begin = s.top().first;
      Node *node = s.top().second;

      const int SHOULD_END_POS = begin + node->size;
      RT_DEBUG("ADJUST: collecting keys at %p, SD_END_POS (%d)= begin (%d) + "
               "size (%d)",
               node, SHOULD_END_POS, begin, node->size.load());
      s.pop();

      int tmpnumkey = 0;

      for (int i = 0; i < node->num_items;
           i++) { // the i-th entry of the node now

        if (BITMAP_GET(node->none_bitmap, i) ==
            0) { // it has data/child; not empty entry
          if (BITMAP_GET(node->child_bitmap, i) == 0) { // means it is a data
            (*keys)[begin] = node->items[i].comp.data.key;
            (*values)[begin] = node->items[i].comp.data.value;
            begin++;
            tmpnumkey++;
          } else {
            RT_DEBUG("ADJUST: so far %d keys collected in this node",
                     tmpnumkey);
            s.push(Segment(begin,
                           node->items[i].comp.child)); // means it is a child
            RT_DEBUG("ADJUST: also pushed <begin=%d, a subtree at child %p> of "
                     "size %d to stack",
                     begin, node->items[i].comp.child,
                     node->items[i].comp.child->size.load());
            begin += node->items[i].comp.child->size;
            RT_DEBUG("ADJUST: begin is updated to=%d", begin);
          }
        } else { // this i-th entry is empty
        }
      }

      if (!(SHOULD_END_POS == begin)) {
        RT_DEBUG("ADJUST Err: just finish working on %p: begin=%d; "
                 "node->size=%d, node->num_items=%d, SHOULD_END_POS=%d",
                 node, begin, node->size.load(), node->num_items.load(),
                 SHOULD_END_POS);
        // show();
        RT_ASSERT(false);
      }
      RT_ASSERT(SHOULD_END_POS == begin);

      if (destory) { // pass to memory reclaimation memory later; @BT
        if (node->is_two) {
          RT_ASSERT(node->build_size == 2);
          RT_ASSERT(node->num_items == 8);
          node->size = 2;
          node->num_inserts = node->num_insert_to_data = 0;
          node->none_bitmap[0] = 0xff;
          node->child_bitmap[0] = 0;
          // node->writeUnlock();
          // pending_two[omp_get_thread_num()].push(node);
          safe_delete_nodes(node, 1);
        } else {
          /*
          delete_items(node->items, node->num_items);
          const int bitmap_size = BITMAP_SIZE(node->num_items);
          delete_bitmap(node->none_bitmap, bitmap_size);
          delete_bitmap(node->child_bitmap, bitmap_size);
          delete_nodes(node, 1);
          */
          safe_delete_nodes(node, 1);
        }
      }
    } // end while
    return ESIZE;
  } // end scan_and_destory

  // Node* insert_tree(Node *_node, const T &key, const P &value) {
  bool insert_tree(const T &key, const P &value) {
    int restartCount = 0; 
  restart:
    if (restartCount++)
      yield(restartCount);
    bool needRestart = false;

    constexpr int MAX_DEPTH = 128;
    Node *path[MAX_DEPTH];
    int path_size = 0;
    int insert_to_data = 0;

    // for lock coupling
    Node *parent = nullptr;
    uint64_t versionParent;

    for (Node *node = root;;) {
      // R-lock this node
      uint64_t versionNode = node->readLockOrRestart(needRestart);
      if (needRestart) {
        node->restart_cnt++;
        goto restart;
      }

      RT_ASSERT(path_size < MAX_DEPTH);
      path[path_size++] = node;

      int pos = PREDICT_POS(node, key);

      if (BITMAP_GET(node->none_bitmap, pos) == 1) // 1 means empty entry
      {
        node->upgradeToWriteLockOrRestart(versionNode, needRestart);
        if (needRestart) {
          node->restart_cnt++;
          goto restart;
        }

        BITMAP_CLEAR(node->none_bitmap, pos);
        node->items[pos].comp.data.key = key;
        node->items[pos].comp.data.value = value;

        RT_DEBUG("Key %d inserted into node %p.  Unlock", key, node);

        node->writeUnlock(); // X-UNLOCK this node; as long as 1 node is locked,
                      // other threads can't carry out adjust

        break;
      } else if (BITMAP_GET(node->child_bitmap, pos) ==
                 0) // 0 means existing entry has data already
      {
        node->upgradeToWriteLockOrRestart(versionNode, needRestart);
        if (needRestart) {
          node->restart_cnt++;
          goto restart;
        }

        BITMAP_SET(node->child_bitmap, pos);
        node->items[pos].comp.child =
            build_tree_two(key, value, node->items[pos].comp.data.key,
                           node->items[pos].comp.data.value);
        insert_to_data = 1;

        RT_DEBUG("Key %d inserted into node %p.  Unlock", key, node);

        node->writeUnlock(); // X-UNLOCK this node; as long as 1 node is locked,
                      // other threads can't carry out adjust
        
        break;
      } else // 1 means has a child, need to go down and see
      {
        // set parent=<current inner node>, and set node=<child-node>
        parent = node;
        versionParent = versionNode;
        
        node = node->items[pos].comp.child;           // now: node is the child

        parent->checkOrRestart(
            versionParent, needRestart); // to ensure nobody else has modified
                                         // the new parent in between
        if (needRestart) {
          parent->restart_cnt++;
          goto restart;
        } // if child is locked by another thread, restart
      }
    }

    for (int i = 0; i < path_size; i++) {
      path[i]->num_insert_to_data += insert_to_data;
      path[i]->num_inserts++;
      path[i]->size++;
      RT_DEBUG("Post insert(%d): update per node stat: %p size=%d, "
                "num_insert=%d, num_insert_to_data=%d",
                key, path[i], path[i]->size.load(), path[i]->num_inserts,
                path[i]->num_insert_to_data);
    }

    //***** so, when reaching here, no node is locked.

    for (int i = 0; i < path_size; i++) {
      Node *node = path[i];
      const int num_inserts = node->num_inserts;
      const int num_insert_to_data = node->num_insert_to_data;
      const bool need_rebuild =
          node->fixed == 0 && node->size >= node->build_size * 4 &&
          node->size >= 64 && num_insert_to_data * 10 >= num_inserts;

      // const bool need_rebuild = false; //@Poh: comment and uncomment above and here

      if (need_rebuild) {
        // const int ESIZE = node->size; //race here
        // T *keys = new T[ESIZE];
        // P *values = new P[ESIZE];
        T *keys;   // make it be a ptr here because we will let scan_and_destroy
                   // to decide the size after getting the locks
        P *values; // scan_and_destroy will fill up the keys/values

#if COLLECT_TIME
        auto start_time_scan = std::chrono::high_resolution_clock::now();
#endif

        int numKeysCollected = scan_and_destory_tree(
            node, &keys, &values); // pass the (address) of the ptr
        if (numKeysCollected < 0) {
          for (int x = 0; x < numKeysCollected; x++) {
            delete &keys[x]; // keys[x] stores keys
            delete &values[x];
          }
          RT_DEBUG("collectKey for adjusting node %p -- one Xlock fails; quit "
                   "rebuild",
                   node);
          break; // give up rebuild on this node (most likely other threads have
                 // done it for you already)
        }
#if COLLECT_TIME
        auto end_time_scan = std::chrono::high_resolution_clock::now();
        auto duration_scan = end_time_scan - start_time_scan;
        stats.time_scan_and_destory_tree +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration_scan)
                .count() *
            1e-9;
#endif

#if COLLECT_TIME
        auto start_time_build = std::chrono::high_resolution_clock::now();
#endif
        Node *new_node = build_tree_bulk(keys, values, numKeysCollected);
#if COLLECT_TIME
        auto end_time_build = std::chrono::high_resolution_clock::now();
        auto duration_build = end_time_build - start_time_build;
        stats.time_build_tree_bulk +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(duration_build)
                .count() *
            1e-9;
#endif

        delete[] keys;
        delete[] values;

        RT_DEBUG(
            "Final step of adjust, try to update parent/root, new node is %p",
            node);

        path[i] = new_node;
        if (i > 0) {

          int retryLockCount = 0;
        retryLock:
          if (retryLockCount++)
            yield(retryLockCount);

          int pos = PREDICT_POS(path[i - 1], key);

          bool needRetry = false;

          path[i - 1]->writeLockOrRestart(needRetry);
          if (needRetry) {
            RT_DEBUG("Final step of adjust, obtain parent %p lock FAIL, retry",
                     path[i - 1]);
            goto retryLock;
          }
          RT_DEBUG("Final step of adjust, obtain parent %p lock OK, now give "
                   "the adjusted tree to parent",
                   path[i - 1]);
          path[i - 1]->items[pos].comp.child = new_node;
          path[i - 1]->writeUnlock();
          adjustsuccess++;
          RT_DEBUG("Adjusted success=%d", adjustsuccess);
        } else { // new node is the root, need to update it
          root = new_node;
        }

        break; // break out for the for loop
      }        // end REBUILD
    }          // end for

    // return path[0];
    return true;
  } // end of insert_tree
};

}

#endif
