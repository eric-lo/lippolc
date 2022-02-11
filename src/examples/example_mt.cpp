#include <iostream>
#include <lipp.h>

#include "omp.h"

using namespace std;

int main() {
  lippolc::LIPP<int, int> lipp;

  int key_num = 500;
  omp_set_num_threads(3);

#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < key_num; i++) {
    // printf("Thread %d insert(%d)\n", omp_get_thread_num(), i);
    lipp.insert(i, i);
    // mix write with read
    // if (i>5)
    // printf("Thread %d, read %d\n", omp_get_thread_num(), lipp.at(i-5,
    // false));
  }

#pragma omp parallel for schedule(static, 4)
  for (int i = 0; i < key_num; i++) {
    int val = lipp.at(i, false);
    if (val != i)
      printf("wrong payload at %d\n", i);
  }

  cout << "exists(1) = " << (lipp.exists(1) ? "true" : "false") << endl;
  cout << "exists(100) = " << (lipp.exists(100) ? "true" : "false") << endl;
  cout << "exists(4000) = " << (lipp.exists(4000) ? "true" : "false") << endl;

  // show tree structure
  // lipp.show();

  return 0;
}
