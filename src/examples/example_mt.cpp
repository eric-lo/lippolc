#include <lipp.h>
#include <iostream>

#include "omp.h"

using namespace std;

int main()
{
    LIPP<int, int> lipp;

    int key_num = 100;
    omp_set_num_threads(3);

    #pragma omp parallel for
    for(int i = 0; i < key_num; i++) {
        lipp.insert(i, i);
    }

    //#pragma omp parallel for
    for(int i = 0; i < key_num; i++) {
        int val = lipp.at(i, false);
        if(val != i) printf("wrong payload at %d\n", i);
    }
    
    cout << "exists(1) = " << (lipp.exists(1) ? "true" : "false") << endl;
    cout << "exists(40) = " << (lipp.exists(40) ? "true" : "false") << endl;
    cout << "exists(400) = " << (lipp.exists(400) ? "true" : "false") << endl;


    // show tree structure
    lipp.show();

    return 0;
}
