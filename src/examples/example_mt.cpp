#include <lipp.h>
#include <iostream>

#include "omp.h"

using namespace std;

int main()
{
    LIPP<int, int> lipp;

    int key_num = 100;
    omp_set_num_threads(2);

    #pragma omp parallel for schedule(static, 5)
    for(int i = 0; i < key_num; i++) {        
        printf("--Insert %d by Thread %d\n", i, omp_get_thread_num());  
        lipp.insert(i, i);        
        //mix write with read
        //if (i>5)
        //printf("Thread %d, read %d\n", omp_get_thread_num(), lipp.at(i-5, false));  
    }

    #pragma omp parallel for schedule(static, 4)
    for(int i = 0; i < key_num; i++) {
        int val = lipp.at(i, false);
        if(val != i) printf("wrong payload at %d\n", i);
    }
    
    cout << "exists(1) = " << (lipp.exists(1) ? "true" : "false") << endl;
    cout << "exists(400) = " << (lipp.exists(400) ? "true" : "false") << endl;
    cout << "exists(800) = " << (lipp.exists(800) ? "true" : "false") << endl;

    // show tree structure
    //lipp.show();

    return 0;
}
