/* Deliberately compiled as C, never nvcc/C++. */
#include "smollm.h"
#include <stdio.h>
int sm_cuda_bootstrap(SmError *error);
int main(void) {
    SmError error={{0}};
    if (sm_cuda_bootstrap(&error)) { fprintf(stderr,"%s\n",error.message); return 1; }
    return 0;
}
