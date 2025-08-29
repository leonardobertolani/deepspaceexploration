#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

inline uint64_t perf_counter_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)(ts.tv_sec) * 1000000000 + (uint64_t)ts.tv_nsec;
}

int main(int argc, char** argv) {
    struct timespec req;
    req.tv_sec = 0;
    req.tv_nsec = 100000000;
    uint64_t start_time = perf_counter_ns();

    for (int i = 0; i < 30; i++) {
        printf("%d\n", (perf_counter_ns() - start_time) / 1000000);
        fflush(stdout);
        system("xlnx_platformstats -p");
        nanosleep(&req, NULL);
    }
}
