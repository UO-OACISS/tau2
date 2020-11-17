
#include "Tau_caching_map.h"
#include "Tau_regular_map.h"
#include "dummy.h"
#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include "main.h"

using namespace std;

std::vector<size_t> keys;

TauCachingMap<size_t,dummy*> theCachingMap;
TauRegularMap<size_t,dummy*> theRegularMap;

#define handle_error_en(en, msg) \
    do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)

#if !defined(_MSC_VER) && !defined(__APPLE__)
void set_thread_affinity(int core) {
    if (!pin_threads_to_cores) { return; }
    int s;
    cpu_set_t cpuset;
    pthread_t thread;

    thread = pthread_self();

    /* Set affinity mask to include CPUs 0 to 7 */

    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (s != 0) handle_error_en(s, "pthread_setaffinity_np");
    return;
}
#endif

/* Thread function to test the caching map */
void fooCaching (size_t id) {
#if !defined(_MSC_VER) && !defined(__APPLE__)
    set_thread_affinity(id);
#endif
    for (size_t i = 0 ; i < iterations ; i++) {
        size_t key = keys[i];
        dummy* d = theCachingMap.findOrInsert(key, [&]() { return new dummy(id,key); });
        assert(d);
    }
}

/* Thread function to test the regular (shared) map */
void fooRegular (size_t id) {
#if !defined(_MSC_VER) && !defined(__APPLE__)
    set_thread_affinity(id);
#endif
    for (size_t i = 0 ; i < iterations ; i++) {
        size_t key = keys[i];
        dummy* d = theRegularMap.findOrInsert(key, [&]() { return new dummy(id,key); });
        assert(d);
    }
}

/* Thread function to test what should be the fastest map - each thread has their own local map */
void fooFastest (size_t id) {
#if !defined(_MSC_VER) && !defined(__APPLE__)
    set_thread_affinity(id);
#endif
    map<size_t, dummy*> fastestMap;
    for (size_t i = 0 ; i < iterations ; i++) {
        size_t key = keys[i];
        auto d = fastestMap.find(key);
        if (d == fastestMap.end()) {
            fastestMap[key] = new dummy(id,key);
        }
    }
}

int main() {
    // test the class with a bunch of threads
    vector<thread> threads;

    srand(time(NULL));

    cout << "Using " << numThreads << " threads, "
         << maxValue << " unique keys, "
         << iterations << " total iterations:" << endl;

    /* Generate a sequence of random keys first (so as not to perturb the test) */
    cout << "Making keys..." << endl;
    for (size_t i = 0 ; i < iterations ; i++) {
        size_t key = rand() % maxValue;
        keys.push_back(key);
    }

    /* Test the caching map with a bunch of threads */
    cout << "Caching map..." << endl;
    using namespace std::chrono;
    auto start = high_resolution_clock::now();

    for (size_t i = 0 ; i < numThreads ; i++) {
        thread t(fooCaching, i);
        threads.push_back(move(t));
    }

    for (size_t i = 0 ; i < numThreads ; i++) {
        threads[i].join();
    }

    auto stop = high_resolution_clock::now();
    auto cachingDuration = duration_cast<milliseconds>(stop - start);
    threads.clear();

    /* test the regular map with a bunch of threads */
    cout << "Shared map..." << endl;
    start = high_resolution_clock::now();

    for (size_t i = 0 ; i < numThreads ; i++) {
        thread t(fooRegular, i);
        threads.push_back(move(t));
    }

    for (size_t i = 0 ; i < numThreads ; i++) {
        threads[i].join();
    }

    stop = high_resolution_clock::now();
    auto regularDuration = duration_cast<milliseconds>(stop - start);
    threads.clear();

    /* test the theoretically fastest map (no sharing) with a bunch of threads */
    cout << "Local map..." << endl;
    start = high_resolution_clock::now();

    for (size_t i = 0 ; i < numThreads ; i++) {
        thread t(fooFastest, i);
        threads.push_back(move(t));
    }

    for (size_t i = 0 ; i < numThreads ; i++) {
        threads[i].join();
    }

    stop = high_resolution_clock::now();
    auto fastestDuration = duration_cast<milliseconds>(stop - start);

    /* Print out the maps, showing which thread created each entry */
    /*
    for (auto it : theCachingMap.getAll()) {
        dummy* d = it.second;
        cout << it.first << " = " << d->_key << "," << d->_tid << endl;
    }

    for (auto it : theRegularMap.getAll()) {
        dummy* d = it.second;
        cout << it.first << " = " << d->_key << "," << d->_tid << endl;
    }
    */

    cout << "Caching: " << cachingDuration.count() << " ms" << endl;
    cout << "Regular: " << regularDuration.count() << " ms" << endl;
    cout << "Fastest: " << fastestDuration.count() << " ms" << endl;
}