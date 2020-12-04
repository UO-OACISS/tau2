#pragma once

#include <map>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <iostream>
#include <sstream>
#include <functional>
#include <assert.h>
#include "dummy.h"

using namespace std;

template <typename T, typename P>
class TauCachingMap {
private:
    map<T,P>& _privateMap() {
        static thread_local map<T,P> theMap;
        return theMap;
    }
    map<T,P>& _sharedMap() {
        static map<T,P> theMap;
        return theMap;
    }
    mutex& _sharedAccess() {
        static mutex theMutex;
        return theMutex;
    }

public:
    TauCachingMap() {};
    ~TauCachingMap() {};  // call the destructor trigger here!

    P find(T key) {
        auto& pmap = _privateMap();
        // is it in the cache?
        auto cached = pmap.find(key);
        if (cached != pmap.end()) {
            return cached->second;
        }
        auto& smap = _sharedMap();
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess());
        // is it in the shared map?
        auto shared = smap.find(key);
        if (shared != smap.end()) {
            // copy it to the thread local cache
            pmap[key] = shared->second;
            return shared->second;
        }
        // not here.
        return nullptr;
    }

    // with a function pointer with defined type...
    //dummy* findOrInsert(size_t key, std::function<dummy*(size_t, size_t)> p) {
    // ...or with a template
    template<typename Func>
    P findOrInsert(T key, Func f) {
        auto& pmap = _privateMap();
        // is it in the cache?
        auto cached = pmap.find(key);
        if (cached != pmap.end()) {
            return cached->second;
        }
        auto& smap = _sharedMap();
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess());
        // is it in the shared map?
        auto shared = smap.find(key);
        if (shared != smap.end()) {
            // copy it to the thread local cache
            pmap[key] = shared->second;
            return shared->second;
        }
        // not here?  then create it.
        auto tmp = f();
        // put it in the shared map
        smap[key] = tmp;
        // ...and in the private, cache map
        pmap[key] = tmp;
        return tmp;
    }

    map<T, P>& getAll() {
        return _sharedMap();
    }

};

/* this has to be a different type for the test, otherwise the
 * map gets reused, and there are no inserts */
class TauDummyMap : public TauCachingMap<size_t, dummy2*> {
private:
    TauDummyMap() {};
    ~TauDummyMap() {};  // call the destructor trigger here!
public:
    static TauDummyMap& instance() {
        static TauDummyMap _i;
        return _i;
    }
};

