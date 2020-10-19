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

class TauCachingMap {
private:
    map<size_t,dummy*> _privateMap;
    static map<size_t,dummy*> _sharedMap;
    static mutex _sharedAccess;

public:
    TauCachingMap() {};
    ~TauCachingMap() {};  // call the destructor trigger here!

    dummy* find(size_t key) {
        // is it in the cache?
        auto cached = _privateMap.find(key);
        if (cached != _privateMap.end()) {
            return cached->second;
        }
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess);
        // is it in the shared map?
        auto shared = _sharedMap.find(key);
        if (shared != _sharedMap.end()) {
            // copy it to the thread local cache
            _privateMap[key] = shared->second;
            return shared->second;
        }
        // not here.
        return nullptr;
    }

    // with a function pointer with defined type...
    //dummy* findOrInsert(size_t id, size_t key, std::function<dummy*(size_t, size_t)> p) {
    // ...or with a template
    template <typename P>
    dummy* findOrInsert(size_t id, size_t key, P p) {
        // is it in the cache?
        auto cached = _privateMap.find(key);
        if (cached != _privateMap.end()) {
            return cached->second;
        }
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess);
        // is it in the shared map?
        auto shared = _sharedMap.find(key);
        if (shared != _sharedMap.end()) {
            // copy it to the thread local cache
            _privateMap[key] = shared->second;
            return shared->second;
        }
        // not here?  then create it.
        auto tmp = p(id, key);
        // put it in the shared map
        _sharedMap[key] = tmp;
        // ...and in the private, cache map
        _privateMap[key] = tmp;
        return tmp;
    }

    map<size_t, dummy*>& getAll() {
        return _sharedMap;
    }

};
