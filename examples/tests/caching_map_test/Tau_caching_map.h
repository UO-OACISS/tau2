#pragma once

#include <map>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <iostream>
#include <sstream>
#include <functional>
#include <assert.h>

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
        // is it in the cache?
        auto cached = _privateMap().find(key);
        if (cached != _privateMap().end()) {
            return cached->second;
        }
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess());
        // is it in the shared map?
        auto shared = _sharedMap().find(key);
        if (shared != _sharedMap().end()) {
            // copy it to the thread local cache
            _privateMap()[key] = shared->second;
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
        // is it in the cache?
        auto cached = _privateMap().find(key);
        if (cached != _privateMap().end()) {
            return cached->second;
        }
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess());
        // is it in the shared map?
        auto shared = _sharedMap().find(key);
        if (shared != _sharedMap().end()) {
            // copy it to the thread local cache
            _privateMap()[key] = shared->second;
            return shared->second;
        }
        // not here?  then create it.
        auto tmp = f();
        // put it in the shared map
        _sharedMap()[key] = tmp;
        // ...and in the private, cache map
        _privateMap()[key] = tmp;
        return tmp;
    }

    map<T, P>& getAll() {
        return _sharedMap();
    }

};
