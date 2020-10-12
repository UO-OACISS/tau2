#pragma once

#include <map>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <iostream>
#include "dummy.h"

using namespace std;

class TauRegularMap {
private:
    static map<size_t,dummy*> _sharedMap;
    static mutex _sharedAccess;

    // ASSUMES THE LOCK IS ACQUIRED FIRST!
    dummy* _find(size_t key) {
        // is it in the shared map?
        auto shared = _sharedMap.find(key);
        if (shared != _sharedMap.end()) {
            return shared->second;
        }
        // not here?  then create it.
        return nullptr;
    }
public:
    TauRegularMap() {};
    ~TauRegularMap() {};

    dummy* find(size_t key) {
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess);
        return _find(key);
    }

    dummy* insert(size_t key, dummy* value) {
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess);
        auto tmp = _find(key);
        if (tmp != nullptr) { return tmp; }
        _sharedMap[key] = value;
        return value;
    }

    template <typename P>
    dummy* findOrInsert(size_t id, size_t key, P p) {
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess);
        // is it in the shared map?
        auto shared = _sharedMap.find(key);
        if (shared != _sharedMap.end()) {
            return shared->second;
        }
        // not here?  then create it.
        auto tmp = p(id, key);
        _sharedMap[key] = tmp;
        return tmp;
    }

    map<size_t, dummy*>& getAll() {
        return _sharedMap;
    }

};
