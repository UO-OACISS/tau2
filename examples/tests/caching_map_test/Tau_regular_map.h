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
class TauRegularMap {
private:
    map<T,P>& _sharedMap() {
        static map<T,P> theMap;
        return theMap;
    }
    mutex& _sharedAccess() {
        static mutex theMutex;
        return theMutex;
    }

public:
    TauRegularMap() {};
    ~TauRegularMap() {};  // call the destructor trigger here!

    P find(T key) {
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess());
        // is it in the shared map?
        auto shared = _sharedMap().find(key);
        if (shared != _sharedMap().end()) {
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
        // acquire the lock
        const lock_guard<mutex> lock(_sharedAccess());
        // is it in the shared map?
        auto shared = _sharedMap().find(key);
        if (shared != _sharedMap().end()) {
            return shared->second;
        }
        // not here?  then create it.
        auto tmp = f();
        // put it in the shared map
        _sharedMap()[key] = tmp;
        return tmp;
    }

    map<T, P>& getAll() {
        return _sharedMap();
    }

};
