#pragma once

class dummy {
    public:
        size_t _tid;
        size_t _key;
        dummy(size_t tid, size_t key) :
            _tid(tid), _key(key) {}
};