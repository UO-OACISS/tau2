#ifndef TAU_MMAP_MEM_MGR_H_
#define TAU_MMAP_MEM_MGR_H_

#include <sys/types.h>
#include <cstddef>
#include <memory>
#include <typeinfo>
#include <stdio.h>

// Note that this is per-thread and is not capped at 1MB blocks.
#define TAU_MEMMGR_MAX_MEMBLOCKS 64
#define TAU_MEMMGR_DEFAULT_BLOCKSIZE 1048576 /* 1024x1024 In bytes */

#define TAU_MEMMGR_ALIGN sizeof(long) /* In bytes */

#define UNUSED(_x_) (void)(_x_)

extern "C" bool Tau_MemMgr_initIfNecessary(void);
extern "C" void *Tau_MemMgr_mmap(int tid, std::size_t size);
extern "C" void *Tau_MemMgr_malloc(int tid, std::size_t size);
extern "C" void Tau_MemMgr_free(int tid, void *addr, std::size_t size);
extern "C" void Tau_MemMgr_finalizeIfNecessary(void);

template <typename T>
class TauSignalSafeAllocator {
public:
    typedef T value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

public:
    // rebind structure
    template<typename U>
    struct rebind {
        typedef TauSignalSafeAllocator<U> other;
    };

public:
    // constructor
    TauSignalSafeAllocator() {};
    // destructor
    ~TauSignalSafeAllocator() {};
    // Copy methods
    TauSignalSafeAllocator(const TauSignalSafeAllocator& other) {UNUSED(other);};
    template <class U> TauSignalSafeAllocator(const TauSignalSafeAllocator<U>& other) {UNUSED(other);}
    // allocate method
    pointer allocate(size_type n, typename std::allocator<void>::const_pointer = 0) {
        //printf("Allocating %d of type %s\n", n, typeid(T).name()); fflush(stdout);
        T *ptr = (T*)Tau_MemMgr_malloc(RtsLayer::unsafeThreadId(), n*sizeof(T));
        return ptr;
    }
    // free method
    void deallocate(pointer p, size_type n) {
        //printf("Freeing %p of type %s, size %d\n", p, typeid(T).name(), n); fflush(stdout);
        Tau_MemMgr_free(RtsLayer::unsafeThreadId(), p, n*sizeof(T));
        //Tau_MemMgr_free(RtsLayer::unsafeThreadId(), p, n);
    }
    // address methods
    pointer address(reference r) { return &r; }
    const_pointer address(const_reference r) { return &r; }

    // construction/destruction
    inline void construct(pointer p, const T& t) { new(p) T(t); }
    inline void destroy(pointer p) { p->~T(); }

    // max size, needed for vectors and whatnot
    size_type max_size(void) const {
        return (TAU_MEMMGR_DEFAULT_BLOCKSIZE / sizeof(T));
    }

    template <class U>
    bool operator==(TauSignalSafeAllocator<U> const & rhs) const {
        UNUSED(rhs);
        return true;
    }

    template <class U>
    bool operator!=(TauSignalSafeAllocator<U> const & rhs) const {
        UNUSED(rhs);
        return false;
    }
};

#endif /* TAU_MMAP_MEM_MGR_H */
