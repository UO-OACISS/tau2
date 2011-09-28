#ifndef _SS_ALLOCATOR_H
#define _SS_ALLOCATOR_H

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable:4018) //signed/unsigned mismatch
#pragma warning(disable:4290) // exception spec ignored
#endif
#include <new>
#include <list>
#include <algorithm>

template <typename T>
class tau_ss_storage
{
  enum ss_defaults{init_size = 0xfffff};
 public:
  tau_ss_storage():size_(sizeof(link)>sizeof(T)?init_size*sizeof(link):init_size*sizeof(T)),
    step_(sizeof(link)>sizeof(T)?sizeof(link):sizeof(T))
      {
	ss_mem_.push_back(new char[size_]);
	link *l;
	head_ = l = reinterpret_cast<link *>(*ss_mem_.begin());
	for(int i = 1; i<init_size; ++i){
	  l->next_ = reinterpret_cast<link *>(*ss_mem_.begin() + i*step_);
	  l = l->next_;
	}
	l->next_ = 0;
      }
    
  T* allocate()
    {
    link *l = head_;
    if(!l) grow();
    head_ = head_->next_;
    return reinterpret_cast<T *>(l);
  }

  void* allocate(size_t n)
    {
      link *l = head_, *conn, *ret;
      if(n%step_) n = step_*(n/step_+1);
      while(1){
	if(verify_memory_contiguous(l->next_, n/step_)){
	  conn = ret = l->next_;
	  for(unsigned int i=0; i<n/step_; ++i) conn = conn->next_;
	  l->next_ = conn;
	  return ret;
	}
	if(l->next_) l = l->next_;
	else {
	  grow();
	  l = head_;
	}
      }
    }

  void deallocate(void *p, size_t n)
  {
    if(n<=step_){
      link *l = head_;
      head_ = reinterpret_cast<link*>(p);
      head_->next_=l;
    }
    else{
      link *l = head_, *conn;
      head_ = reinterpret_cast<link*>(p);
      conn = head_;
      for(unsigned int i=0; i<n/step_; ++i){
	conn->next_ = reinterpret_cast<link*>(p) + i;
	conn = conn->next_;
      }
      conn->next_ = l;
    }
  }
  ~tau_ss_storage()
    {
      std::for_each(ss_mem_.begin(), ss_mem_.end(), killer());
    }
  
 private:
  struct link
  {
    link *next_;
  };
  void grow()
  {
    char *new_chunk = new char[size_];
    ss_mem_.push_back(new_chunk);
    link *old_head = head_;
    link *l = reinterpret_cast<link *>(new_chunk);
    head_ = l;
    for(int i = 1; i<init_size; ++i){
      l->next_ = reinterpret_cast<link *>(new_chunk + i*step_);
      l = l->next_;
    }
    l->next_ = old_head;
  }

  bool verify_memory_contiguous(link *l, int n)
  {
    if(!l) return false;
    for(int i=0; i<n; ++i){
      if(l->next_){
	if(reinterpret_cast<char*>(l->next_) - reinterpret_cast<char*>(l) == step_){
	  l = l->next_;
	}
	else{
	  return false;
	}
      }
      else{
	return false;
      }
    }
    return true;
  }
  struct killer
  {
    void operator()(char *p){delete [] p;}
  };
  size_t size_;
  size_t step_;
  std::list<char *> ss_mem_;
  link *head_;
};
#ifdef _WIN32
#pragma warning(pop)
#endif

template <typename T> class tau_ss_allocator;
template <> class tau_ss_allocator<void>
{
 public:
  typedef void* pointer;
  typedef const void* const_pointer;
  // reference to void members are impossible.
  typedef void value_type;
    template <class U>
      struct rebind { typedef tau_ss_allocator<U> other; };
};

namespace ss_alloc {
  inline void destruct(char *){}
  inline void destruct(wchar_t*){}
    template <typename T>
      inline void destruct(T *t){t->~T();}
} // namespace ss_alloc

template <typename T>
class tau_ss_allocator
{
 public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T* pointer;
  typedef const T* const_pointer;
  typedef T& reference;
  typedef const T& const_reference;
  typedef T value_type;

  template <class U>
    struct rebind { typedef tau_ss_allocator<U> other; };
  
  tau_ss_allocator(){}
  pointer address(reference x) const {return &x;}
  const_pointer address(const_reference x) const {return &x;}
  pointer allocate(size_type size, tau_ss_allocator<void>::const_pointer hint = 0) const
  {
    if(size == 1) return mem_.allocate();
    return static_cast<pointer>(mem_.allocate(size*sizeof(T)));
  }
  
  // For Dinkumware (VC6SP5):                                                 
  char *_Charalloc(size_type n){return static_cast<char*>(mem_.allocate(n));}
  // end Dinkumware
  
  template <class U> tau_ss_allocator(const tau_ss_allocator<U>&){}
  tau_ss_allocator(const tau_ss_allocator<T>&){}
  void deallocate(pointer p, size_type n) const
  {
    mem_.deallocate(p, n);
  }
  void deallocate(void *p, size_type n) const
  {
    mem_.deallocate(p, n);
  }
  size_type max_size() const throw() {return size_t(-1) / sizeof(value_type);}
  void construct(pointer p, const T& val)
  {
    new(static_cast<void*>(p)) T(val);
  }
  void construct(pointer p)
  {
    new(static_cast<void*>(p)) T();
  }
  void destroy(pointer p){ss_alloc::destruct(p);}
  //    static void dump() {mem_.dump();}
 private:
  static tau_ss_storage<T> mem_;
};

template <typename T> tau_ss_storage<T> tau_ss_allocator<T>::mem_;

template <typename T, typename U>
  inline bool operator==(const tau_ss_allocator<T>&, const tau_ss_allocator<U>){return true;}

template <typename T, typename U>
  inline bool operator!=(const tau_ss_allocator<T>&, const tau_ss_allocator<U>){return false;}

// For VC6/STLPort 4-5-3 see /stl/_alloc.h, line 464
// "If custom allocators are being used without member template classes support:                                                                              
// user (on purpose) is forced to define rebind/get operations !!!"
#ifdef _WIN32
#define SS_ALLOC_CDECL __cdecl
#else
#define SS_ALLOC_CDECL
#endif

namespace std {
  template <class _Tp1, class _Tp2>
    inline tau_ss_allocator<_Tp2>& SS_ALLOC_CDECL
    __stl_alloc_rebind(tau_ss_allocator<_Tp1>& __a, const _Tp2*)
  {
    return (tau_ss_allocator<_Tp2>&)(__a);
  }
  
  template <class _Tp1, class _Tp2>
    inline tau_ss_allocator<_Tp2> SS_ALLOC_CDECL
    __stl_alloc_create(const tau_ss_allocator<_Tp1>&, const _Tp2*)
  {
    return tau_ss_allocator<_Tp2>();
  }

} // namespace std
// end STLPort

#endif /* _SS_ALLOCATOR_H */
