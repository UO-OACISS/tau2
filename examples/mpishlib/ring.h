#ifndef _RING_H_
#define _RING_H_
#include <Profile/Profiler.h>
class C {
public:
  C(int m, int p) : me(m), proc(p) {
  TAU_PROFILE("C &C::C(int, int)", " ", TAU_DEFAULT);
  }
  void method();

private:
  int proc, me;
};

#endif /* _RING_H_ */
