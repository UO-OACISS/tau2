#ifndef TComplex_H
#define TComplex_H

#include <math.h>

template <class T>
class TComplex {
public:
  // default constructor
  TComplex(T r = 0.0, T i = 0.0) {re = r; im = i;}

  // field access functions
  T real(void) const {return re;}
  T imag(void) const {return im;}

  // addition operator
  const TComplex operator+(const TComplex<T>& rhs) const {
    return TComplex<T>(re+rhs.re, im+rhs.im);
  }

  // multiplication operator
  const TComplex operator*(const TComplex<T>& rhs) const {
    return TComplex<T>(re*rhs.re - im*rhs.im, re*rhs.im + im*rhs.re);
  }

private:
    T re, im;
};


template <class T>
T norm(const TComplex<T>& x) {
  return x.real()*x.real() + x.imag()*x.imag();
}

#endif
