#include "headers1/A.h"

class B : public A {
    public:
        B() : A() {
            REPORT
        };
        ~B() {
            REPORT
        };
        void method1 () {
            REPORT
        }
        void method2 ();
        virtual void vmethod ();
};

void B::method2() {
    REPORT
}

void B::vmethod() {
    REPORT
}
