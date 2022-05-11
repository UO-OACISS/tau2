#include "headers1/A.h"

class B : public A {
    public:
        B() : A(), flag2(false) {
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
        bool getFlag2() { return flag2; };
    private:
        bool flag2;
};

void B::method2() {
    REPORT
}

void B::vmethod() {
    REPORT
}
