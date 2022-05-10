#include <iostream>

// macro to report source location.  The line number is subtracted by 1 to
// match the beginning of the function definition.
#define REPORT \
std::cout << "in " << __func__ << " line " << __FILE__ << " line " << (__LINE__-1) << std::endl;

class A {
    public:
        A() {
            REPORT
        };
        ~A() {
            REPORT
        };
        void method1 () {
            REPORT
        }
        void method2 ();
        virtual void vmethod();
};

void A::method2() {
    REPORT
}

void A::vmethod() {
    REPORT
}
