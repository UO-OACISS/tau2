
#include "headers2/B.h"

int main ( int argc, char * argv[] ) {
    REPORT
    A a;
    B b;

    a.method1();
    b.method1();
    a.method2();
    b.method2();
    a.vmethod();
    b.vmethod();

    return 0;
}
