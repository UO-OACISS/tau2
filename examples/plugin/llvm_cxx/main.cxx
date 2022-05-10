
#include "headers2/B.h"

using namespace std;

int main ( int argc, char * argv[] ) {
    cout << "in " << __func__ << endl;
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
