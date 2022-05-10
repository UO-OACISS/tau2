#include "headers1/A.h"

using namespace std;

class B : public A {
    public:
        B() : A() {};
        ~B() { };
        void method1 () {
            cout << "in " << __func__ << endl;
        }
        void method2 ();
        virtual void vmethod ();
};

void B::method2() {
    cout << "in " << __func__ << endl;
}

void B::vmethod() {
    cout << "in " << __func__ << endl;
}
