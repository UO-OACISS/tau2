#include <iostream>

using namespace std;

class A {
    public:
        A() {};
        ~A() {};
        void method1 () {
            cout << "in " << __func__ << endl;
        }
        void method2 ();
        virtual void vmethod();
};

void A::method2() {
    cout << "in " << __func__ << endl;
}

void A::vmethod() {
    cout << "in " << __func__ << endl;
}
