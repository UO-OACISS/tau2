int foo(int x) { return x + 1; }
int bar_x(int x) { return x * 2; }
int baz() { return 3; }
namespace ns { int y() { return 4; } }
int main() { return foo(1) + bar_x(2) + baz() + ns::y(); }
