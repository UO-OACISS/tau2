#include <stdio.h>
int
foo()
{
    int i = 0;
#pragma pomp inst begin(foo)
    //usefull work could be done here which changes i
    printf( "Hello from foo.\n" );
    if ( i == 0 )
    {
#pragma pomp inst altend(foo)
        return 42;
    }
    //other work might be done here
    return i;
#pragma pomp inst end(foo)
}

int
main( int argc, char** argv )
{
#pragma pomp inst init
    printf( "Hello from main.\n" );
    foo();
    return 0;
}
