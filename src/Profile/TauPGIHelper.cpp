/* This file is required to compile TAU with PGI 1.7 pgCC. It doesn't provide
the pgCC --prelink_objects prior to putting the objects in the archive using
ar rcv lib<> *.o. So, this is included during compilation. Its a fix.
*/
#include <iostream>
using namespace std;

void NoOneInvokesThis(void)
{
  cout <<"TAU PGI 1.7 Helper" <<endl;
}
