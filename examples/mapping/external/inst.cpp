/* This example illustrates the use of the TAU Mapping API for tracking the 
time spent in each instance of an object. This example illustrates an 
external association which involves a has table lookup (more expensive than 
embedded) for each invocation of the routine. */

#include <Profile/Profiler.h>
#include <iostream.h>
#include <unistd.h>
#include <stdio.h>

class MyClass
{
  public:
	MyClass()
	{

	}
	~MyClass()
	{

	}

	void Run(void)
	{
	  TAU_MAPPING_OBJECT(runtimer)
	  TAU_MAPPING_LINK(runtimer, (TauGroup_t) this);  // EXTERNAL ASSOCIATION
	  TAU_MAPPING_PROFILE(runtimer); // For one object
	  TAU_PROFILE("MyClass::Run()", " void (void)", TAU_USER1); // For all
	
	  cout <<"Sleeping for 2 secs..."<<endl;
	  sleep(2);
	}
  private:
};

int main(int argc, char **argv)
{
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
  MyClass x, y, z;
  MyClass a;
  TAU_MAPPING_CREATE("MyClass::Run() for object a", " " , (TauGroup_t) &a, "TAU_USER", 0);
  TAU_MAPPING_CREATE("MyClass::Run() for object x", " " , (TauGroup_t) &x, "TAU_USER", 0);
  TAU_PROFILE_SET_NODE(0);
  cout <<"Inside main"<<endl;

  a.Run();
  x.Run();
  y.Run();
}
  
