/* This example illustrates the use of the TAU Mapping API for tracking the 
time spent in each instance of an object. We use an embedded association here. */

#include <Profile/Profiler.h>
#include <iostream.h>
#include <unistd.h>

class MyClass
{
  public:
	MyClass()
	{
	  TAU_MAPPING_LINK(runtimer, TAU_USER); 

	}
	~MyClass()
	{

	}

	void Run(void)
	{
	  TAU_MAPPING_PROFILE(runtimer); // For one object
	  TAU_PROFILE("MyClass::Run()", " void (void)", TAU_USER1); // For all
	
	  cout <<"Sleeping for 2 secs..."<<endl;
	  sleep(2);
	}
  private:
	TAU_MAPPING_OBJECT(runtimer)  // EMBEDDED ASSOCIATION
};

int main(int argc, char **argv)
{
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
  MyClass x, y, z;
  TAU_MAPPING_CREATE("MyClass::Run() for object a", " " , TAU_USER, "TAU_USER", 0);
  MyClass a;
  TAU_PROFILE_SET_NODE(0);
  cout <<"Inside main"<<endl;

  a.Run();
  x.Run();
  y.Run();
}
  
