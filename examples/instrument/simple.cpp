/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
// Note : This is a trivial application that does nothing. It is written to 
// illustrate how to insert TAU Profiling Instrumentation in an application
// that uses templates.
// Tests the classes, functions, templated classes, templated functions with
// profiling. 
// Some compilers donot support member templates of class templates yet, so 
// NEWCC is defined so that compilation doesn't break on most systems.
#include "Profile/Profiler.h"
#include <sys/types.h>
#include <unistd.h>
#ifdef TAU_DOT_H_LESS_HEADERS
#include <iostream>
using namespace std;
#else /* TAU_DOT_H_LESS_HEADERS */ 
#include <iostream.h>
#endif /* TAU_DOT_H_LESS_HEADERS */

TAU_REGISTER_EVENT(getdata, "Get Data values");

class NoTempl { // Simple class with Get and Set methods
  private :
    int Data;
  public : 
    NoTempl() {
      TAU_PROFILE("NoTempl::NoTempl()","void ()", TAU_USER); 
    };
    int GetData (void) { 
      TAU_PROFILE("NoTempl::GetData()","void ()", TAU_UTILITY | TAU_USER);
      TAU_EVENT(getdata,(double) Data);
      return Data;
    }
    int SetData (int d) {
      TAU_PROFILE("NoTempl::SetData()", "int (int)", TAU_SPARSE | TAU_USER);
      Data = d;
      return Data;
    }

#ifdef NEWCC
    // Templated member function in a non-templated class.
    template <class Dim> 
    int SetDataTemplMem (Dim dimen, int dat) {

      TAU_TYPE_STRING(p1, "int (" + CT(dimen) + ", int)" ); 
      TAU_PROFILE("NoTempl::SetDataTemplMem()", p1, 
	TAU_UTILITY | TAU_FIELD | TAU_IO | TAU_USER);
//      PoomaInfo::abort("Exiting from NoTempl::SetDataTemplMem");
      Data = dat * dimen;
      return Data;
    }
#endif //NEWCC

    ~NoTempl() {
      TAU_PROFILE("NoTempl::~NoTempl","void ()", TAU_REGION | TAU_USER);
     }
};


template <class T>
class TemplClass { // Simple class template with Get and Set methods
  private :
    T Data;
  public :
    TemplClass() { 
      TAU_TYPE_STRING(taustr, CT(*this) + " void (void)");
      TAU_PROFILE("TemplClass::TemplClass()", taustr, TAU_USER);
    };
    T GetData(void) {
      TAU_TYPE_STRING(taustr, CT(*this) + " void (void)");
      TAU_PROFILE("TemplClass::GetData()", taustr, TAU_ACLMPL | TAU_USER);
      TAU_EVENT(getdata,(double) Data * 2);
      return Data;
    }
    T SetData(T d) {
      TAU_TYPE_STRING(taustr, CT(*this) + " void (void)");
      TAU_PROFILE("TemplClass::SetData()", taustr, TAU_FFT | TAU_USER);
      Data = d;
      return Data;
    }
   
#ifdef NEWCC
    // member template of a class template
    template <class Dim> 
    T SetDataTemplMem (Dim dimen, T dat) {

      TAU_TYPE_STRING(p1, CT(Data) + " (" + CT(dimen) + ", " + CT(dat) + " )" );
      TAU_PROFILE("TemplClass::SetDataTemplMem()", p1, TAU_USER1 | TAU_USER);

      Data = dat * dimen;
      return Data;
    }
#endif //NEWCC

    ~TemplClass() { 
	
      TAU_TYPE_STRING(taustr, CT(*this) + " void (void)");
      TAU_PROFILE("TemplClass::~TemplClass()", taustr, TAU_USER2 | TAU_USER);
      // Can't get address of constructor or destructor in C++. Using class name
    }
};

// Simple non-templated function  equivalent to getpid()
int func_no_templ(void) {
  TAU_PROFILE("func_no_templ","int ()", TAU_USER3 | TAU_USER);

  int x;

  x = getpid();
  return x;
}

// Function template returns size of the type with which it is instantiated
template <class T>
int func_templ( T x) {
  TAU_TYPE_STRING(buf, "int (" + CT(x) + " )" );
  TAU_PROFILE("func_templ", buf, TAU_PAWS3 | TAU_USER);

  return sizeof(x);
}



int main(int argc, char *argv[])
{
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(0); // sequential code 
  TAU_PROFILE("main", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_TIMER(notemp, "main-notemp-loop", "int (int, char **)", TAU_USER1 | TAU_USER);
  TAU_PROFILE_TIMER(templ, "main-temp", "int (int, char **)", TAU_USER1 | TAU_USER);
  


// Testing data 
  NoTempl X;
  TemplClass<double> Y;
  TemplClass<char> C;
  int idim = 1;
  double ddim = 3.0;




  // Testing the classes and functions defined above 
  X.SetData(34);
  if (X.GetData() != 34) { 
    cout <<"NoTempl Class Test failed X.GetData() = "<< X.GetData() << endl;
  }
  else
    cout <<"Test Successful : NoTempl::GetData() " << endl;


  // Testing TemplClass
  Y.SetData(34);
  if (X.GetData() != (int) Y.GetData() ) {
    cout <<"Test failed - X.GetData = "<< X.GetData() << " Y.GetData = "
      << Y.GetData() << endl;
  }
  else
    cout << "Test Successful : TemplClass<double>::GetData() " << endl;
  

  // Calling TemplClass with another instantiation 
  C.SetData(34);
  if (X.GetData() != (int) C.GetData() ) {
    cout <<"Test failed - X.GetData = "<< X.GetData() << " C.GetData = "
      << C.GetData() << endl;
  }
  else
	cout << "Test Successful : TemplClass<char>::GetData() " << endl;

  TAU_PROFILE_START(notemp);
#ifdef NEWCC
  // Testing NoTempl::SetDataTemplMem <int,int> function
  if (X.SetDataTemplMem(idim, 100) != (int) (idim * 100)) {
    cout<<"X.SetDataTemplMem test failed X.SetDataTemplMem(idim, 100) = "<< 
      X.SetDataTemplMem(idim,100) << " idim = "<< idim << endl; 
  }
  else	
    cout << "Test Successful : NoTempl::SetDataTemplMem<int,int>() " << endl;

  // Testing NoTempl::SetDataTemplMem <double,int> function
  if (X.SetDataTemplMem(ddim, 100) != (int) (ddim * 100)) {
    cout<<"X.SetDataTemplMem test failed X.SetDataTemplMem(ddim, 100) = "<< 
      X.SetDataTemplMem(ddim,100) << " ddim = "<< ddim << endl; 
  }
  else	
    cout << "Test Successful : NoTempl::SetDataTemplMem<double,int>() " << endl;

#endif //NEWCC
  TAU_PROFILE_STOP(notemp);
  // Testing func_no_templ
  if (getpid() != func_no_templ()) {
    cout <<"func_no_templ test failed func_no_templ returns "<< func_no_templ() << endl;
  }
  else
    cout <<"Test Successful : func_no_templ() " << endl;

  TAU_PROFILE_START(templ);
  // Testing func_templ
  if (sizeof(double) != func_templ (Y.GetData())) {
    cout <<"Test Failed sizeof (double) = "<< sizeof(double) << 
      " func_templ(Y.GetData()) returns "<< func_templ(Y.GetData()) << endl;
  }
  else
    cout <<"Test Successful : func_templ<double>() "<<endl;
 
  if (sizeof(char) != func_templ (C.GetData())) {
    cout <<"Test Failed sizeof (double) = "<< sizeof(double) << 
      " func_templ(C.GetData()) returns "<< func_templ(C.GetData()) << endl;
  }
  else
    cout <<"Test Successful : func_templ<char>() "<<endl;

#ifdef NEWCC
  // Testing TemplClass::SetDataTemplMem <int,double> function
  if (Y.SetDataTemplMem(idim, 100) != (double) (idim * 100.0)) {
    cout<<"Y.SetDataTemplMem test failed Y.SetDataTemplMem(idim, 100) = "<< 
      Y.SetDataTemplMem(idim,100) << " idim = "<< idim << endl; 
  }
  else	
    cout <<
      "Test Successful : TemplClasss<double>::SetDataTemplMem<int,double>()" 
      << endl;

  // Testing TemplClass::SetDataTemplMem <double,double> function
  if (Y.SetDataTemplMem(ddim, 10) != (double) (ddim * 10)) {
    cout<<"Y.SetDataTemplMem test failed Y.SetDataTemplMem(ddim, 100) = "<< 
      Y.SetDataTemplMem(ddim,100) << " ddim = "<< ddim << endl; 
  }
  else	
    cout << 
      "Test Successful : TemplClasss<double>::SetDataTemplMem<double,double>() "      << endl;
  
  // Testing TemplClass::SetDataTemplMem <int,char> function
  if (C.SetDataTemplMem(idim, 100) != (char) (idim * 100)) {
    cout<<"C.SetDataTemplMem test failed C.SetDataTemplMem(idim, 100) = "<< 
	C.SetDataTemplMem(idim,100) << " idim = "<< idim << endl; 
  }
  else	
    cout << "Test Successful : TemplClasss<char>::SetDataTemplMem<int,char>() " << endl;

  // Testing TemplClass::SetDataTemplMem <double,char> function
  if (C.SetDataTemplMem(ddim, 10) != (char) (ddim * 10)) {
    cout<<"C.SetDataTemplMem test failed C.SetDataTemplMem(ddim, 10) = "<< 
	C.SetDataTemplMem(ddim,10) << " ddim = "<< ddim << endl; 
  }
  else	
    cout << 
      "Test Successful : TemplClasss<char>::SetDataTemplMem<double,char>() " 
      << endl;
#endif //NEWCC

  TAU_PROFILE_STOP(templ);


  return 0;
}
