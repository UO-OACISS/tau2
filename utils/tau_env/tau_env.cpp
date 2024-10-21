#include "Profile/TauUtil.h"
#include <unistd.h>
#include <string>
#include "Profile/TauEnv.h"
#include <iostream>
#include <iomanip>
using namespace std;

extern int Tau_metadata_writeMetaData(Tau_util_outputDevice *out);

int main(int argc, char **argv)
{
	int c = getopt(argc, argv, "h");

  if (c == 'h') {
    cout << "tau_env : Writes to stdout the list of possible TAU environment variables and well as their values as currently read from the environment (or default values).\nNo options. " << endl;
    exit(0);
  }

  TauEnv_initialize();

  Tau_util_outputDevice *out = Tau_util_createBufferOutputDevice();

  int retval = Tau_metadata_writeMetaData(out);

  if (retval == 0) {
    string metadata = Tau_util_getOutputBuffer(out);

    /** Quick and dirty xml parser that removes the xml tags and prints the
     * name/value pairs one at a time.
     */
    
    // keeps track of the position in the string of the attribute we are
    // currently processing
    int pos = 0;

    int attr_start, name_start, value_start;
    int attr_end, name_end, value_end;

    const string ATTRIBUTE_OPEN = string("<attribute>");
    const string ATTRIBUTE_CLOSE = string("</attribute>");
    const size_t ATTRIBUTE_OPEN_LEN = ATTRIBUTE_OPEN.length();
    const size_t ATTRIBUTE_CLOSE_LEN = ATTRIBUTE_CLOSE.length();
    const string NAME_OPEN = string("<name>");
    const string NAME_CLOSE = string("</name>");
    const size_t NAME_OPEN_LEN = NAME_OPEN.length();
    const size_t NAME_CLOSE_LEN = NAME_CLOSE.length();
    const string VALUE_OPEN = string("<value>");
    const string VALUE_CLOSE = string("</value>");
    const size_t VALUE_OPEN_LEN = VALUE_OPEN.length();
    const size_t VALUE_CLOSE_LEN = VALUE_CLOSE.length();

    while(true)
    {
    
      //find the next attribute tag.
      attr_start = metadata.find(ATTRIBUTE_OPEN, pos);

      //if no open attribute tag found we must be at the end of the string.
      if (attr_start == string::npos) {
        break;
      }
      //find the next ending attribute tag.
      attr_end = metadata.find(ATTRIBUTE_CLOSE, attr_start);

      //something goes wrong if we cannot find closing attribute tag or it
      //occurs before the opening attribute tag.
      if (attr_end == string::npos || attr_end <= attr_start) {
        cerr << "Invalid metadata output." << endl;
        exit(1);
      }
      
      //move the starting position so as not to print the opening tag.
      attr_start += ATTRIBUTE_OPEN_LEN;

      name_start = metadata.find(NAME_OPEN, attr_start);
      name_end = metadata.find(NAME_CLOSE, name_start);

      value_start = metadata.find(VALUE_OPEN, attr_start);
      value_end = metadata.find(VALUE_CLOSE, value_start);

      //move the starting position so as not to print the opening tag.
      name_start += NAME_OPEN_LEN;
      value_start += VALUE_OPEN_LEN;

      //print name
      cout << left;
      cout << setw(35);
      cout << metadata.substr(name_start , name_end - (name_start));
      cout << " : ";
      //print value 
      cout << setw(15);
      cout << metadata.substr(value_start , value_end - (value_start));
      cout << endl;
    
      pos = attr_end;
    }
  }

  return retval;
}
