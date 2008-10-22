#include <stdio.h>
#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <vector> 

using std::string;
int processInstrumentationRequests(char *fname);
bool processFileForInstrumentation(const string& file_name);

int main(int argc, char **argv) {

  if (argc != 3) {
    fprintf (stderr, "Usage: tau_selectfile <select.tau> <filename>\n");
    return -1;
  }
  
  processInstrumentationRequests(argv[1]);
  if (processFileForInstrumentation(argv[2])) {
    printf ("yes\n");
  } else {
    printf ("no\n");
  }
  return 0;
}
