// Evaluates the common SI test cases against the runtime filter plugin (plugins/examples/Tau_plugin_example_disable_instrumentation_runtime.cpp).
#include <list>
#include <string>
#include "../common/si_check.h"
using std::string;

int processInstrumentationRequests(char *fname);
bool instrumentEntity(const string &function_name);
bool processFileForInstrumentation(const string &file_name);


int main(int argc, char **argv) {
  if (argc != 3) { std::cerr << "usage: driver select.tau case.expect\n"; return 2; }
  processInstrumentationRequests(argv[1]);
  return si_check(argv[2], instrumentEntity, processFileForInstrumentation);
}
