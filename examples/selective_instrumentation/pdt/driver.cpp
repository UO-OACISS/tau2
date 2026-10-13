// Evaluates the common SI test cases against utils/tau_selective.cpp (tau_instrumentor / tau_selectfile).
#include <list>
#include <string>
#include "../common/si_check.h"
using std::string;

int processInstrumentationRequests(char *fname);
bool instrumentEntity(const string &function_name);
bool processFileForInstrumentation(const string &file_name);

// tau_selective.cpp calls into tau_instrument.cpp for BEGIN_INSTRUMENT_SECTION; not under test here.
void parseInstrumentationCommand(char *, int) {}
void printInstrumentList(void) {}

int main(int argc, char **argv) {
  if (argc != 3) { std::cerr << "usage: driver select.tau case.expect\n"; return 2; }
  processInstrumentationRequests(argv[1]);
  return si_check(argv[2], instrumentEntity, processFileForInstrumentation);
}
