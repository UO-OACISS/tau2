// Evaluates the common SI test cases against utils/tau_selective.dyn.cpp (the SI implementation used by tau_run).
#include <list>
#include <string>
#include <vector>
#include "../common/si_check.h"
#include "tau_instrument.dyn.h"
using std::string;

int processInstrumentationRequests(char *fname, vector<tauInstrument *> &instrumentList);
bool instrumentEntity(const string &function_name);
bool processFileForInstrumentation(const string &file_name);

// BEGIN_INSTRUMENT_SECTION entries are handled by tau_instrument.dyn.cpp; not under test here.
void parseDyninstInstrumentationCommand(char *, int, vector<tauInstrument *> &) {}
void printInstrumentList(void) {}

int main(int argc, char **argv) {
  if (argc != 3) { std::cerr << "usage: driver select.tau case.expect\n"; return 2; }
  vector<tauInstrument *> instrumentList;
  processInstrumentationRequests(argv[1], instrumentList);
  return si_check(argv[2], instrumentEntity, processFileForInstrumentation);
}
