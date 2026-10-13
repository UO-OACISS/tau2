// Shared by the C++ test drivers: replays a .expect file against consumer-provided predicates.
// Line format:  func: <name> => yes|no   file: <path> => yes|no   (other kinds, e.g. loop:, are ignored)
#ifndef SI_CHECK_H
#define SI_CHECK_H
#include <fstream>
#include <iostream>
#include <string>

typedef bool (*si_pred)(const std::string &);

static int si_check(const char *expectfile, si_pred func, si_pred file) {
  std::ifstream exp(expectfile);
  std::string line;
  int bad = 0;
  while (std::getline(exp, line)) {
    size_t colon = line.find(": "), arrow = line.rfind(" => ");
    if (colon == std::string::npos || arrow == std::string::npos) continue;
    std::string kind = line.substr(0, colon);
    std::string name = line.substr(colon + 2, arrow - colon - 2);
    bool want = line.substr(arrow + 4) == "yes", got;
    if (kind == "func") got = func(name);
    else if (kind == "file") got = file(name);
    else continue;
    if (got != want) {
      std::cout << "  FAIL " << kind << " '" << name << "': expected " << (want ? "yes" : "no") << std::endl;
      ++bad;
    }
  }
  return bad ? 1 : 0;
}
#endif
