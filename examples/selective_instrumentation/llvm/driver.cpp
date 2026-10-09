// Evaluates the common SI test cases against the selection logic of the TAU LLVM plugin.
// Usage: driver <select.tau> <case.expect>
#include "../../../plugins/llvm/src/Instrument.cpp"
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 3) { std::cerr << "usage: driver select.tau case.expect\n"; return 2; }
    Tools t;
    t.verbose = false;
    std::ifstream sel(argv[1]);
    if (!sel) { std::cerr << "cannot open " << argv[1] << "\n"; return 2; }
    t.loadFunctionsFromFile(sel);
    std::ifstream exp(argv[2]);
    std::string line;
    int bad = 0;
    while (std::getline(exp, line)) {
        size_t colon = line.find(": "), arrow = line.rfind(" => ");
        if (colon == std::string::npos || arrow == std::string::npos) continue;
        std::string kind = line.substr(0, colon);
        std::string name = line.substr(colon + 2, arrow - colon - 2);
        bool want = line.substr(arrow + 4) == "yes", got;
        if (kind == "func") got = t.includeFunctionName(name);
        else if (kind == "file") got = t.includeFile(name);
        else if (kind == "loop") got = t.includeLoop(name);
        else continue;
        if (got != want) {
            std::cout << "  FAIL " << kind << " '" << name << "': expected " << (want ? "yes" : "no") << "\n";
            ++bad;
        }
    }
    return bad ? 1 : 0;
}
