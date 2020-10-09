
#include "Tau_regular_map.h"

using namespace std;

map<size_t,dummy*> TauRegularMap::_sharedMap;
mutex TauRegularMap::_sharedAccess;

