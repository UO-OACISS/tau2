
#include "Tau_caching_map.h"

using namespace std;

map<size_t,dummy*> TauCachingMap::_sharedMap;
mutex TauCachingMap::_sharedAccess;

