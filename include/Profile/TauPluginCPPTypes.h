#include <map>
#include <set>
#include <string>

class PluginKey {
   public:
   int plugin_event;
   size_t specific_event_hash;

   PluginKey(int _plugin_event, size_t _specific_event_hash) {
     plugin_event = _plugin_event;
     specific_event_hash = _specific_event_hash;
   }
   
   bool operator< (const PluginKey &rhs) const {
     if(plugin_event != rhs.plugin_event) return plugin_event < rhs.plugin_event;
     else return specific_event_hash < rhs.specific_event_hash;
   }
   
   ~PluginKey() { }
};

extern std::map<PluginKey, std::set<unsigned int> > plugins_for_named_specific_event;
