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

/* TODO: This class is used as a subtiture for
 * plugins_for_named_specific_event for ompt events.
 * We encountered an issue with the C++ runtime destroying
 * plugins_for_named_specific_event before the end of the OpenMP
 * runtime with Intel and LLVM. Hence why this is not using anything
 * that would be destroyed by the C++ runtime in order to fix the
 * issues.
 *
 * For Intel the issue was happening when a plugins asked for the
 * ompt_finalize callback. For LLVM it also happened with other
 * callbacks happening after the end of the application but before
 * the ompt_finalize, and the ompt_finalize. These bugs were
 * especially present when OpenMP tasks were used.
 *
 * Once this is not needed anymore everything below this comment can
 * be removed, ant the "Tau_util_invoke_callbacks_" in TauUtil.cpp
 * for ompt events can be changed back to be the same as the other
 * ones. And also removing the specific cases in
 * "Tau_enable/disable_(all)_plugin_for_for_specific_event" inside
 * TauCAPI.cpp
 *
 * The performance here could probably be increased as this was
 * quickly put together as a temporary fix until using C++ global
 * variables are not destroyed before we stop needing them. */
class OmptPluginsVect {
   private:
   unsigned int *plugins;
   unsigned int nb_plugins;
   unsigned int max_plugins;
   bool is_ompt_event;

   public:
   OmptPluginsVect() {
     plugins = NULL;
     nb_plugins = 0;
     max_plugins = 0;
     is_ompt_event = false;
   }

   void insert(unsigned int id) {
     if(nb_plugins >= max_plugins)
     {
       max_plugins = max_plugins ? max_plugins * 2 : 5;
       plugins = (unsigned int*)realloc(plugins, max_plugins * sizeof(unsigned int));
     }

     plugins[nb_plugins] = id;
     nb_plugins++;
   }

   void clear() {
     nb_plugins = 0;
   }

   void erase(unsigned int id) {
    unsigned int i = 0;
    unsigned int found = 0;
    while(!found && i < nb_plugins)
    {
      if(plugins[i] == id)
      {
        found = 1;
        while(i < nb_plugins - 1)
        {
          plugins[i] = plugins[i+1];
          ++i;
        }
        nb_plugins--;
      }
    }
   }

   unsigned int size() {
     return nb_plugins;
   }

   void flag_as_ompt() {
     is_ompt_event = true;
   }

   bool is_ompt() {
     return is_ompt_event;
   }

   void destroy() {
     if(plugins) free(plugins);
     plugins = NULL;
     nb_plugins = 0;
     max_plugins = 0;
     is_ompt_event = false;
   }

   unsigned int operator [](int i) const {return plugins[i];}
   unsigned int & operator [](int i) {return plugins[i];}

   ~OmptPluginsVect() { }
};

extern OmptPluginsVect plugins_for_ompt_event[NB_TAU_PLUGIN_EVENTS];
