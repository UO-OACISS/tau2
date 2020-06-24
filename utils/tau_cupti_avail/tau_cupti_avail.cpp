#include <Profile/CuptiLayer.h>
#include <unistd.h>
#include <sstream>
#include <string>
#include <vector>

// CrayCC is stupid and doesn't support weak symbols
void * pomp_rd_table;
int POMP_MAX_ID = 0;

int main(int argc, char **argv)
{
    int c;
    bool listCounters = true, checkCounters = false, listMetrics = false;
    char* counter_list;

    while ((c = getopt(argc, argv, "mhc:")) != -1) {
        switch (c) {
        case 'm':
            checkCounters = false;
            listCounters = false;
            listMetrics = true;
            break;
        case 'h':
            //printUsage()
            break;
        case 'c':
            checkCounters = true;
            listCounters = false;
            listMetrics = false;
            counter_list = optarg;
            break;
        case '?':
            if (optopt == 'c') {
                fprintf(stderr, "Error: Option -c require an argument.\n");
                //printUsage();
                exit(1);
            } else {
                fprintf(stderr, "Error: could not parse arguments.\n");
                //printUsage();
                exit(1);
            }
        default:
            exit(1);
        }
    }

    if (listCounters) {
	Tau_CuptiLayer_Initialize_Map(1);
        CuptiCounterEvent::printHeader();
        for (counter_map_it it = Tau_CuptiLayer_Counter_Map().begin(); it != Tau_CuptiLayer_Counter_Map().end(); it++) {
            it->second->print();
        }
    }

    if (listMetrics) {
        Tau_CuptiLayer_Initialize_Map(1);
        CuptiMetric::printHeader();
        for (metric_map_it it = Tau_CuptiLayer_Metric_Map().begin(); it != Tau_CuptiLayer_Metric_Map().end(); it++) {
            it->second->print();
        }
    }

    if (checkCounters) {
	Tau_CuptiLayer_Initialize_Map(1);
        if (counter_list == NULL) {
            fprintf(stderr, "ERROR: counter list empty.\n");
            exit(1);
        }
        printf("conter list arg is: %s.\n", counter_list);
        //split counter list by ':' delimiter.
        std::string counter_list_str = std::string(counter_list);

        std::stringstream iss(counter_list_str);
        std::string item;
        std::vector<std::string> tags;
        while (std::getline(iss, item, ':')) {
            tags.push_back(item);
        }

        std::vector<std::string> tags_added;
        std::vector<std::string> tags_failed;
        std::vector<CuptiCounterEvent*> counters_added;

        for (std::vector<std::string>::iterator it = tags.begin(); it != tags.end(); it++) {
            //printf("size of available counters: %d.\n", Tau_CuptiLayer_Counter_Map.size());

            if (Tau_CuptiLayer_Counter_Map().count(*it) > 0) {
                CuptiCounterEvent *ev = Tau_CuptiLayer_Counter_Map().find(*it)->second;
                //ev->print();
                tags_added.push_back(*it);
                counters_added.push_back(ev);
            } else {
                tags_failed.push_back(*it);
            }
        }

        std::cout << "Counters successfully set:" << std::endl;
        for (std::vector<std::string>::iterator it = tags_added.begin(); it != tags_added.end(); it++) {
            std::cout << "\t * " << *it << std::endl;
        }
        std::cout << "Failed to set these counters:" << std::endl;
        for (std::vector<std::string>::iterator it = tags_failed.begin(); it != tags_failed.end(); it++) {
            std::cout << "\t * " << *it << std::endl;
        }

    }

    return 0;
}
