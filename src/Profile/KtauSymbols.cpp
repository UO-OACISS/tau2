/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: KtauSymbols.h					  **
**	Description 	: TAU Kernel Profiling Interface		  **
**	Author		: Aroon Nataraj					  **
**			: Suravee Suthikulpanit				  **
**	Contact		: {suravee,anataraj}@cs.uoregon.edu               **
**	Flags		: Compile with				          **
**			  -DTAU_KTAU to enable KTAU	                  **
**	Documentation	: 					          **
***************************************************************************/

#ifdef TAUKTAU

#include <Profile/KtauSymbols.h>
#include <Profile/KtauProfiler.h>

#include <string>
#include <iostream>
#include <fstream>

#define LINE_SIZE               1024

using namespace std;

#define DEBUG 0

#define ktau_str2ll (unsigned int)strtoul


KtauSymbols::KtauSymbols(const string& path):filepath(path) {
	//dummy call to KTauGetMHz() - so that this time is not included in any function's time!
	if(DEBUG) 
		printf("KtauSymbols::cons Calling KTauGetMHz() to store the TSC tick-rate.\n");
	KTauGetMHz();

	ReadKallsyms();
}

KtauSymbols::~KtauSymbols(){};


string& KtauSymbols::MapSym(unsigned int addr) {
	return table[addr];
}

int KtauSymbols::ReadKallsyms()
{
        char line[LINE_SIZE];
        char *ptr = line;
        char *addr;
        char *flag;
        char *func_name;
        unsigned int stext = 0;
        int found_stext = 0;

        /* Open and read file */
        ifstream fs_kallsyms (filepath.c_str(),ios::in);
        if(!fs_kallsyms.is_open()){
                cout << "Error opening file: " << filepath << "\n";
                return(-1);
        }

        /* Tokenize each line*/
        while(!fs_kallsyms.eof()){
                fs_kallsyms.getline(line,LINE_SIZE);
                ptr = line;
                addr = strtok(ptr," ");
                flag = strtok(NULL," ");
                func_name = strtok(NULL," ");

                if(!strcmp("T",flag) || !strcmp("t",flag)){
                        if(!found_stext && !strcmp("stext",func_name)){
				table[ktau_str2ll(addr,NULL,16)] = string(func_name);
                                //kallsyms_map[kallsyms_size].addr = ktau_str2ll(addr,NULL,16);
                                //strcpy(kallsyms_map[kallsyms_size++].name, func_name);
                                found_stext = 1;
                                if(DEBUG)printf("ReadKallsyms: address %llx , name %s \n",
                                                ktau_str2ll(addr,NULL,16),
                                                table[ktau_str2ll(addr,NULL,16)].c_str());
                        } else{
				table[ktau_str2ll(addr,NULL,16)] = string(func_name);
                                //kallsyms_map[kallsyms_size].addr = ktau_str2ll(addr,NULL,16);
                                //strcpy(kallsyms_map[kallsyms_size++].name, func_name);
                                if(DEBUG)printf("ReadKallsyms: address %llx , name %s \n",
                                                ktau_str2ll(addr,NULL,16),
                                                table[ktau_str2ll(addr,NULL,16)].c_str());
                        }
                }else if(!strcmp("_etext",func_name)){
			table[ktau_str2ll(addr,NULL,16)] = string(func_name);
                        //kallsyms_map[kallsyms_size].addr = ktau_str2ll(addr,NULL,16);
                        //strcpy(kallsyms_map[kallsyms_size++].name, func_name);
			if(DEBUG)printf("ReadKallsyms: address %llx , name %s \n",
					ktau_str2ll(addr,NULL,16),
					table[ktau_str2ll(addr,NULL,16)].c_str());
                        break;
                }
        }
        /* Close /proc/kallsyms */
        fs_kallsyms.close();

        return table.size();
}

#endif /* TAUKTAU */
