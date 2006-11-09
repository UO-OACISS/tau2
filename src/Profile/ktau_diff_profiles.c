/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.acl.lanl.gov/tau		           **
*****************************************************************************
**    Copyright 1999  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: ktau_diff_profiles.c				  **
**	Description 	: 			 			  **
**	Author		: Aroon Nataraj					  **
**	Contact		: anataraj@cs.uoregon.edu		 	  **
**	Flags		: Compile with				          **
**			  -DTAUKTAU or -DTAUKTAU_MERGE			  **
**	Documentation	: 						  **
***************************************************************************/

#include <stdlib.h>
#ifndef CONFIG_KTAU_MERGE 
#define CONFIG_KTAU_MERGE

#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <linux/unistd.h>

#define DEBUG                   0

#include "Profile/ktau_proc_interface.h"
//#include <ktau_proc_interface.h>
//#include <ktau_proc_map.h>

#ifdef __cplusplus
extern "C" {
#endif



int compar_ktau_output(const void* parg1, const void* parg2) {
	ktau_output* ktau_out1 = (ktau_output*)parg1;
	ktau_output* ktau_out2 = (ktau_output*)parg2;

	if(ktau_out1->pid < ktau_out2->pid) {
		return -1;
	} else if(ktau_out1->pid == ktau_out2->pid) {
		return 0;
	} else {
		return 1;
	}
}


int ktau_diff_oent(o_ent* poentlist1, int size1, o_ent* poentlist2, int size2, o_ent** outoentlist) {
	int outno = 0, restsz = 0;
	o_ent *po1, *po2, *pout, *rest;

	if(size1 >= size2) {
		outno = size1;
	} else {
		outno = size2;
	}

	//lists are already sorted
	//allocate higher size (later o_ent will include the earlier indices)
	
	(*outoentlist) = (o_ent*)calloc(sizeof(o_ent), outno);

	pout = (*outoentlist);
	po1 = poentlist1;
	po2 = poentlist2;

	while((size1 > 0) && (size2 > 0)) {
		if(po1->index == po2->index) {
			
			pout->index = po1->index;

			pout->entry.addr = po1->entry.addr;
			pout->entry.type = po1->entry.type;

			pout->entry.data.timer.count = po2->entry.data.timer.count - po1->entry.data.timer.count;
			pout->entry.data.timer.incl = po2->entry.data.timer.incl - po1->entry.data.timer.incl;
			pout->entry.data.timer.excl = po2->entry.data.timer.excl - po1->entry.data.timer.excl;

			po1++;
			size1--;
			
			po2++;
			size2--;

		} else if(po1->index < po2->index) {
			
			pout->index = po1->index;

			pout->entry.addr = po1->entry.addr;
			pout->entry.type = po1->entry.type;

			pout->entry.data.timer.count = po1->entry.data.timer.count;
			pout->entry.data.timer.incl = po1->entry.data.timer.incl;
			pout->entry.data.timer.excl = po1->entry.data.timer.excl;

			po1++;
			size1--;

		} else {

			pout->index = po2->index;

			pout->entry.addr = po2->entry.addr;
			pout->entry.type = po2->entry.type;

			pout->entry.data.timer.count = po2->entry.data.timer.count;
			pout->entry.data.timer.incl = po2->entry.data.timer.incl;
			pout->entry.data.timer.excl = po2->entry.data.timer.excl;

			po2++;
			size2--;
		}

		pout++;
	}

	if(size1 > 0) {
		rest = po1;
		restsz = size1;
	} else if(size2 > 0) {
		rest = po2;
		restsz = size2;
	}
	
	while(restsz > 0) {
		pout->index = rest->index;

		pout->entry.addr = rest->entry.addr;
		pout->entry.type = rest->entry.type;

		pout->entry.data.timer.count = rest->entry.data.timer.count;
		pout->entry.data.timer.incl = rest->entry.data.timer.incl;
		pout->entry.data.timer.excl = rest->entry.data.timer.excl;

		rest++;
		restsz--;
		pout++;
	}

	return outno;
}


int ktau_diff_profiles(ktau_output* plist1, int size1, ktau_output* plist2, int size2, ktau_output** outlist) {
	
	ktau_output *pl1, *pl2, *plout, *rest;
	int outno = 0, no1 = 0, no2 = 0, restsz = 0;

	//1st sort list1
	qsort(plist1, size1, sizeof(ktau_output), compar_ktau_output);

	//2nd sort list2
	qsort(plist2, size2, sizeof(ktau_output), compar_ktau_output);
	
	//next find out number of elements required in out array
	pl1 = plist1;
	pl2 = plist2;
	no1 = size1;
	no2 = size2;

	while((no1 > 0) && (no2 > 0)) {
		
		if(pl1->pid == pl2->pid) {
			pl1++;
			no1--;

			pl2++;
			no2--;

			outno++;
		} else if(pl1->pid < pl2->pid) {
			pl1++;
			no1--;

			//in this case an old process went away when we didnt look - so just skip it
		} else {
			pl2++;
			no2--;

			outno++;
		}

	}
	

	if(no1 > 0) {
		//many processes went away wehn we didnt look? do we skip them?
		//outno += no1;
	} else if(no2 > 0) {
		outno += no2;
	}

	//so outno is the total number of out entries - so allocate
	*outlist = (ktau_output*)calloc(outno,sizeof(ktau_output));

	//next do the diffing
	pl1 = plist1;
	pl2 = plist2;
	no1 = size1;
	no2 = size2;

	plout = (*outlist);

	while((no1 > 0) && (no2 > 0)) {
		
		if(pl1->pid == pl2->pid) {

			//then must do the subtraction
			plout->size = ktau_diff_oent(pl1->ent_lst, pl1->size, pl2->ent_lst, pl2->size, &(plout->ent_lst));
			plout->pid = pl1->pid;

			pl1++;
			no1--;

			pl2++;
			no2--;

			plout++;

		} else if(pl1->pid < pl2->pid) {

			//this means we saw a process - then it vanished - we must not include it
			//its times would have been aggragted into its parent

			//skip it
			// DONT:(*plout) = (*pl1);
			
			pl1++;
			no1--;
		} else {

			//then just transfer pl2's entries directly
			(*plout) = (*pl2);

			pl2++;
			no2--;

			plout++;
		}
	
	}
	
	//now one list may be non-empty
	if(no1 > 0) {
		//but the earlier list is non-empty - then we didnt see the process deaths
		//their numbers will be added to their parents - forget about them. We assume that the pid-range has not wrapped?
		//rest = pl1;
		//restsz = no1;
	} else if (no2 > 0) {
		rest = pl2;
		restsz = no2;
	}
	
	while(restsz > 0) {
		(*plout) = (*rest);
		plout++;
		rest++;
		restsz--;
	}

	return outno;
}	

#ifdef __cplusplus
}
#endif
