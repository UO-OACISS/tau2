/************************************************************************
************************************************************************/

/************************************************************************
*	readVTF.h															*
*	Author: Ryan K. Harris & Wyatt Spear								*
*																		*
*	Definitions for	the 'readVTF' module: a series of user record		*
*	handlers to replace the default Vampir record handlers in order		*
*	to manipulate selected trace file (VTF) records in the desired 		*
*	fashion.															*
*************************************************************************
*	class readVTF
*
*	Declared prototypes (private methods):
*		MyDowntoHandler -- handle subroutine call events: entering
*								functions.
*		MyUpfromHandler -- handle return from subroutine events: this
*								one takes the ID of subroutine being
*								exited.
*		MyUptoHandler -- handle return from subroutine events: this
*								one takes the ID of subroutine one
*								level up from the one we're leaving.
*		MyExchangeHandler -- handle enter/exit subroutine events: this
*								is an older method which handles
*								all of the above events in older VTF
*								formats.
*		MyDefactHandler -- handle activity events: gets activity/group
*								ID's.
*		MyDefstateHandler -- handle state events: get unique function
*								ID's.
*		MyDefthreadnumsHandler -- handle threadnumarray if any, allows
*								me to determine if the trace is threaded.
*		MyDefclkperiodHandler -- provides # ticks per sec on given
*									machine. In absence of Deftimeoffset,
*									it is used like:
*									time = timestamp * clkperiod.
*		MyDeftimeoffsetHandler -- if this exists, used in calculation of
*									timestamps:
*									time = (timestamp * clkperiod)
*										+ timeoffset + date.
*		MyUnrecognizableHandler -- handles records that VTF doesn't
*									recognize.
		MyDefsampHandler -- Handles sample definitions, behavior
							depending on the sample type
		MySampHandler -- Handles sample records associated with functions
							or events
*
*	THE FOLLOWING TWO ARE OUTDATED AND INCLUDED FOR PURPOSES OF
*	SUPPORTING OLD FILE TYPES
*		MyExchange_ObsolHandler -- handles old-style exchange records.
*		MyDefstate_ObsolHandler -- handles old-style exhcange events.
*
*
*	Public methods:
*		readfile -- the main method which reads in a VTF and calls
*					all of the required sub-classes to operate on
*					it's contents.
************************************************************************/
#ifndef __READVTF_H
#define __READVTF_H

#include <iostream>
#include <string>
using namespace std;

#include <string.h>
#include "binarytree.h"
#include "node.h"
#include "nodeC.h"
#include "vtf3.h"

class readVTF
{
private:
	/* Define communication control block type */
	typedef struct
	{
		void *fcbin;	// The file to be read
	}
	ccb_t;

	/* Declare user record handler prototypes here. */
	VTF3_DCL_DOWNTO(static, MyDowntoHandler);
	VTF3_DCL_UPFROM(static, MyUpfromHandler);
	VTF3_DCL_UPTO(static, MyUptoHandler);
	VTF3_DCL_EXCHANGE(static, MyExchangeHandler);
	VTF3_DCL_DEFACT(static, MyDefactHandler);
	VTF3_DCL_DEFSTATE(static, MyDefstateHandler);
	VTF3_DCL_DEFTHREADNUMS(static, MyDefthreadnumsHandler);
	VTF3_DCL_DEFCLKPERIOD(static, MyDefclkperiodHandler);
	VTF3_DCL_DEFTIMEOFFSET(static, MyDeftimeoffsetHandler);

	/* UNRECOGNIZABLE is for the lines in VTF(s) which the
		parser is unable to understand, it describes how
		to handle those cases. */
	VTF3_DCL_UNRECOGNIZABLE(static, MyUnrecognizableHandler);

	/*New Wyatt-coded handler(s)*/
	VTF3_DCL_DEFSAMP(static, MyDefsampHandler);
	VTF3_DCL_SAMP(static, MySampHandler);
	VTF3_DCL_DEFCPUGRP(static, MyCPUGrpHandler);
	VTF3_DCL_DEFCPUNAME(static, MyCPUNameHandler);

	int DowntoBatch(void *ccbvoid);
	int UpfromBatch(void *ccbvoid);
	int UptoBatch(void *ccbvoid);


	/* Now private member variables */
	string filename;			// Holds the filename to be read
	string destPath;			// Dir path for profiles to be placed
	int fileformat;				// Holds format #
	int  numrec_types;			// Will be filled with # of types
	int *recordtypes;			// Handle to array of recordtypes
	ccb_t ccb;					/* Define a ccb_t(struct from the header)*/
	VTF3_handler_t *handlers;	// Array of handlers
	void **firsthandlerargs;	// First args passed to handlers
	int totalbytesread;
	int substituteupfrom;		// Int used as fourth arg to OpenFileInput.
	int bytesread;
	int threadNumArrayDim;
	int * threadNumArray;
	double clockPeriod;
	double timeOffset;
	int nextToken;
	double startTime;
	double endTime;

	int maxstate;				//The highest state encountered
	int usermets; 				//The number of user defined metrics
	int globmets;				//The global metrics
	unsigned long long * sampholder;		//Holds transition sample data
	string * sampnames;			//Holds the names of the samples (global metrics)
	int lastcall;   //The last exchange handler called (will run this exchange operation when all samples are collected)
	int sampindex;
	int sampstatetoken;
	unsigned int sampcpupid;
	int sampscltoken;
	double samptime;
	int ngroups;
	int * threadlist;


	/* we need to define binary tree's to work with */
	binarytree * cpu_tree;				//'cpu' nodes in this VTF
	binarytree * functionLookUp;		//For looking up function properties
	binarytree * activityLookUp;		//Looking up activities/groups
	//binarytree * metricLookUp;			//For looking up user defined metrics


public:

	/* Constructor takes one parameter:
		filename = char array containing the filename to be
		read by the class. We could also accept the parameter
		in the form 'const char filename[]. That is just a note
		for me(the programmer). */
	readVTF(string * n_filename, double n_startTime, double n_endTime)
	{
		/* Initialize all vars */
		filename = *n_filename;
		destPath = "./";
		fileformat = -1;
		numrec_types = -1;
		recordtypes = 0;
		ccb.fcbin = 0;
		handlers = 0;
		firsthandlerargs = 0;
		totalbytesread = -1;
		substituteupfrom = -1;
		bytesread = -1;
		threadNumArrayDim = 0;
		clockPeriod = 0;
		timeOffset = 0;
		nextToken = 100000;
		startTime = n_startTime;
		endTime = n_endTime;

		maxstate = 0;
		usermets = 0;
		globmets = 0;
		lastcall = -1;
		sampindex = -1;
		sampstatetoken = -1;
		sampcpupid = 0;
		sampscltoken = 0;
		samptime = 0;
		sampholder = 0;
		sampnames = 0;
		ngroups = 0;
		threadlist = 0;

		/* initialize the tree */
		cpu_tree = new binarytree();
		functionLookUp = new binarytree();
		activityLookUp = new binarytree();
		//metricLookUp = new binarytree();

	}//constructor

	/* Second constructor takes a second arg:
		string specifying dir path for profiles. */
	readVTF(string * n_filename, string * n_destPath, double n_startTime, double n_endTime)
	{
		/* Initialize all vars */
		filename = *n_filename;
		destPath = *n_destPath;
		fileformat = -1;
		numrec_types = -1;
		recordtypes = 0;
		ccb.fcbin = 0;
		handlers = 0;
		firsthandlerargs = 0;
		totalbytesread = -1;
		substituteupfrom = -1;
		bytesread = -1;
		threadNumArrayDim = 0;
		clockPeriod = 0;
		timeOffset = 0;
		nextToken = 100000;
		startTime = n_startTime;
		endTime = n_endTime;
		
		maxstate = 0;
		usermets = 0;
		globmets = 0;
		lastcall = -1;
		sampindex = -1;
		sampstatetoken = -1;
		sampcpupid = 0;
		sampscltoken = 0;
		samptime = 0;
		sampholder = 0;
		sampnames = 0;
		ngroups = 0;

		/* initialize the tree */
		cpu_tree = new binarytree();
		functionLookUp = new binarytree();
		activityLookUp = new binarytree();
		//metricLookUp = new binarytree();

	}//constructor

	/* Destructor, so we make sure to get rid of our trees */
	~readVTF()
	{
		delete cpu_tree;
		delete functionLookUp;
		delete activityLookUp;
		//delete metricLookUp;
	}//destructor

	void readfile(void);

}
;//readVTF


#endif
