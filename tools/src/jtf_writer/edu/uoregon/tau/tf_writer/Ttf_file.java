/*****************************************************************************
 **			TAU Portable Profiling Package			    **
 **			http://www.cs.uoregon.edu/research/paracomp/tau     **
 *****************************************************************************
 **    Copyright 2006							    **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 **    Research Center Juelich, Germany                                     **
 ****************************************************************************/
/*****************************************************************************
 **	File 		: Ttf_file.java			                    **
 **	Description 	: TAU trace format writer object       		    **
 **	Author		: Wyatt Spear            			    **
 **	Contact		: wspear@cs.uoregon.edu 	                    **
 ****************************************************************************/

package edu.uoregon.tau.tf_writer;

import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.Vector;

public class Ttf_file {
		DataOutputStream Fid;
		String EdfFile;
		Vector NidTidMap;//=new HashMap();
		TreeMap EventIdMap;//=new HashMap();;
		HashMap GroupIdMap;//=new HashMap();
		//boolean ClkInitialized;
		double FirstTimestamp;
		boolean subtractFirstTimestamp;
		boolean nonBlocking;
		int format;    // see above
		int eventSize; // sizeof() the corresponding format struct

		// For Trace Writing */
		//Event[] traceBuffer; //EVENT
		ArrayList traceBuffer;
		int tracePosition;
		boolean needsEdfFlush;
		HashMap groupNameMap;//=new HashMap();
		boolean initialized;
		long lastTimestamp;
}
