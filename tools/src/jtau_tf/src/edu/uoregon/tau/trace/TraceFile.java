/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

/*
 * This file (Ttf_file.java) contains several  classes used by the TAU_tf library
 */

package edu.uoregon.tau.trace;

import java.util.Map;

/* 
 * Event stores the essential data of a TAU event, event 64 represents
*/
class Event {
	int ev;    /* -- event id   int32     -- */
	char nid;   /* -- node id    uint16     -- */
	char tid;   /* -- thread id  uint16     -- */
	long par;   /* -- event parameter int64 -- */
	long ti;    /* -- time [us]?   uint64   -- */
}

/* As event, but for 64 bit platforms */
/*class Event64 {
	long ev;    // -- event id int64       -- //
	char nid;   // -- node id   uint16      -- //
	char tid;   // -- thread id uint16      -- //
	int padding; //  space wasted for 8-byte aligning the next item x_uint32// 
	long par;   // -- event parameter -- x_int64//
	long ti;    // -- time [us]?   x_uint64   -- //
}*/


/* Stores the definition info for a TAU event */
class EventDescr {
	int  Eid; /* event id */
	String Group; /* state as in TAU_VIZ */
	String EventName; /* name as in "foo" */
	int  Tag; /* -7 for send etc. */
	String Param; /* param as in EntryExit */
}

/*Stores a pair of Objects*/
class Pair implements Comparable{
	private Object first;
	private Object second;
	public Pair(Object first, Object second){
		this.first=first;
		this.second=second;
	}
	public Pair(Pair pair){
		this.first=pair.first();
		this.second=pair.second();
	}
	public Pair(){
		this.first=null;
		this.second=null;
	}
	public Object first(){
		return this.first;
	}
	public Object second(){
		return this.second;
	}
	public void setFirst(Object first){
		this.first=first;
	}
	public void setSecond(Object second){
		this.second=second;
	}
	public String toString(){
		return "("+first.toString()+", "+second.toString()+")";
	}
	public boolean equals(Object target){
		if(this == target)return true;
		if(!(target instanceof Pair))return false;
		Pair targ = (Pair)target;
		return(this.first.equals(targ.first)&&this.second.equals(targ.second));
	}
	public int compareTo(Object arg) {
		if(!(arg instanceof Pair))return -1;
		Pair parg = (Pair)arg;
		if(this.equals(parg))
			return 0;
		else return 1;
	}
}

/*This holds all of the structures and data relevant to input/output of a trace file*/
public class TraceFile {
	String EdfFile;//Name of the edf file being read or written
	Map NidTidMap;//=new HashMap();
	Map EventIdMap;//=new HashMap();;
	Map GroupIdMap;//=new HashMap();
	long FirstTimestamp;
	boolean ClkInitialized;
	boolean subtractFirstTimestamp;
	boolean nonBlocking;
	int format;    // The format of the Events
	int eventSize; // size of the corresponding Event class in bytes
	// For Trace Writing
	Event[] traceBuffer;
	int tracePosition;
	boolean needsEdfFlush;
	Map groupNameMap;
	boolean initialized;
	long lastTimestamp;
}
