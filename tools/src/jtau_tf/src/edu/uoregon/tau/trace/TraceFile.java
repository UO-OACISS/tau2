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
	private int ev;    /* -- event id   int32     -- */
	private char nid;   /* -- node id    uint16     -- */
	private char tid;   /* -- thread id  uint16     -- */
	private long par;   /* -- event parameter int64 -- */
	private long ti;    /* -- time [us]?   uint64   -- */
	
	Event(int ev, char nid, char tid, long par, long ti){
		this.ev=ev;
		this.nid=nid;
		this.tid=tid;
		this.par=par;
		this.ti=ti;
	}
	
	Event(){}
	
	public int getEventID() {
		return ev;
	}
	public void setEventID(int ev) {
		this.ev = ev;
	}
	public char getNodeID() {
		return nid;
	}
	public void setNodeID(char nid) {
		this.nid = nid;
	}
	public long getParameter() {
		return par;
	}
	public void setParameter(long par) {
		this.par = par;
	}
	public long getTime() {
		return ti;
	}
	public void setTime(long ti) {
		this.ti = ti;
	}
	public char getThreadID() {
		return tid;
	}
	public void setThreadID(char tid) {
		this.tid = tid;
	}
	public int getNidTid(){
		return (nid<<16)+tid;
	}
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
	private int  Eid; /* event id */
	private String Group; /* state as in TAU_VIZ */
	private String EventName; /* name as in "foo" */
	private long Tag; /* -7 for send etc. */
	private String Param; /* param as in EntryExit */
	EventDescr(){}
	EventDescr(int eid, String group, String eventName, long tag, String param) {
		Eid = eid;
		Group = group;
		EventName = eventName;
		Tag = tag;
		Param = param;
	}
	public int getEventId() {
		return Eid;
	}
	public void setEventId(int eid) {
		Eid = eid;
	}
	public String getEventName() {
		return EventName;
	}
	public void setEventName(String eventName) {
		EventName = eventName;
	}
	public String getGroup() {
		return Group;
	}
	public void setGroup(String group) {
		Group = group;
	}
	public String getParameter() {
		return Param;
	}
	public void setParameter(String param) {
		Param = param;
	}
	public long getTag() {
		return Tag;
	}
	public void setTag(int tag) {
		Tag = tag;
	}
}

/*Stores a pair of Objects*/
/*
class CharPair implements Comparable{
	private char first;
	private char second;
	public CharPair(char first, char second){
		this.first=first;
		this.second=second;
	}
	public CharPair(CharPair pair){
		this.first=pair.first();
		this.second=pair.second();
	}
	public CharPair(){
		this.first=0;
		this.second=0;
	}
	public char first(){
		return this.first;
	}
	public char second(){
		return this.second;
	}
	public void setFirst(char first){
		this.first=first;
	}
	public void setSecond(char second){
		this.second=second;
	}
	public String toString(){
		return "("+first+", "+second+")";
	}
	public boolean equals(Object target){
		if(this == target)return true;
		if(!(target instanceof CharPair))return false;
		CharPair targ = (CharPair)target;
		return(this.first==(targ.first)&&this.second==(targ.second));
	}
	public int compareTo(Object arg) {
		if(!(arg instanceof CharPair))return -1;
		CharPair parg = (CharPair)arg;
		if(this.equals(parg))
			return 0;
		else return 1;
	}
}*/

/*This holds all of the structures and data relevant to input/output of a trace file*/
public class TraceFile {
	String TrcFile;
	int node;
	int context;
	int thread;
	String EdfFile;//Name of the edf file being read or written
	//Map NidTidMap;//=new HashMap();
	Map EventIdMap;//=new HashMap();;
	Map GroupIdMap;//=new HashMap();
	//long FirstTimestamp;
	//boolean ClkInitialized;
	//boolean subtractFirstTimestamp;
	//boolean nonBlocking;
	//int format;    // The format of the Events
	//int eventSize; // size of the corresponding Event class in bytes
	// For Trace Writing
	//Event[] traceBuffer;
	//int tracePosition;
	//boolean needsEdfFlush;
	//Map groupNameMap;
	//boolean initialized;
	//long lastTimestamp;
}
