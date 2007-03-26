/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

package edu.uoregon.tau.tau_tf;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.TreeMap;
import java.util.Iterator;
import java.util.Set;

public class TAU_tf_writer {
	/* -- pcxx tracer events ------------------- */
	static int PCXX_EV_INIT = 60000;
	//int PCXX_EV_FLUSH_ENTER = 60001;
	//int PCXX_EV_FLUSH_EXIT  = 60002;
	static int PCXX_EV_CLOSE = 60003;
	//int PCXX_EV_INITM       = 60004;
	static int PCXX_EV_WALL_CLOCK = 60005;
	//int PCXX_EV_CONT_EVENT  = 60006;
	static int TAU_MESSAGE_SEND	= 60007;
	static int TAU_MESSAGE_RECV	= 60008;

	        /* -- the following two events are only the ----- */
	        /* -- base numbers, actually both represent ----- */
	        /* -- 64 events (60[1234]00 to 60[1234]64)  ----- */
	//int PCXX_WTIMER_CLEAR = 60199;
	//int PCXX_WTIMER_START = 60100;
	//int PCXX_WTIMER_STOP  = 60200;
	//int PCXX_UTIMER_CLEAR = 60399;
	//int PCXX_UTIMER_START = 60300;
	//int PCXX_UTIMER_STOP  = 60400;
	
	static int TAU_MAX_RECORDS = 64*1024;

	static void checkFlush(Ttf_file tFile) {
		// Write the current trace buffer to file if necessary.
		if (tFile.tracePosition >= TAU_MAX_RECORDS) {
			Ttf_FlushTrace(tFile);
		}
	}

	static int flushEdf(Ttf_file tFile) {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(tFile.EdfFile));

			int numEvents = tFile.EventIdMap.size();
			out.write(numEvents+" dynamic_trace_events\n");
			out.write("# FunctionId Group Tag \"Name Type\" Parameters\n");
			Iterator it = tFile.EventIdMap.values().iterator(); //entrySet().iterator();
			int id;
			Ttf_EventDescr eventDesc;
			while(it.hasNext())
			{
				eventDesc=(Ttf_EventDescr)it.next();
				id = eventDesc.Eid;
				out.write(id+" "+eventDesc.Group+" "+eventDesc.Tag+" \""+eventDesc.EventName+"\" "+eventDesc.Param+"\n");
			}
			out.close();
		} catch (IOException e) {
			System.out.println("Error opening edf file");
			return -1;
	    }

		tFile.needsEdfFlush = false;
		return 0;
	}

	public static Ttf_file Ttf_OpenFileForOutput( String name, String edf){
		FileOutputStream ostream;
		try {
			ostream = new FileOutputStream(name);

		BufferedOutputStream bw = new BufferedOutputStream(ostream);
		DataOutputStream p = new DataOutputStream(bw);

		Ttf_file tFile = new Ttf_file();
		tFile.traceBuffer = new Event[TAU_MAX_RECORDS];
		tFile.traceBuffer[0]=new Event();
		tFile.tracePosition = 1; // 0 will be the EV_INIT record
		tFile.initialized = false;

		tFile.Foid=p;


		BufferedWriter out = new BufferedWriter(new FileWriter(edf));
		out.close();

		/* make a copy of the EDF file name */
		tFile.EdfFile = edf;

		tFile.NidTidMap = new HashMap();

		/* Allocate space for maps */
		tFile.EventIdMap = new TreeMap();//Tree map for output ordered by id.

		tFile.GroupIdMap = new HashMap();

		tFile.groupNameMap = new HashMap();

		tFile.needsEdfFlush = true;

		/* initialize clock */
		//tFile.ClkInitialized = false;
		
		/* initialize the first timestamp for the trace */
		tFile.FirstTimestamp = 0;

		/* define some events */

		Ttf_EventDescr newEventDesc = new Ttf_EventDescr();

		newEventDesc.Eid = PCXX_EV_INIT;
		newEventDesc.Group = "TRACER";
		newEventDesc.EventName = "EV_INIT";
		newEventDesc.Tag = 0;
		newEventDesc.Param = "none";
		tFile.EventIdMap.put(new Integer(PCXX_EV_INIT),newEventDesc);//[PCXX_EV_INIT] = newEventDesc;
		newEventDesc = new Ttf_EventDescr();
		newEventDesc.Eid = PCXX_EV_CLOSE;
		newEventDesc.Group = "TRACER";
		newEventDesc.EventName = "FLUSH_CLOSE";
		newEventDesc.Tag = 0;
		newEventDesc.Param = "none";
		tFile.EventIdMap.put(new Integer(PCXX_EV_CLOSE),newEventDesc);//[PCXX_EV_CLOSE] = newEventDesc;
		newEventDesc = new Ttf_EventDescr();
		newEventDesc.Eid = PCXX_EV_WALL_CLOCK;
		newEventDesc.Group = "TRACER";
		newEventDesc.EventName = "WALL_CLOCK";
		newEventDesc.Tag = 0;
		newEventDesc.Param = "none";
		tFile.EventIdMap.put(new Integer(PCXX_EV_WALL_CLOCK),newEventDesc);//[PCXX_EV_WALL_CLOCK] = newEventDesc;
		newEventDesc = new Ttf_EventDescr();
		newEventDesc.Eid = TAU_MESSAGE_SEND;
		newEventDesc.Group = "TAU_MESSAGE";
		newEventDesc.EventName = "MESSAGE_SEND";
		newEventDesc.Tag = -7;
		newEventDesc.Param = "par";
		tFile.EventIdMap.put(new Integer(TAU_MESSAGE_SEND),newEventDesc);//[TAU_MESSAGE_SEND] = newEventDesc;
		newEventDesc = new Ttf_EventDescr();
		newEventDesc.Eid = TAU_MESSAGE_RECV;
		newEventDesc.Group = "TAU_MESSAGE";
		newEventDesc.EventName = "MESSAGE_RECV";
		newEventDesc.Tag = -8;
		newEventDesc.Param = "par";
		tFile.EventIdMap.put(new Integer(TAU_MESSAGE_RECV),newEventDesc);//[TAU_MESSAGE_RECV] = newEventDesc;

		/* return file handle */
		return tFile;
		} catch (SecurityException e1) {
			e1.printStackTrace();
			return null;
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
			return null;
	    } catch (IOException e1) {
	    	e1.printStackTrace();
	    	return null;
		}	
	}

	static int checkInitialized(Ttf_file tFile, int nodeToken, int threadToken, long time) {
		// Adds the initialization record
		if (!tFile.initialized) {
			int pos = 0;
			tFile.traceBuffer[pos].ev = PCXX_EV_INIT;
			tFile.traceBuffer[pos].nid = (char)nodeToken;
			tFile.traceBuffer[pos].tid = (char)threadToken;
			tFile.traceBuffer[pos].ti =  time;//(x_uint64)
			tFile.traceBuffer[pos].par = 3;
			tFile.initialized = true;
		}
		return 0;
	}

	public static int Ttf_DefThread(Ttf_file tFile, int nodeToken, int threadToken,String threadName) {
		Integer nid = new Integer(nodeToken);
		Integer tid = new Integer(threadToken);
		if(!tFile.NidTidMap.containsKey(new Pair(nid, tid)))
		{
			tFile.NidTidMap.put(new Pair(nid, tid), threadName);
		}
	    return 0;
	  }

	  // returns stateGroupToken
	public  static int Ttf_DefStateGroup(Ttf_file tFile, String stateGroupName, int stateGroupToken) {
		tFile.groupNameMap.put(new Integer(stateGroupToken),stateGroupName);
		return 0;
	}

	public  static int Ttf_DefState(Ttf_file tFile, int stateToken, String stateName, int stateGroupToken){
		Ttf_EventDescr newEventDesc = new Ttf_EventDescr();
		if(!tFile.groupNameMap.containsKey(new Integer(stateGroupToken))){
			//throw new Exception("Ttf_DefState: Have not seen"+stateGroupToken+"stateGroupToken before, please define it first\n");
			return -1;
		}

		newEventDesc.Eid = stateToken;
		newEventDesc.Group = (String)tFile.groupNameMap.get(new Integer(stateGroupToken));
		newEventDesc.EventName = stateName;
		newEventDesc.Tag = 0;
		newEventDesc.Param = "EntryExit";

		tFile.EventIdMap.put(new Integer(stateToken),newEventDesc);

		tFile.needsEdfFlush = true;

		return 0;
	}

	public static int Ttf_FlushTrace(Ttf_file tFile){
		int events = tFile.tracePosition;
		if(events>1)
			checkInitialized(tFile, tFile.traceBuffer[1].nid,tFile.traceBuffer[1].tid, tFile.traceBuffer[1].ti);
		else
			checkInitialized(tFile, 0,0,0);
		// reset trace position
		tFile.tracePosition = 0;
		// must write out edf file first
		if (tFile.needsEdfFlush) {
			if (flushEdf(tFile) != 0) {
				return -1;
			}
		}

		//Event evt;
		for(int i=0; i<events;i++)
		{
			try {
	    		//evt=tFile.traceBuffer[i];
				tFile.Foid.writeInt(tFile.traceBuffer[i].ev);
				tFile.Foid.writeChar(tFile.traceBuffer[i].nid);
				tFile.Foid.writeChar(tFile.traceBuffer[i].tid);
				tFile.Foid.writeLong(tFile.traceBuffer[i].par);
				tFile.Foid.writeLong(tFile.traceBuffer[i].ti);
	    	} catch (IOException e) {
				e.printStackTrace();
			}
	    }
	    return 0;
	  }

	public  static int Ttf_CloseOutputFile(Ttf_file tFile){
		Set keyset = tFile.NidTidMap.keySet();// .iterator();
		Iterator it=keyset.iterator();
		Pair nidtid;
		int first;
		int second;
		while(it.hasNext())
		{
			nidtid=(Pair)it.next();
			//System.out.println(nidtid.first().getClass().getSimpleName());
			first=(((Integer)nidtid.first()).intValue());
			second=(((Integer)nidtid.second()).intValue());
			checkFlush(tFile);
			int pos = tFile.tracePosition;
			Event evt = new Event();
			evt.ev = PCXX_EV_CLOSE;
			evt.nid = (char)first;
			evt.tid = (char)second;
			evt.ti = tFile.lastTimestamp;
			evt.par = 0;
			tFile.traceBuffer[pos]=(evt);//pos,
			tFile.tracePosition++;
	      
			pos = tFile.tracePosition;
			evt = new Event();
			evt.ev = PCXX_EV_WALL_CLOCK;
			evt.nid = (char)first;
			evt.tid = (char)second;
			evt.ti = tFile.lastTimestamp;
			evt.par = 0;
			tFile.traceBuffer[pos]=evt;//pos,
			tFile.tracePosition++;
		}
		Ttf_FlushTrace(tFile);
		try {
			tFile.Foid.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return 0;
	}

	static int enterExit(Ttf_file tFile, long time, 
			int nodeToken, int threadToken, 
			int stateToken, int parameter) {//time formerly (x_uint64)
		//Ttf_fileT *tFile = (Ttf_fileT*)file;

		checkFlush(tFile);
		int pos = tFile.tracePosition;
		//System.out.println(pos);
		Event evt = new Event();
		evt.ev = stateToken;
		evt.nid = (char)nodeToken;
		evt.tid = (char)threadToken;
		evt.ti = time;
		evt.par = parameter;
		tFile.traceBuffer[pos]=evt;//pos,
		tFile.tracePosition++;
		tFile.lastTimestamp = time;
		return 0;
	}

	public  static int Ttf_EnterState(Ttf_file tFile, long time, 
			int nodeToken, int threadToken, 
			int stateToken) {//time formerly (x_uint64)
		return enterExit(tFile, time, nodeToken, threadToken, stateToken, 1); // entry
	}

	public  static int Ttf_LeaveState(Ttf_file tFile, long time, 
			int nodeToken, int threadToken, int stateToken) {//time formerly (x_uint64)
		return enterExit(tFile, time, nodeToken, threadToken, stateToken, -1); // exit
	}

	public  static int Ttf_DefClkPeriod(Ttf_file tFile, double clkPeriod) {
		return 0;
	}

	static int sendRecv(Ttf_file tFile, long time, int sourceNodeToken,
			int sourceThreadToken,
			int destinationNodeToken,
			int destinationThreadToken,
			int messageSize,
			int messageTag,
			int messageComm, int eventId) {
		long parameter;
		long xother, xtype, xlength, xcomm;
	    xother = destinationNodeToken;
	    xtype = messageTag;
	    xlength = messageSize;
	    xcomm = messageComm;

		parameter = (xlength >> 16 << 54 >> 22) |
		((xtype >> 8 & 0xFF) << 48) |
		((xother >> 8 & 0xFF) << 56) |
		(xlength & 0xFFFF) | 
		((xtype & 0xFF)  << 16) | 
		((xother & 0xFF) << 24) |
		(xcomm << 58 >> 16);

		checkFlush(tFile);
		int pos = tFile.tracePosition;
		Event evt = new Event();
		evt.ev = eventId;
		evt.nid = (char)sourceNodeToken;
		evt.tid = (char)sourceThreadToken;
		evt.ti = time;
		evt.par = parameter;
		tFile.traceBuffer[pos]=evt;//pos,
		tFile.tracePosition++;
		tFile.lastTimestamp = time;

		return 0;
	}

	public static int Ttf_SendMessage(Ttf_file tFile, long time, int sourceNodeToken,
			int sourceThreadToken,
			int destinationNodeToken,
			int destinationThreadToken,
			int messageSize,
			int messageTag,
			int messageComm) {
	return sendRecv(tFile, time, sourceNodeToken, sourceThreadToken, destinationNodeToken, 
			destinationThreadToken, messageSize, messageTag, messageComm, TAU_MESSAGE_SEND);
		//return 0;
	}

	public static int Ttf_RecvMessage(Ttf_file tFile, long time, int sourceNodeToken,
			int sourceThreadToken,
			int destinationNodeToken,
			int destinationThreadToken,
			int messageSize,
			int messageTag,
			int messageComm) {
		return sendRecv(tFile, time, destinationNodeToken, 
				destinationThreadToken, sourceNodeToken, sourceThreadToken, 
				messageSize, messageTag, messageComm, TAU_MESSAGE_RECV);
		//return 0;
	}

	public static int Ttf_DefUserEvent(Ttf_file tFile, int userEventToken,String userEventName, int monotonicallyIncreasing) {
		Ttf_EventDescr newEventDesc=new Ttf_EventDescr();
		newEventDesc.Eid = userEventToken;
		newEventDesc.Group = "TAUEVENT";
		newEventDesc.EventName = userEventName;
		newEventDesc.Tag = monotonicallyIncreasing;
		newEventDesc.Param = "TriggerValue";

		tFile.EventIdMap.put(new Integer(userEventToken),newEventDesc);

		tFile.needsEdfFlush = true;
		return 0;
	}

	public static int Ttf_EventTrigger(Ttf_file tFile, long time, 
			int nodeToken,
			int threadToken,
			int userEventToken,
			long userEventValue) {
		checkFlush(tFile);
		int pos = tFile.tracePosition;
		Event evt = new Event();
		evt.ev = userEventToken;
		evt.nid = (char)nodeToken;
		evt.tid = (char)threadToken;
		evt.ti = time;
		evt.par = userEventValue;
		tFile.traceBuffer[pos]=evt;//pos,
		tFile.tracePosition++;
		tFile.lastTimestamp = time;
		return 0;
	}
}

