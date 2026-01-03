/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

package edu.uoregon.tau.trace;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;
import java.util.TreeMap;

public class TraceWriter extends TraceFile {
	
	private static final int IO_BUFFER_SIZE = 64 * 1024;
	
	DataOutputStream Foid;
	
	private final static int PCXX_EV_INIT = 60000;
	private final static int PCXX_EV_CLOSE = 60003;
	private final static int PCXX_EV_WALL_CLOCK = 60005;
	private final static int TAU_MESSAGE_SEND = 60007;
	private final static int TAU_MESSAGE_RECV = 60008;
	
	// Track which node/thread combinations have been initialized
	private final HashSet<Integer> checkInit = new HashSet<Integer>();
	private final HashMap<Integer, String> nidTidNames = new HashMap<Integer, String>();
	long lastTimestamp;
	
	public TraceWriter(String name, String edf) {
		try {
			FileOutputStream ostream = new FileOutputStream(name);
			BufferedOutputStream bw = new BufferedOutputStream(ostream, IO_BUFFER_SIZE);
			Foid = new DataOutputStream(bw);

			BufferedWriter out = new BufferedWriter(new FileWriter(edf));
			out.close();

			EdfFile = edf;

			// TreeMap maintains events ordered by ID for EDF output
			EventIdMap = new TreeMap<Integer, EventDescr>();
			IdGroupMap = new HashMap<Integer, String>();

			// Define standard tracer events

			EventDescr newEventDesc = new EventDescr(PCXX_EV_INIT, "TRACER", "EV_INIT", 0, "none");
			EventIdMap.put(Integer.valueOf(PCXX_EV_INIT), newEventDesc);
			
			newEventDesc = new EventDescr(PCXX_EV_CLOSE, "TRACER", "FLUSH_CLOSE", 0, "none");
			EventIdMap.put(Integer.valueOf(PCXX_EV_CLOSE), newEventDesc);
			
			newEventDesc = new EventDescr(PCXX_EV_WALL_CLOCK, "TRACER", "WALL_CLOCK", 0, "none");
			EventIdMap.put(Integer.valueOf(PCXX_EV_WALL_CLOCK), newEventDesc);
			
			newEventDesc = new EventDescr(TAU_MESSAGE_SEND, "TAU_MESSAGE", "MESSAGE_SEND", -7, "par");
			EventIdMap.put(Integer.valueOf(TAU_MESSAGE_SEND), newEventDesc);
			
			newEventDesc = new EventDescr(TAU_MESSAGE_RECV, "TAU_MESSAGE", "MESSAGE_RECV", -8, "par");
			EventIdMap.put(Integer.valueOf(TAU_MESSAGE_RECV), newEventDesc);

		} catch (SecurityException e1) {
			e1.printStackTrace();
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
	    } catch (IOException e1) {
	    	e1.printStackTrace();
		}	
		
	}
	
	// Combine node and thread IDs into single integer key
	private int CharPair(int nid, int tid) {
		return (nid << 16) + tid;
	}

	private int flushEdf() {
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter(EdfFile));

			int numEvents = EventIdMap.size();
			out.write(numEvents + " dynamic_trace_events\n");
			out.write("# FunctionId Group Tag \"Name Type\" Parameters\n");
			
			for (EventDescr eventDesc : EventIdMap.values()) {
				int id = eventDesc.getEventId();
				out.write(id + " " + eventDesc.getGroup() + " " + eventDesc.getTag() + 
				          " \"" + eventDesc.getEventName() + "\" " + eventDesc.getParameter() + "\n");
			}
			out.close();
		} catch (IOException e) {
			System.out.println("Error opening edf file");
			return -1;
		}
		return 0;
	}
public int defThread(int nodeToken, int threadToken, String threadName) {
		Integer nidtid = Integer.valueOf(CharPair(nodeToken, threadToken));
		nidTidNames.putIfAbsent(nidtid, threadName);
		return 0;
	}

	public int defStateGroup(String stateGroupName, int stateGroupToken) {
		IdGroupMap.put(Integer.valueOf(stateGroupToken), stateGroupName);
		return 0;
	}

	public int defState(int stateToken, String stateName, int stateGroupToken) {
		Integer groupKey = Integer.valueOf(stateGroupToken);
		if (!IdGroupMap.containsKey(groupKey)) {
			return -1;
		}

		EventDescr newEventDesc = new EventDescr(stateToken, IdGroupMap.get(groupKey), 
		                                          stateName, 0, "EntryExit");
		EventIdMap.put(Integer.valueOf(stateToken), newEventDesc);
		return 0;
	}

	public int writeEvent(int eventID, char nodeID, char threadID, long parameter, long time) {
		try {
			// Write initialization record for new node/thread combinations
			if (checkInit.add(Integer.valueOf(CharPair(nodeID, threadID)))) {
				Foid.writeInt(PCXX_EV_INIT);
				Foid.writeChar(nodeID);
				Foid.writeChar(threadID);
				Foid.writeLong(3);
				Foid.writeLong(time);
			}

			Foid.writeInt(eventID);
			Foid.writeChar(nodeID);
			Foid.writeChar(threadID);
			Foid.writeLong(parameter);
			Foid.writeLong(time);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return 0;
	}
	
	public int closeTrace() {
		// Write closing events for all node/thread combinations
		for (Integer nidtid : nidTidNames.keySet()) {
			char nid = (char)(nidtid >>> 16);
			char tid = (char)nidtid.intValue();
			
			writeEvent(PCXX_EV_CLOSE, nid, tid, 0, lastTimestamp);
			writeEvent(PCXX_EV_WALL_CLOCK, nid, tid, 0, lastTimestamp);
		}
		
		flushEdf();
		
		try {
			Foid.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return 0;
	}

	private int enterExit(long time, int nodeToken, int threadToken, 
	                      int stateToken, int parameter) {
		writeEvent(stateToken, (char)nodeToken, (char)threadToken, parameter, time);
		lastTimestamp = time;
		return 0;
	}

	public int enterState(long time, int nodeToken, int threadToken, int stateToken) {
		return enterExit(time, nodeToken, threadToken, stateToken, 1);
	}

	public int leaveState(long time, int nodeToken, int threadToken, int stateToken) {
		return enterExit(time, nodeToken, threadToken, stateToken, -1);
	}

	public int defClkPeriod(double clkPeriod) {
		return 0;
	}

	private int sendRecv(long time, int sourceNodeToken, int sourceThreadToken,
	                     int destinationNodeToken, int destinationThreadToken,
	                     int messageSize, int messageTag, int messageComm, int eventId) {
		// Pack message metadata into parameter field
		long xother = destinationNodeToken;
		long xtype = messageTag;
		long xlength = messageSize;
		long xcomm = messageComm;

		long parameter = (xlength >> 16 << 54 >> 22) |
		                 ((xtype >> 8 & 0xFF) << 48) |
		                 ((xother >> 8 & 0xFF) << 56) |
		                 (xlength & 0xFFFF) | 
		                 ((xtype & 0xFF)  << 16) | 
		                 ((xother & 0xFF) << 24) |
		                 (xcomm << 58 >> 16);

		writeEvent(eventId, (char)sourceNodeToken, (char)sourceThreadToken, parameter, time);
		lastTimestamp = time;
		return 0;
	}

	public int sendMessage(long time, int sourceNodeToken, int sourceThreadToken,
	                       int destinationNodeToken, int destinationThreadToken,
	                       int messageSize, int messageTag, int messageComm) {
		return sendRecv(time, sourceNodeToken, sourceThreadToken, destinationNodeToken, 
		                destinationThreadToken, messageSize, messageTag, messageComm, TAU_MESSAGE_SEND);
	}

	public int recvMessage(long time, int sourceNodeToken, int sourceThreadToken,
	                       int destinationNodeToken, int destinationThreadToken,
	                       int messageSize, int messageTag, int messageComm) {
		return sendRecv(time, destinationNodeToken, destinationThreadToken, 
		                sourceNodeToken, sourceThreadToken, 
		                messageSize, messageTag, messageComm, TAU_MESSAGE_RECV);
	}

	public int defUserEvent(int userEventToken, String userEventName, int monotonicallyIncreasing) {
		EventDescr newEventDesc = new EventDescr(userEventToken, "TAUEVENT", userEventName, 
		                                          monotonicallyIncreasing, "TriggerValue");
		EventIdMap.put(Integer.valueOf(userEventToken), newEventDesc);
		return 0;
	}

	public int eventTrigger(long time, int nodeToken, int threadToken,
	                        int userEventToken, long userEventValue) {
		writeEvent(userEventToken, (char)nodeToken, (char)threadToken, userEventValue, time);
		lastTimestamp = time;
		return 0;
	}
}