/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

package edu.uoregon.tau.trace;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.BitSet;

public class TraceReader extends TraceFile {
	
	private final static int FORMAT_NATIVE = 0;
	private final static int FORMAT_32 = 1;
	private final static int FORMAT_32_SWAP = 2;
	private final static int FORMAT_64 = 3;
	private final static int FORMAT_64_SWAP = 4;
	private static final int IO_BUFFER_SIZE = 64 * 1024;
	
	// Event buffering: maintain two Event objects to avoid allocation on every read
	private Event currentEvent = new Event();
	private Event spareEvent = new Event();
	private boolean hasBufferedEvent = false;
	
	boolean ClkInitialized;
	boolean subtractFirstTimestamp;
	boolean nonBlocking;
	boolean definitionsOnly = false;
	boolean done = false;
	int format;
	long totalRecords = 0;
	long totalRead = 0;
	char nid_offset = 0;
	
	ThreadTracker tracker = new ThreadTracker();
	private int cachedNid = Integer.MIN_VALUE;
	private ThreadTracker.NodeState cachedNodeState;
	
	/**
	 * Tracks thread state per node using BitSets for memory efficiency.
	 * Uses a two-level map structure: NodeID -> NodeState -> ThreadID
	 * to avoid computing global IDs and to support dynamic thread counts.
	 */
	public static class ThreadTracker {
		public static class NodeState {
			public final BitSet seen = new BitSet();
			public final BitSet done = new BitSet();
		}

		private final Map<Integer, NodeState> nodeStates = new HashMap<>();

		public boolean checkAndMarkSeen(int nid, int tid) {
			NodeState state = getNodeState(nid);
			if (state.seen.get(tid)) {
				return true;
			}
			state.seen.set(tid);
			return false;
		}

		public void markDone(int nid, int tid) {
			getNodeState(nid).done.set(tid);
		}
		
		public boolean isDone(int nid, int tid) {
			NodeState state = nodeStates.get(nid);
			return state != null && state.done.get(tid);
		}

		public NodeState getNodeState(int nid) {
			NodeState state = nodeStates.get(nid);
			if (state == null) {
				state = new NodeState();
				nodeStates.put(nid, state);
			}
			return state;
		}
	}
	
	public TraceReader(String trace, String edf, char nid_offset) {
		this.nid_offset = nid_offset;
		initialize(trace, edf);
		// Read first event into buffer for immediate availability
		try {
			advance();
		} catch (IOException e) {
			e.printStackTrace();
			this.done = true;
		}
	}
	
	public TraceReader(String trace, String edf) {
		this(trace, edf, (char)0);
	}
	
	
	private void initialize(String trace, String edf){
		subtractFirstTimestamp = true;
		nonBlocking = false;

		
		/* Open the trace file */
		FileInputStream istream=null;;
		
		try {	  
			istream = new FileInputStream(trace);
			BufferedInputStream bw = new BufferedInputStream(istream, IO_BUFFER_SIZE);
			Fiid = new DataInputStream(bw);
			format=TraceReader.determineFormat(Fiid);
		
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		totalRecords=getNumRecords(trace);

		/* make a copy of the EDF file name */
		EdfFile = edf;
		TrcFile=trace;
		String[] cutname = trace.split(".");
		if(cutname.length==5){
			node=Integer.parseInt(cutname[1])+nid_offset;
			//System.out.println("Set Node to "+node);
			context=Integer.parseInt(cutname[2]);
			thread=Integer.parseInt(cutname[3]);
		}
	/* Allocate space for event id map */
	EventIdMap = new HashMap<Integer, EventDescr>();
	/* Allocate space for group id map */
	GroupIdMap = new HashMap<String, Integer>();
	/* initialize clock */
	ClkInitialized = false;
	}

	// Advance to next event by swapping buffers and reading into spare
	private void advance() throws IOException {
		Event temp = currentEvent;
		currentEvent = spareEvent;
		spareEvent = temp;

		hasBufferedEvent = readInto(currentEvent, Fiid, format);
		
		if (!hasBufferedEvent) {
			this.done = true;
		}
	}
	
	public static long getNumRecords(String name) {
		File f = new File(name);
		return f.length() / 24;
	}
	
	public long getNumRecords() {
		return totalRecords;
	}
	
	public String getTraceFile() {
		return this.TrcFile;
	}
	
	final private static char charReverseBytes(char value) {
		return (char)(((value >> 8) & 0x000000ff) | ((value << 8) & 0x0000ff00));
	}
	
	final private static int intReverseBytes(int value) {
		return ((((int)charReverseBytes((char)value)) << 16) | charReverseBytes((char)(value >>> 16)));
	}
	
	public static long longReverseBytes(long i) {
		i = (i & 0x00ff00ff00ff00ffL) << 8 | (i >>> 8) & 0x00ff00ff00ff00ffL;
		return (i << 48) | ((i & 0xffff0000L) << 16) |
		       ((i >>> 16) & 0xffff0000L) | (i >>> 48);
	}
	
	DataInputStream Fiid;
	
	private final static int TAU_MESSAGE_SEND_EVENT = -7;
	private final static int TAU_MESSAGE_RECV_EVENT = -8;

	// Determine trace file format (32/64 bit, native/swapped endianness)
	static int determineFormat(DataInputStream Fiid) throws IOException {
		Event evt = new Event();
		Fiid.mark(128);
		evt.setEventID(Fiid.readInt());
		evt.setNodeID(Fiid.readChar());
		evt.setThreadID(Fiid.readChar());
		evt.setParameter(Fiid.readLong());
		evt.setTime(Fiid.readLong());
		
		int format = FORMAT_NATIVE;
		if (evt.getParameter() == 3) {
			format = FORMAT_32;
		} else if (longReverseBytes(evt.getParameter()) == 3) {
			format = FORMAT_32_SWAP;
		} else {
			Fiid.reset();
			//Event64 evt64 = new Event64();
			evt.setEventID((int)Fiid.readLong());
			evt.setNodeID(Fiid.readChar());
			evt.setThreadID(Fiid.readChar());
			Fiid.readInt();
			evt.setParameter(Fiid.readLong());
			evt.setTime(Fiid.readLong());
			if(evt.getParameter()==3)
			{
				format=FORMAT_64;
				//eventSize=32;
				//System.out.println("64!");
			}
			else
			if(longReverseBytes(evt.getParameter())==3)
			{
				format=FORMAT_64_SWAP;
				//eventSize=32;
				//System.out.println("Swapping 64!");
			}
			else
			{
				System.out.println("Could not determine trace format, using native.");
				//eventSize=24;
			}
		}
		Fiid.reset();
		return format;
	}
	
	
	 /**
     * Internal method to populate an existing Event object from the stream.
     * Replaces the old static readEvent().
     */
    private boolean readInto(Event evt, DataInputStream stream, int fmt) throws IOException {
        if (stream == null) return false;
        
        try {
            // Check for EOF before starting to read
            // Note: DataInputStream doesn't always handle available() reliably for EOF, 
            // but the read methods throw EOFException.
            
            if (fmt < 2) {
                evt.setEventID(stream.readInt());
                evt.setNodeID(stream.readChar());
                evt.setThreadID(stream.readChar());
                evt.setParameter(stream.readLong());
                evt.setTime(stream.readLong());
            } else if (fmt == 2) {
                evt.setEventID(intReverseBytes(stream.readInt()));
                evt.setNodeID(charReverseBytes(stream.readChar()));
                evt.setThreadID(charReverseBytes(stream.readChar()));
                evt.setParameter(longReverseBytes(stream.readLong()));
                evt.setTime(longReverseBytes(stream.readLong()));
            } else if (fmt == 3) {
                evt.setEventID((int) stream.readLong());
                evt.setNodeID(stream.readChar());
                evt.setThreadID(stream.readChar());
                stream.readInt(); // Padding
                evt.setParameter(stream.readLong());
                evt.setTime(stream.readLong());
            } else if (fmt == 4) {
                evt.setEventID((int) longReverseBytes(stream.readLong()));
                evt.setNodeID(charReverseBytes(stream.readChar()));
                evt.setThreadID(charReverseBytes(stream.readChar()));
                stream.readInt(); // Padding
                evt.setParameter(longReverseBytes(stream.readLong()));
                evt.setTime(longReverseBytes(stream.readLong()));
            }
            evt.nid_offset = this.nid_offset;
            return true;
        } catch (EOFException e) {
            return false;
        }
    }
	
	/* Look for an event in the event map */
	/*private boolean isEventIDRegistered(int event)
	{
		return EventIdMap.containsKey(Integer.valueOf(event));
	}	*/
	
	/* Event ID is not found in the event map. Re-read the event 
	 * description file */
	private boolean refreshTables(TraceReaderCallbacks cb, Object userData)//, 
	{
		int i,j,k; 
		String linebuf, eventname, traceflag; //[LINEMAX]=2||64*1024,[LINEMAX],[32]
		String group, param;//[512]
		long tag;
		int numevents, groupid; 
		int localEventId;
		boolean dynamictrace = false;

		/* first, open the edf file */
		BufferedReader edf;
	try {
		edf = new BufferedReader(new FileReader(EdfFile));
		linebuf =edf.readLine();
		String[] asplit = linebuf.split(" ");
		traceflag=asplit[1];
		numevents = Integer.parseInt(asplit[0]);
		if ((traceflag != null) && (traceflag.equals("dynamic_trace_events"))) 
		{ 
			dynamictrace = true;
		}

		for (i=0; i<numevents; i++)
		{
			linebuf=edf.readLine();
			
			if(linebuf==null){
				int c = i-1;
				System.out.println("Warning: Expected "+numevents+" event definitions. Found "+c);
				edf.close();
				return false;
			}
			
			if ( (linebuf.charAt(0) == '\n') || (linebuf.charAt(0) == '#') )
			{
				/* -- skip empty, header and comment lines -- */
				i--;
				continue;
			}

			localEventId = -1;
			if (dynamictrace) /* get eventname in quotes */
			{
				asplit=linebuf.split(" ");
				localEventId=Integer.parseInt(asplit[0]);
				group=asplit[1];
				tag=Long.parseLong(asplit[2]);
				
				j = linebuf.indexOf('"');
				k= linebuf.indexOf('"', j+1);
				eventname=linebuf.substring(j, k+1);
				param=linebuf.substring(k+2);

				
				if(eventname.startsWith("\"")&&eventname.endsWith("\"")){
					eventname=eventname.substring(1, eventname.length()-1);
				}
				
				/* see if the event id exists in the map */
				if (!EventIdMap.containsKey(Integer.valueOf(localEventId)))//isEventIDRegistered(localEventId))
				{
					/* couldn't locate the event id */
					/* fill an event description object */
					EventDescr eventDescr = new EventDescr(localEventId, new String(group), new String(eventname),(long)tag,new String(param));

					EventIdMap.put(Integer.valueOf(localEventId),eventDescr); /* add it to the map */

					if (!GroupIdMap.containsKey(eventDescr.getGroup()))
					{ 
						/* group id not found. Generate group id on the fly */
						groupid = GroupIdMap.size()+1;
						
						/* invoke group callback */
						/* check Param to see if its a user defined event */
						if (eventDescr.getParameter().equals("EntryExit"))
						{ /* it is not a user defined event */
							GroupIdMap.put(eventDescr.getGroup(),Integer.valueOf(groupid));
							//if (cb.DefStateGroup!=null)
								cb.defStateGroup(userData, groupid, eventDescr.getGroup()); 
						}
					}
					else
					{ /* retrieve the stored group id token */
						groupid = GroupIdMap.get(eventDescr.getGroup()).intValue();
					}
					/* invoke callback for registering a new state */
					if (eventDescr.getParameter().equals("TriggerValue"))//||eventDescr.Param.equals("none")
					{ /* it is a user defined event */
						//if (cb.DefUserEvent!=null)
							cb.defUserEvent(userData, localEventId, eventDescr.getEventName(), (int)eventDescr.getTag());
					}
					else if(eventDescr.getParameter().equals("EntryExit"))//(!eventDescr.Param.equals("TriggerValue"))//
					{ /* it is an entry/exit event */
						//if (cb.DefState!=null)
							cb.defState(userData, localEventId, eventDescr.getEventName(),groupid);
					}
				}
				//else
				//System.out.println("SKIPPED "+linebuf);
				/* else, do nothing, examine the next record */
			} /* not dynamic trace- what is to be done? */ 
			else 
			{
				asplit=linebuf.split(" ");
				localEventId=Integer.parseInt(asplit[0]);
				group=asplit[1];
				tag=Integer.parseInt(asplit[2]);
				eventname=asplit[3];
				if(eventname!=null && eventname.startsWith("\"")&&eventname.endsWith("\"")){
					eventname=eventname.substring(1, eventname.length()-2);
				}
				param=asplit[4];
			}

			if ( (localEventId < 0) || eventname==null )
			{
				System.out.println("Blurb error?");
				edf.close();
				return false;
			}
		} /* for loop */
		edf.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return true;
	}//REFRESHTABLES
	
	public void setSubtractFirstTimestamp(boolean value ){
		subtractFirstTimestamp=value;
	}
	
	public void setNonBlocking(boolean value ){
		nonBlocking=value;
	}

	/* Seek to an absolute event position. 
	 * A negative position indicates to start from the tail of the event stream. 
	 * Returns the position if successful or 0 if an error occured */
	public int absSeek(int eventPosition ){
		return 0;
	}

	/* seek to a event position relative to the current position (just for completeness!) 
	 * Returns the position if successful or 0 if an error occured */
	public int relSeek(int plusMinusNumEvents ){
		return 0;
	}
	
	public void setDefsOnly(boolean value)
	{
		definitionsOnly=value;
	}

	
	long FirstTimestamp=0;
	
	public boolean isDone(){
		return this.done;
	}

    /**
     * Optimized peekTime. O(1) cost, no file IO.
     */
    public long peekTime() {
        if (!hasBufferedEvent || done) {
            return -1; 
        }
        return currentEvent.getTime();
    }
	
	/* read n events and call appropriate handlers.
	 * Returns the number of records read (can be 0).
	 * Returns a -1 value when an error takes place. Check errno */
	public int readNumEvents(TraceReaderCallbacks callbacks, int nOEvents, Object userData){
		int recordsRead=0;
		long numberOfEvents = (nOEvents <= 0) ? (totalRecords - totalRead) : nOEvents;
		long otherTid, otherNid, msgLen, msgTag;//,numberOfEvents;
		
		if(this.done){
		return 0;
		}
		//if (tFile == null)
		//	return 0; /* ERROR */

		/* How many bytes are to be read? */
		//recordsToRead = numberOfEvents;// > TAU_BUFSIZE ? TAU_BUFSIZE : numberOfEvents;

		/* if clock needs to be initialized, initialize it */
		if (!ClkInitialized)
		{
			refreshTables(callbacks, userData);
			//if (callbacks.DefClkPeriod != null)
			callbacks.defClkPeriod(userData, 1E-6);
			
			if(definitionsOnly){
				return 0;
			}
	
			
			/* set flag to initialized */
			ClkInitialized = true; 

			/* Read the first record and check its timestamp 
			 * For this we need to lseek to the beginning of the file, read one 
			 * record and then lseek it back to where it was */
			
			FirstTimestamp=peekTime();
			
		}//!clkinit

		/*if(nOEvents<=0){
			
			numberOfEvents=totalRecords-totalRead;
		}
		else{
			numberOfEvents=nOEvents;
		}*/
		

		
//		if(Fiid.available()<=0){
//			return null;
//		}
		
		/* Read n records and go through each event record */
		String nodename ="";
		/* the number of records read */
		/* See if the events are all present */
		for (int i = 0; i < numberOfEvents; i++)
		{
	            if (!hasBufferedEvent) break;

	            Event evt = currentEvent; // Use the already loaded event
	            recordsRead++;
	            totalRead++;

	            ThreadTracker.NodeState state;
	            if (evt.nid == cachedNid && cachedNodeState != null) {
	            	state = cachedNodeState;
	            } else {
	            	state = tracker.getNodeState(evt.nid);
	            	cachedNodeState = state;
	            	cachedNid = evt.nid;
	            }
			
		    /* event is OK. Examine each event and invoke callbacks for Entry/Exit/Node*/
		    /* first check nodeid, threadid */

			boolean alreadySeen = state.seen.get(evt.tid);
			if (!alreadySeen)
			{
				/* this pair of node and thread has not been encountered before*/
				nodename="process "+(int)evt.nid+":"+(int)evt.tid;
				/* invoke callback routine */
				callbacks.defThread(userData, evt.nid, evt.tid, nodename);
				//System.out.println("Nodename: "+nodename);
				/* add it to the map! */
				//NidTidMap.put(nidtid, one);
				//nidTidSeen.add(nidtid);
				state.seen.set(evt.tid);
			}
		    /* check the event to see if it is entry or exit */
			//time=traceBuffer[i].getTime();//event_GetTi(traceBuffer, i);
			if (subtractFirstTimestamp) {
				evt.time -= FirstTimestamp;
			}
			
			//parameter = traceBuffer[i].getParameter();//event_GetPar(traceBuffer, i);
			/* Get param entry from EventIdMap */
			
//			if(evt.evid>60001){
//				System.out.println("Built In Event");
//			}
			
			EventDescr eventDescr = EventIdMap.get(Integer.valueOf(evt.evid));
			if(eventDescr==null){
				System.out.println("Warning: no event definiton for event ID "+evt.evid);
				try {
					advance();
				} catch (IOException e) {
					e.printStackTrace();
					return -1;
				}				
				continue;
			}
			if ((eventDescr.getParameter() != null) && ((eventDescr.getParameter().equals("EntryExit"))))
			{ /* entry/exit event */
				if (evt.parameter == 1)
				{ /* entry event, invoke the callback routine */
					//if (callbacks.EnterState!=null)
						callbacks.enterState(userData, evt.time, evt.nid, 
								evt.tid,evt.evid);
				}
				else
				{ if (evt.parameter == -1)
					{ /* exit event */
						//if (callbacks.LeaveState!=null)
							callbacks.leaveState(userData,evt.time, evt.nid, 
									evt.tid,evt.evid);
					}
				}
			} /* entry exit events *//* add message passing events here */
			else 
			{
				if ((eventDescr.getParameter() != null) && (eventDescr.getParameter().equals("TriggerValue")))//||eventDescr.Param.equals("none")
				{ /* User defined event */
					//if (callbacks.EventTrigger!=null) {
						//parameter = event_GetPar(traceBuffer, i);

						callbacks.eventTrigger(userData, evt.time, evt.nid, evt.tid, evt.evid,evt.parameter);
					//}
				}
				if (eventDescr.getTag() == TAU_MESSAGE_SEND_EVENT||eventDescr.getEventId()==60007) 
				{/* send message */
		        /* See RtsLayer::TraceSendMsg for documentation on the bit patterns of "parameter" */
					long xpar = evt.parameter;
					/* extract the information from the parameter */
					msgTag   = ((xpar>>16) & 0x000000FF) | (((xpar >> 48) & 0xFF) << 8);
					otherNid = ((xpar>>24) & 0x000000FF) | (((xpar >> 56) & 0xFF) << 8);
					msgLen   = xpar & 0x0000FFFF | (xpar << 22 >> 54 << 16);
					long comm = xpar << 16 >> 58;

					/* If the application is multithreaded, insert call for matching sends/recvs here */
					otherTid = evt.parameter;
					//if (callbacks.SendMessage!=null) 
//					if(eventDescr.getTag()==TAU_MESSAGE_SEND_EVENT){
//						callbacks.sendMessage(userData, time, nid, tid, (int)otherNid, 
//								(int)otherTid, (int)msgLen, (int)msgTag, (int)comm);
//					}
//					else{
						callbacks.sendMessage(userData, evt.time, evt.nid, evt.tid, (int)otherNid, 
								(int)otherTid, (int)msgLen, (int)msgTag, (int)comm);
//					}
			/* the args are user, time, source nid (my), source tid (my), dest nid (other), dest
			 * tid (other), size, tag */
				}
				else
				{ /* Check if it is a message receive operation */
					if (eventDescr.getTag() == TAU_MESSAGE_RECV_EVENT||eventDescr.getEventId()==60008)
					{
						/* See RtsLayer::TraceSendMsg for documentation on the bit patterns of "parameter" */
						long xpar = evt.parameter;
						/* extract the information from the parameter */
						msgTag   = ((xpar>>16) & 0x000000FF) | (((xpar >> 48) & 0xFF) << 8);
						otherNid = ((xpar>>24) & 0x000000FF) | (((xpar >> 56) & 0xFF) << 8);
						msgLen   = xpar & 0x0000FFFF | (xpar << 22 >> 54 << 16);
						long comm = xpar << 16 >> 58;

						/* If the application is multithreaded, insert call for matching sends/recvs here */
						otherTid = evt.parameter;//TODO: not 0 any more
//						if(eventDescr.getTag()==TAU_MESSAGE_RECV_EVENT){
//						//if (callbacks.RecvMessage!=null) 
//							callbacks.recvMessage(userData, time, (int)otherNid, 
//									(int)otherTid, nid, tid, (int)msgLen, (int)msgTag, (int)comm);
//						/* the args are user, time, source nid (my), source tid (my), dest nid (other), dest
//						 * tid (other), size, tag */
//						}else{
							callbacks.recvMessage(userData, evt.time, (int)otherNid, 
									(int)otherTid, evt.nid, evt.tid, (int)msgLen, (int)msgTag, (int)comm);
						//}
					}
				}
			}
			if ((evt.parameter == 0) && (eventDescr.getEventName()!= null) &&(eventDescr.getEventName().equals("\"FLUSH_CLOSE\"")||eventDescr.getEventName().equals("FLUSH_CLOSE"))) {
				/* reset the flag in NidTidMap to 0 (from 1) */
				state.done.set(evt.tid);//nidTidDone.add(nidtid);//NidTidMap.put(nidtid,zero);
				//callbacks.eventTrigger(userData, evt.time, evt.nid, evt.tid, evt.evid, evt.parameter);
				/* setting this flag to 0 tells us that a flush close has taken place 
				 * on this node, thread */
			} 
			else 
			{/* see if it is a WALL_CLOCK record */
				if ((evt.parameter != 1) && (evt.parameter != -1) && (eventDescr.getEventName() != null) 
						&& (eventDescr.getEventName().equals("\"WALL_CLOCK\"")||eventDescr.getEventName().equals("WALL_CLOCK"))) {
			/* ok, it is a wallclock event alright. But is it the *last* wallclock event?
			 * We can confirm that it is if the NidTidMap flag has been set to 0 by a 
			 * previous FLUSH_CLOSE call */
					//callbacks.eventTrigger(userData, evt.time, evt.nid, evt.tid, evt.evid, evt.parameter);
					//if (NidTidMap.containsKey(nidtid))
					//{/*printf("LAST WALL_CLOCK! End of trace file detected \n");*/
						/* see if an end of the trace callback is registered and 
						 * if it is, invoke it.*/
					if(state.done.get(evt.tid))//if(NidTidMap.get(nidtid).equals(zero))//if (callbacks.EndTrace!=null) 
					{	
						callbacks.endTrace(userData, evt.nid, evt.tid);
						this.done=true;
						    try {
								advance(); 
							} catch (IOException e) {
								e.printStackTrace();
							}
						return recordsRead;
					}
						//System.out.println("Wallclock at "+ts);
					//}
				}
			} /* is it a WALL_CLOCK record? */

			try {
                advance();
            } catch (IOException e) {
                e.printStackTrace();
                return -1;
            }
			
		} /* cycle through all records */
		/* return the number of event records read */
		return recordsRead;
	}

	/* close a trace file */
	public void closeTrace()
	{
		
		try {
			if(Fiid!=null)
			{
				Fiid.close();
			}else
			{
				System.out.println("Warning: tried to close null file handle");
			}
			
				
		} catch (IOException e) {
			e.printStackTrace();
		}
	}	
}
