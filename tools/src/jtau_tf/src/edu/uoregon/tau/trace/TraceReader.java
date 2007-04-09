/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

package edu.uoregon.tau.trace;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class TraceReader extends TraceFile{	
	
	private final static int FORMAT_NATIVE =0;//Java default
	private final static int FORMAT_32 = 1;
	private final static int FORMAT_32_SWAP = 2;
	private final static int FORMAT_64 =3;
	private final static int FORMAT_64_SWAP=4;
	
	
	TraceReader(){}
	
	private static int intReverseBytes(int value){
		//Integer.
		ByteBuffer bb = ByteBuffer.allocate(4);
		bb.putInt(value);
		bb.rewind();
		//ByteBuffer.wrap( new byte[]{(byte)(value >>> 24), (byte)(value >> 16 & 0xff), (byte)(value >> 8 & 0xff), (byte)(value & 0xff)} )	
		return bb.order( ByteOrder.LITTLE_ENDIAN ).getInt();// .getFloat();
	}
	
	private static char charReverseBytes(char value){
		ByteBuffer bb = ByteBuffer.allocate(2);
		bb.putChar(value);
		bb.rewind();
		return bb.order( ByteOrder.LITTLE_ENDIAN ).getChar();// .getFloat();
	}
	
	private static long longReverseBytes(long value){
		ByteBuffer bb = ByteBuffer.allocate(8);
		bb.putLong(value);
		bb.rewind();
		return bb.order( ByteOrder.LITTLE_ENDIAN ).getLong();// .getFloat();
	}
	
	DataInputStream Fiid;//The trace file input handle
	
	/*public interface eventReader{
		public static final int ID=0;
	}*/
	private final static int TAU_BUFSIZE = 1024;
	private final static int TAU_MESSAGE_SEND_EVENT = -7;
	private final static int TAU_MESSAGE_RECV_EVENT = -8;

	/* for 32 bit platforms (24 bytes)*/
	
	void determineFormat() throws IOException{
		Event evt = new Event();//Fiid.readInt(),Fiid.readChar(),Fiid.readChar(),Fiid.readLong(),Fiid.readLong());
		Fiid.mark(128);
		evt.setEventID(Fiid.readInt());
		evt.setNodeID(Fiid.readChar());
		evt.setThreadID(Fiid.readChar());
		evt.setParameter(Fiid.readLong());
		evt.setTime(Fiid.readLong());
		format=FORMAT_NATIVE;
		if(evt.getParameter()==3)
		{
			format=FORMAT_32;
			eventSize=24;
			//System.out.println("Default!");
		}
		else
		if(longReverseBytes(evt.getParameter())==3)
		{
			format=FORMAT_32_SWAP;
			eventSize=24;
			//System.out.println("Swapping!");
		}
		else{
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
				eventSize=32;
				//System.out.println("64!");
			}
			else
			if(longReverseBytes(evt.getParameter())==3)
			{
				format=FORMAT_64_SWAP;
				eventSize=32;
				//System.out.println("Swapping 64!");
			}
			else
			{
				System.out.println("Could not determine trace format, using native.");
				eventSize=24;
			}
		}
		Fiid.reset();
		return;
	}

	private int readEvents(Event[] traceBuffer, int numread) throws IOException{
		int x=0;
		Event evt;// = new Event();
		int bytes=Fiid.available();
		int records=bytes/eventSize;
		
		//byte[] b = new byte[24];
		if(format<2)
			while(x<numread&&x<records){
				//tFile.Fid.read(b);
				//Integer.
				evt = new Event();//Fiid.readInt(),Fiid.readChar(),Fiid.readChar(),Fiid.readLong(),Fiid.readLong());
				evt.setEventID(Fiid.readInt());
				evt.setNodeID(Fiid.readChar());
				evt.setThreadID(Fiid.readChar());
				evt.setParameter(Fiid.readLong());
				evt.setTime(Fiid.readLong());
				traceBuffer[x]=evt;
				x++;
				//System.out.println("ID: "+x+" NID: "+c1+" TID: "+c2+" PAR: "+l1+" TID: "+l2);
			}
		else
		if(format==2)
		{
			while(x<numread&&x<records){
				evt = new Event();//intReverseBytes(Fiid.readInt()),charReverseBytes(Fiid.readChar()),charReverseBytes(Fiid.readChar()),longReverseBytes(Fiid.readLong()),longReverseBytes(Fiid.readLong()));
				evt.setEventID(intReverseBytes(Fiid.readInt()));
				evt.setNodeID(charReverseBytes(Fiid.readChar()));
				evt.setThreadID(charReverseBytes(Fiid.readChar()));
				evt.setParameter(longReverseBytes(Fiid.readLong()));
				evt.setTime(longReverseBytes(Fiid.readLong()));
				traceBuffer[x]=evt;
				x++;
				//System.out.println("ID: "+x+" NID: "+c1+" TID: "+c2+" PAR: "+l1+" TID: "+l2);
			}	
		}
		else
		if(format==3)
		{
			while(x<numread&&x<records){
				evt = new Event();
				evt.setEventID((int)Fiid.readLong());
				evt.setNodeID(Fiid.readChar());
				evt.setThreadID(Fiid.readChar());
				Fiid.readInt();
				evt.setParameter(Fiid.readLong());
				evt.setTime(Fiid.readLong());
			
				//System.out.println("ID: "+evt.ev+" NID: "+(int)evt.nid+" TID: "+(int)evt.tid+" PAR: "+evt.par+" TIM: "+evt.ti);
			
				traceBuffer[x]=evt;
				x++;
			}
		}
		else
		if(format==4)
		{
			while(x<numread&&x<records){
				evt = new Event();
				evt.setEventID((int)longReverseBytes(Fiid.readLong()));
				evt.setNodeID(charReverseBytes(Fiid.readChar()));
				evt.setThreadID(charReverseBytes(Fiid.readChar()));
				Fiid.readInt();
				evt.setParameter(longReverseBytes(Fiid.readLong()));
				evt.setTime(longReverseBytes(Fiid.readLong()));
			
				//System.out.println("ID: "+evt.ev+" NID: "+(int)evt.nid+" TID: "+(int)evt.tid+" PAR: "+evt.par+" TIM: "+evt.ti);
			
				traceBuffer[x]=evt;
				x++;
			}			
		}
		return x;
	}
	
	private static int event_GetEv(Event[] traceBuffer, int index){
		return (traceBuffer[index]).getEventID();
	}
	
	private static int event_GetNid(Event[] traceBuffer, int index){
		return (traceBuffer[index]).getNodeID();
	}
	
	private static int event_GetTid(Event[] traceBuffer, int index){
		return (traceBuffer[index]).getThreadID();
	}
	
	private static long event_GetPar(Event[] traceBuffer, int index){
		return (traceBuffer[index]).getParameter();
	}
	
	private static long event_GetTi(Event[] traceBuffer, int index){
		return (traceBuffer[index]).getTime();
	}
	
	/* Look for an event in the event map */
	private boolean isEventIDRegistered(int event)
	{
		return EventIdMap.containsKey(new Integer(event));
	}	
	
	/* Event ID is not found in the event map. Re-read the event 
	 * description file */
	private boolean refreshTables(TraceReaderCallbacks cb, Object userData)//, 
	{
		int i,j,k; 
		String linebuf, eventname, traceflag; //[LINEMAX]=2||64*1024,[LINEMAX],[32]
		String group, param;//[512]
		int numevents, tag, groupid; 
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
				tag=Integer.parseInt(asplit[2]);
				
				j = linebuf.indexOf('"');
				k= linebuf.indexOf('"', j+1);
				eventname=linebuf.substring(j, k+1);
				param=linebuf.substring(k+2);

				/* see if the event id exists in the map */
				if (!isEventIDRegistered(localEventId))
				{
					/* couldn't locate the event id */
					/* fill an event description object */
					EventDescr eventDescr = new EventDescr(localEventId, new String(eventname), new String(group),tag,new String(param));
					/*eventDescr.Eid = localEventId;
					eventDescr.EventName = new String(eventname);
					eventDescr.Group = new String(group);
					eventDescr.Tag = tag;
					eventDescr.Param = new String(param);*/
					EventIdMap.put(new Integer(localEventId),eventDescr); /* add it to the map */

					if (!GroupIdMap.containsKey(eventDescr.getGroup()))
					{ 
						/* group id not found. Generate group id on the fly */
						groupid = GroupIdMap.size()+1;
						
						/* invoke group callback */
						/* check Param to see if its a user defined event */
						if (eventDescr.getParameter().equals("EntryExit"))
						{ /* it is not a user defined event */
							GroupIdMap.put(eventDescr.getGroup(),new Integer(groupid));
							//if (cb.DefStateGroup!=null)
								cb.defStateGroup(userData, groupid, eventDescr.getGroup()); 
						}
					}
					else
					{ /* retrieve the stored group id token */
						groupid = ((Integer)GroupIdMap.get(eventDescr.getGroup())).intValue();
					}
					/* invoke callback for registering a new state */
					if (eventDescr.getParameter().equals("TriggerValue"))//||eventDescr.Param.equals("none")
					{ /* it is a user defined event */
						//if (cb.DefUserEvent!=null)
							cb.defUserEvent(userData, localEventId, eventDescr.getEventName(), eventDescr.getTag());
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
				param=asplit[4];
			}

			if ( (localEventId < 0) || eventname==null )
			{
				System.out.println("Blurb error?");
				return false;
			}
		} /* for loop */
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

	/* read n events and call appropriate handlers.
	 * Returns the number of records read (can be 0).
	 * Returns a -1 value when an error takes place. Check errno */
	public int readNumEvents(TraceReaderCallbacks callbacks, int numberOfEvents, Object userData){
		Event[] traceBuffer = new Event[TAU_BUFSIZE];
		int recordsRead=0, recordsToRead;
		long otherTid, otherNid, msgLen, msgTag;

		//if (tFile == null)
		//	return 0; /* ERROR */

		/* How many bytes are to be read? */
		recordsToRead = numberOfEvents > TAU_BUFSIZE ? TAU_BUFSIZE : numberOfEvents;

		/* if clock needs to be initialized, initialize it */
		if (!ClkInitialized)
		{
			//if (callbacks.DefClkPeriod != null)
				callbacks.defClkPeriod(userData, 1E-6);
			/* set flag to initialized */
			ClkInitialized = true; 

			/* Read the first record and check its timestamp 
			 * For this we need to lseek to the beginning of the file, read one 
			 * record and then lseek it back to where it was */
			Fiid.mark(64);
			try {
			readEvents(traceBuffer,1);
			
				Fiid.reset();
			} catch (IOException e) {
				e.printStackTrace();
			}
			FirstTimestamp = event_GetTi(traceBuffer,0);
		}//!clkinit

		/* Read n records and go through each event record */
		int read=0;
		try {
			read = readEvents(traceBuffer,recordsToRead);
		} catch (IOException e) {
			e.printStackTrace();
		}
		recordsToRead-=read;
		recordsRead+=read;
		int nid=0;
		int tid=0;
		Integer zero = new Integer(0);
		Integer one = new Integer(1);
		String nodename ="";
		String nidtid="";
		/* the number of records read */
		/* See if the events are all present */
		for (int i = 0; i < recordsRead; i++)
		{
			if (!isEventIDRegistered(event_GetEv(traceBuffer, i)))
			{
				/* if event id is not found in the event id map, read the EDF file */
				if (!refreshTables(callbacks, userData))
				{ /* error */
					System.out.println("Refresh Tables Error");
					return -1;
				}
				if (!isEventIDRegistered(event_GetEv(traceBuffer, i)))
				{ /* even after reading the edf file, if we don't find the event id, then there's an error */
					System.out.println("ID Reg error");
					return -1;
				}
				/* we did find the event id, process the trace file */
			}
		    /* event is OK. Examine each event and invoke callbacks for Entry/Exit/Node*/
		    /* first check nodeid, threadid */

			nid = event_GetNid(traceBuffer, i);
			tid = event_GetTid(traceBuffer, i);
			nidtid=nid+":"+tid;
			
			if (!NidTidMap.containsKey(nidtid))
			{
				/* this pair of node and thread has not been encountered before*/
				nodename="process "+nidtid;//nid+":"+tid;
				/* invoke callback routine */
				callbacks.defThread(userData, nid, tid, nodename);
				/* add it to the map! */
				NidTidMap.put(nidtid, one);
			}
		    /* check the event to see if it is entry or exit */
			long ts=event_GetTi(traceBuffer, i);
			if (subtractFirstTimestamp) {
				ts -= FirstTimestamp;
			}
			
			long parameter = event_GetPar(traceBuffer, i);
			/* Get param entry from EventIdMap */
			
			EventDescr eventDescr = (EventDescr)EventIdMap.get(new Integer(event_GetEv(traceBuffer, i)));
			if ((eventDescr.getParameter() != null) && ((eventDescr.getParameter().equals("EntryExit"))))
			{ /* entry/exit event */
				if (parameter == 1)
				{ /* entry event, invoke the callback routine */
					//if (callbacks.EnterState!=null)
						callbacks.enterState(userData, ts, nid, 
								tid,event_GetEv(traceBuffer, i));
				}
				else
				{ if (parameter == -1)
					{ /* exit event */
						//if (callbacks.LeaveState!=null)
							callbacks.leaveState(userData,ts, nid, 
									tid,event_GetEv(traceBuffer, i));
					}
				}
			} /* entry exit events *//* add message passing events here */
			else 
			{
				if ((eventDescr.getParameter() != null) && (eventDescr.getParameter().equals("TriggerValue")))//||eventDescr.Param.equals("none")
				{ /* User defined event */
					//if (callbacks.EventTrigger!=null) {
						//parameter = event_GetPar(traceBuffer, i);

						callbacks.eventTrigger(userData, ts, nid, tid, event_GetEv(traceBuffer, i),parameter);
					//}
				}
				if (eventDescr.getTag() == TAU_MESSAGE_SEND_EVENT) 
				{/* send message */
		        /* See RtsLayer::TraceSendMsg for documentation on the bit patterns of "parameter" */
					long xpar = parameter;
					/* extract the information from the parameter */
					msgTag   = ((xpar>>16) & 0x000000FF) | (((xpar >> 48) & 0xFF) << 8);
					otherNid = ((xpar>>24) & 0x000000FF) | (((xpar >> 56) & 0xFF) << 8);
					msgLen   = xpar & 0x0000FFFF | (xpar << 22 >> 54 << 16);
					long comm = xpar << 16 >> 58;

					/* If the application is multithreaded, insert call for matching sends/recvs here */
					otherTid = 0;
					//if (callbacks.SendMessage!=null) 
						callbacks.sendMessage(userData, ts, nid, tid, (int)otherNid, 
								(int)otherTid, (int)msgLen, (int)msgTag, (int)comm);
			/* the args are user, time, source nid (my), source tid (my), dest nid (other), dest
			 * tid (other), size, tag */
				}
				else
				{ /* Check if it is a message receive operation */
					if (eventDescr.getTag() == TAU_MESSAGE_RECV_EVENT)
					{/* See RtsLayer::TraceSendMsg for documentation on the bit patterns of "parameter" */
						long xpar = parameter;
						/* extract the information from the parameter */
						msgTag   = ((xpar>>16) & 0x000000FF) | (((xpar >> 48) & 0xFF) << 8);
						otherNid = ((xpar>>24) & 0x000000FF) | (((xpar >> 56) & 0xFF) << 8);
						msgLen   = xpar & 0x0000FFFF | (xpar << 22 >> 54 << 16);
						long comm = xpar << 16 >> 58;

						/* If the application is multithreaded, insert call for matching sends/recvs here */
						otherTid = 0;
						//if (callbacks.RecvMessage!=null) 
							callbacks.recvMessage(userData, ts, (int)otherNid, 
									(int)otherTid, nid, tid, (int)msgLen, (int)msgTag, (int)comm);
						/* the args are user, time, source nid (my), source tid (my), dest nid (other), dest
						 * tid (other), size, tag */
					}
				}
			}
			if ((parameter == 0) && (eventDescr.getEventName()!= null) &&(eventDescr.getEventName().equals("\"FLUSH_CLOSE\""))) {
				/* reset the flag in NidTidMap to 0 (from 1) */
				NidTidMap.put(nidtid,zero);
				/* setting this flag to 0 tells us that a flush close has taken place 
				 * on this node, thread */
			} 
			else 
			{/* see if it is a WALL_CLOCK record */
				if ((parameter != 1) && (parameter != -1) && (eventDescr.getEventName() != null) 
						&& (eventDescr.getEventName().equals("\"WALL_CLOCK\""))) {
			/* ok, it is a wallclock event alright. But is it the *last* wallclock event?
			 * We can confirm that it is if the NidTidMap flag has been set to 0 by a 
			 * previous FLUSH_CLOSE call */
			
					if (NidTidMap.containsKey(nidtid))
					{/*printf("LAST WALL_CLOCK! End of trace file detected \n");*/
						/* see if an end of the trace callback is registered and 
						 * if it is, invoke it.*/
					if(NidTidMap.get(nidtid).equals(zero))//if (callbacks.EndTrace!=null) 
							callbacks.endTrace(userData, nid, tid);
						//System.out.println("Wallclock at "+ts);
					}
				}
			} /* is it a WALL_CLOCK record? */      
		} /* cycle through all records */
		/* return the number of event records read */
		return recordsRead;
	}

	/* close a trace file */
	public void closeTrace()
	{
		
		try {
			Fiid.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}	
}
