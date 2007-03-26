/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

package edu.uoregon.tau.tau_tf;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;

public class TAU_tf_reader {
	
	public interface Ttf_DefClkPeriod{
		public int DefClkPeriod(Object userData, double clkPeriod);
	}
	
	public interface Ttf_DefThread{
		public int DefThread(Object userData, int nodeToken, int threadToken, String threadName);
	}
	
	public interface Ttf_DefStateGroup{
		public int DefStateGroup(Object userData, int stateGroupToken, String stateGroupName);
	}	
	
	public interface Ttf_DefState{
		public int DefState(Object userData, int stateToken, String stateName, int stateGoupToken);
	}	
	
	public interface Ttf_DefUserEvent{
		public int DefUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing);
	}
	
	public interface Ttf_EnterState{
		public int EnterState(Object userData, long time, int nodeToken, int threadToken, int stateToken);
	}		

	public interface Ttf_LeaveState{
		public int LeaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken);
	}	
	
	public interface Ttf_SendMessage{
		public int SendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom);
	}	
	
	public interface Ttf_RecvMessage{
		public int RecvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom);
	}
	
	public interface Ttf_EventTrigger{
		public int EventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken,
				double userEventValue);
	}		
	
	public interface Ttf_EndTrace{
		public int EndTrace(Object userData, int nodeToken, int threadToken);
	}	
	
	
	static int intReverseBytes(int value){
		//Integer.
		ByteBuffer bb = ByteBuffer.allocate(4);
		bb.putInt(value);
		bb.rewind();
		//ByteBuffer.wrap( new byte[]{(byte)(value >>> 24), (byte)(value >> 16 & 0xff), (byte)(value >> 8 & 0xff), (byte)(value & 0xff)} )	
		return bb.order( ByteOrder.LITTLE_ENDIAN ).getInt();// .getFloat();
	}
	
	static char charReverseBytes(char value){
		ByteBuffer bb = ByteBuffer.allocate(2);
		bb.putChar(value);
		bb.rewind();
		return bb.order( ByteOrder.LITTLE_ENDIAN ).getChar();// .getFloat();
	}
	
	static long longReverseBytes(long value){
		ByteBuffer bb = ByteBuffer.allocate(8);
		bb.putLong(value);
		bb.rewind();
		return bb.order( ByteOrder.LITTLE_ENDIAN ).getLong();// .getFloat();
	}
	
	/*public interface eventReader{
		public static final int ID=0;
	}*/
	static int TAU_BUFSIZE = 1024;
	static int TAU_MESSAGE_SEND_EVENT = -7;
	static int TAU_MESSAGE_RECV_EVENT = -8;
	static int FORMAT_NATIVE =0;//Java default
	static int FORMAT_32 = 1;
	static int FORMAT_32_SWAP = 2;
	static int FORMAT_64 =3;
	static int FORMAT_64_SWAP=4;
	/* for 32 bit platforms (24 bytes)*/
	
	static int readEvents(Ttf_file tFile, Event[] traceBuffer, int numread) throws IOException{
		int x=0;
		Event evt;// = new Event();
		int bytes=tFile.Fiid.available();
		int records=bytes/tFile.eventSize;
		
		//byte[] b = new byte[24];
		if(tFile.format<2)
			while(x<numread&&x<records){
				//tFile.Fid.read(b);
				//Integer.
				evt = new Event();
				//tFile.Fid.r
				evt.ev=tFile.Fiid.readInt();
				evt.nid=tFile.Fiid.readChar(); //.readChar();
				evt.tid=tFile.Fiid.readChar();//readChar();
				evt.par=tFile.Fiid.readLong();
				evt.ti=tFile.Fiid.readLong();
				traceBuffer[x]=evt;
				x++;
				//System.out.println("ID: "+x+" NID: "+c1+" TID: "+c2+" PAR: "+l1+" TID: "+l2);
			}
		else
		if(tFile.format==2)
		{
			while(x<numread&&x<records){
				evt = new Event();
				evt.ev=intReverseBytes(tFile.Fiid.readInt());
				evt.nid=charReverseBytes(tFile.Fiid.readChar());
				evt.tid=charReverseBytes(tFile.Fiid.readChar());
				evt.par=longReverseBytes(tFile.Fiid.readLong());
				evt.ti=longReverseBytes(tFile.Fiid.readLong());
				traceBuffer[x]=evt;
				x++;
				//System.out.println("ID: "+x+" NID: "+c1+" TID: "+c2+" PAR: "+l1+" TID: "+l2);
			}	
		}
		else
		if(tFile.format==3)
		{
			while(x<numread&&x<records){
				evt = new Event();
				evt.ev=(int)tFile.Fiid.readLong();
				evt.nid=tFile.Fiid.readChar();
				evt.tid=tFile.Fiid.readChar();
				tFile.Fiid.readInt();
				evt.par=tFile.Fiid.readLong();
				evt.ti=tFile.Fiid.readLong();
			
				//System.out.println("ID: "+evt.ev+" NID: "+(int)evt.nid+" TID: "+(int)evt.tid+" PAR: "+evt.par+" TIM: "+evt.ti);
			
				traceBuffer[x]=evt;
				x++;
			}
		}
		else
		if(tFile.format==4)
		{
			while(x<numread&&x<records){
				evt = new Event();
				evt.ev=(int)longReverseBytes(tFile.Fiid.readLong());
				evt.nid=charReverseBytes(tFile.Fiid.readChar());
				evt.tid=charReverseBytes(tFile.Fiid.readChar());
				tFile.Fiid.readInt();
				evt.par=longReverseBytes(tFile.Fiid.readLong());
				evt.ti=longReverseBytes(tFile.Fiid.readLong());
			
				//System.out.println("ID: "+evt.ev+" NID: "+(int)evt.nid+" TID: "+(int)evt.tid+" PAR: "+evt.par+" TIM: "+evt.ti);
			
				traceBuffer[x]=evt;
				x++;
			}			
		}
		return x;
	}
	
	static int event_GetEv(Ttf_file tFile, Event[] traceBuffer, int index){
		return (traceBuffer[index]).ev;
	}
	
	static int event_GetNid(Ttf_file tFile, Event[] traceBuffer, int index){
		return (traceBuffer[index]).nid;
	}
	
	static int event_GetTid(Ttf_file tFile, Event[] traceBuffer, int index){
		return (traceBuffer[index]).tid;
	}
	
	static long event_GetPar(Ttf_file tFile, Event[] traceBuffer, int index){
		return (traceBuffer[index]).par;
	}
	
	static long event_GetTi(Ttf_file tFile, Event[] traceBuffer, int index){
		return (traceBuffer[index]).ti;
	}
	
	static void determineFormat(Ttf_file tFile) throws IOException{
		Event evt = new Event();
		tFile.Fiid.mark(128);
		evt.ev=tFile.Fiid.readInt();
		evt.nid=tFile.Fiid.readChar();
		evt.tid=tFile.Fiid.readChar();
		evt.par=tFile.Fiid.readLong();
		evt.ti=tFile.Fiid.readLong();
		int format=FORMAT_NATIVE;
		if(evt.par==3)
		{
			format=FORMAT_32;
			tFile.eventSize=24;
			//System.out.println("Default!");
		}
		else
		if(longReverseBytes(evt.par)==3)
		{
			format=FORMAT_32_SWAP;
			tFile.eventSize=24;
			//System.out.println("Swapping!");
		}
		else{
			tFile.Fiid.reset();
			//Event64 evt64 = new Event64();
			evt.ev=(int)tFile.Fiid.readLong();
			evt.nid=tFile.Fiid.readChar();
			evt.tid=tFile.Fiid.readChar();
			tFile.Fiid.readInt();
			evt.par=tFile.Fiid.readLong();
			evt.ti=tFile.Fiid.readLong();
			if(evt.par==3)
			{
				format=FORMAT_64;
				tFile.eventSize=32;
				//System.out.println("64!");
			}
			else
			if(longReverseBytes(evt.par)==3)
			{
				format=FORMAT_64_SWAP;
				tFile.eventSize=32;
				//System.out.println("Swapping 64!");
			}
			else
			{
				System.out.println("Could not determine trace format, using native.");
				tFile.eventSize=24;
			}
		}
		tFile.format=format;
		tFile.Fiid.reset();
		return;
	}
	
	/* Look for an event in the event map */
	static boolean isEventIDRegistered(Ttf_file tFile, int event)
	{
		return tFile.EventIdMap.containsKey(new Integer(event));
	}	
	
	/* Event ID is not found in the event map. Re-read the event 
	 * description file */
	static boolean refreshTables(Ttf_file tFile, Ttf_Callbacks cb)
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
		edf = new BufferedReader(new FileReader(tFile.EdfFile));
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
				if (!isEventIDRegistered(tFile,localEventId))
				{
					/* couldn't locate the event id */
					/* fill an event description object */
					Ttf_EventDescr eventDescr = new Ttf_EventDescr();
					eventDescr.Eid = localEventId;
					eventDescr.EventName = new String(eventname);
					eventDescr.Group = new String(group);
					eventDescr.Tag = tag;
					eventDescr.Param = new String(param);
					tFile.EventIdMap.put(new Integer(localEventId),eventDescr); /* add it to the map */

					if (!tFile.GroupIdMap.containsKey(eventDescr.Group))
					{ 
						/* group id not found. Generate group id on the fly */
						groupid = tFile.GroupIdMap.size()+1;
						
						/* invoke group callback */
						/* check Param to see if its a user defined event */
						if (eventDescr.Param.equals("EntryExit"))
						{ /* it is not a user defined event */
							tFile.GroupIdMap.put(eventDescr.Group,new Integer(groupid));
							if (cb.DefStateGroup!=null)
								cb.DefStateGroup.DefStateGroup(cb.UserData, groupid, eventDescr.Group); 
						}
					}
					else
					{ /* retrieve the stored group id token */
						groupid = ((Integer)tFile.GroupIdMap.get(eventDescr.Group)).intValue();
					}
					/* invoke callback for registering a new state */
					if (eventDescr.Param.equals("TriggerValue"))
					{ /* it is a user defined event */
						if (cb.DefUserEvent!=null)
							cb.DefUserEvent.DefUserEvent(cb.UserData, localEventId, 
									eventDescr.EventName, eventDescr.Tag);
					}
					else if(eventDescr.Param.equals("EntryExit"))//(!eventDescr.Param.equals("TriggerValue"))//
					{ /* it is an entry/exit event */
						if (cb.DefState!=null)
							cb.DefState.DefState(cb.UserData, localEventId, eventDescr.EventName,groupid);
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
	
	
	/* open a trace file for reading */
	public static Ttf_file Ttf_OpenFileForInput( String name, String edf){
		Ttf_file tFile= new Ttf_file();
		try {	  
			tFile.subtractFirstTimestamp = true;
			tFile.nonBlocking = false;

			/* Open the trace file */
			FileInputStream istream;
			istream = new FileInputStream(name);
			BufferedInputStream bw = new BufferedInputStream(istream);
			DataInputStream p = new DataInputStream(bw);
			tFile.Fiid=p;

			BufferedReader in = new BufferedReader(new FileReader(edf));
			in.close();
			/* make a copy of the EDF file name */
			tFile.EdfFile = edf;
			/* Allocate space for nodeid, thread id map */
			tFile.NidTidMap=new HashMap();
			/* Allocate space for event id map */
			tFile.EventIdMap = new HashMap();
			/* Allocate space for group id map */
			tFile.GroupIdMap = new HashMap();
			/* initialize clock */
			tFile.ClkInitialized = false;
			/* initialize the first timestamp for the trace */
			tFile.FirstTimestamp = 0;
			/* determine the format */
			//determineFormat (tFile);
			/* return file handle */
			determineFormat(tFile);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return tFile;
	}

	public static void Ttf_SetSubtractFirstTimestamp( Ttf_file handle, boolean value ){
		handle.subtractFirstTimestamp=value;
	}
	
	public static void Ttf_SetNonBlocking( Ttf_file handle, boolean value ){
		handle.nonBlocking=value;
	}

	/* Seek to an absolute event position. 
	 * A negative position indicates to start from the tail of the event stream. 
	 * Returns the position if successful or 0 if an error occured */
	public static int Ttf_AbsSeek( Ttf_file handle, int eventPosition ){
		return 0;
	}

	/* seek to a event position relative to the current position (just for completeness!) 
	 * Returns the position if successful or 0 if an error occured */
	public static int Ttf_RelSeek( Ttf_file handle, int plusMinusNumEvents ){
		return 0;
	}

	/* read n events and call appropriate handlers.
	 * Returns the number of records read (can be 0).
	 * Returns a -1 value when an error takes place. Check errno */
	public static int Ttf_ReadNumEvents( Ttf_file tFile, Ttf_Callbacks callbacks, int numberOfEvents ){
		Event[] traceBuffer = new Event[TAU_BUFSIZE];
		int recordsRead=0, recordsToRead;
		long otherTid, otherNid, msgLen, msgTag;

		if (tFile == null)
			return 0; /* ERROR */

		/* How many bytes are to be read? */
		recordsToRead = numberOfEvents > TAU_BUFSIZE ? TAU_BUFSIZE : numberOfEvents;

		/* if clock needs to be initialized, initialize it */
		if (!tFile.ClkInitialized)
		{
			if (callbacks.DefClkPeriod != null)
				callbacks.DefClkPeriod.DefClkPeriod(callbacks.UserData, 1E-6);
			/* set flag to initialized */
			tFile.ClkInitialized = true; 

			/* Read the first record and check its timestamp 
			 * For this we need to lseek to the beginning of the file, read one 
			 * record and then lseek it back to where it was */
			tFile.Fiid.mark(64);
			try {
			readEvents(tFile, traceBuffer,1);
			
				tFile.Fiid.reset();
			} catch (IOException e) {
				e.printStackTrace();
			}
			tFile.FirstTimestamp = event_GetTi(tFile,traceBuffer,0);
		}//!clkinit

		/* Read n records and go through each event record */
		int read=0;
		try {
			read = readEvents(tFile,traceBuffer,recordsToRead);
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
			//convertEvent(tFile, traceBuffer, i);
			if (!isEventIDRegistered(tFile, event_GetEv(tFile, traceBuffer, i)))
			{
				/* if event id is not found in the event id map, read the EDF file */
				if (!refreshTables(tFile, callbacks))
				{ /* error */
					System.out.println("Refresh Tables Error");
					return -1;
				}
				if (!isEventIDRegistered(tFile, event_GetEv(tFile, traceBuffer, i)))
				{ /* even after reading the edf file, if we don't find the event id, 
				then there's an error */
					System.out.println("ID Reg error");
					return -1;
				}
				/* we did find the event id, process the trace file */
			}
		    /* event is OK. Examine each event and invoke callbacks for Entry/Exit/Node*/
		    /* first check nodeid, threadid */

			nid = event_GetNid(tFile, traceBuffer, i);//ints
			tid = event_GetTid(tFile, traceBuffer, i);
			nidtid=nid+":"+tid;
			
			if (!tFile.NidTidMap.containsKey(nidtid))
			{
				/* this pair of node and thread has not been encountered before*/
				nodename="process "+nidtid;//nid+":"+tid;
				/* invoke callback routine */
				if (callbacks.DefThread!=null)
					callbacks.DefThread.DefThread(callbacks.UserData, nid, tid, nodename);
				/* add it to the map! */
				tFile.NidTidMap.put(nidtid, one);
			}
		    /* check the event to see if it is entry or exit */
			long ts;
			if (tFile.subtractFirstTimestamp) {
				ts = event_GetTi(tFile, traceBuffer, i) - tFile.FirstTimestamp;
			} else {
				ts = event_GetTi(tFile, traceBuffer, i);
			}
			long parameter = event_GetPar(tFile, traceBuffer, i);
			/* Get param entry from EventIdMap */
			
			Ttf_EventDescr eventDescr = (Ttf_EventDescr)tFile.EventIdMap.get(
					new Integer(event_GetEv(tFile, traceBuffer, i)));
			//if(eventDescr==null)
			//System.out.println("DEF: "+event_GetEv(tFile,traceBuffer,i));
			//System.out.println("DEF:"+eventDescr.Eid+" "+eventDescr.EventName+" "+eventDescr.Group+" "+eventDescr.Param+" "+eventDescr.Tag);
			if ((eventDescr.Param != null) && ((eventDescr.Param.equals("EntryExit"))))
			{ /* entry/exit event */
				if (parameter == 1)
				{ /* entry event, invoke the callback routine */
					if (callbacks.EnterState!=null)
						callbacks.EnterState.EnterState(callbacks.UserData, ts, nid, 
								tid,event_GetEv(tFile, traceBuffer, i));
				}
				else
				{ if (parameter == -1)
					{ /* exit event */
						if (callbacks.LeaveState!=null)
							callbacks.LeaveState.LeaveState(callbacks.UserData,ts, nid, 
									tid,event_GetEv(tFile, traceBuffer, i));
					}
				}
			} /* entry exit events *//* add message passing events here */
			else 
			{
				if ((eventDescr.Param != null) && (eventDescr.Param.equals("TriggerValue")))
				{ /* User defined event */
					if (callbacks.EventTrigger!=null) {
						parameter = event_GetPar(tFile, traceBuffer, i);

						callbacks.EventTrigger.EventTrigger(callbacks.UserData, ts, nid, tid, 
								event_GetEv(tFile, traceBuffer, i), 
								parameter);
					}
				}
				if (eventDescr.Tag == TAU_MESSAGE_SEND_EVENT) 
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
					if (callbacks.SendMessage!=null) 
						callbacks.SendMessage.SendMessage(callbacks.UserData, ts, nid, tid, (int)otherNid, 
								(int)otherTid, (int)msgLen, (int)msgTag, (int)comm);
			/* the args are user, time, source nid (my), source tid (my), dest nid (other), dest
			 * tid (other), size, tag */
				}
				else
				{ /* Check if it is a message receive operation */
					if (eventDescr.Tag == TAU_MESSAGE_RECV_EVENT)
					{/* See RtsLayer::TraceSendMsg for documentation on the bit patterns of "parameter" */
						long xpar = parameter;
						/* extract the information from the parameter */
						msgTag   = ((xpar>>16) & 0x000000FF) | (((xpar >> 48) & 0xFF) << 8);
						otherNid = ((xpar>>24) & 0x000000FF) | (((xpar >> 56) & 0xFF) << 8);
						msgLen   = xpar & 0x0000FFFF | (xpar << 22 >> 54 << 16);
						long comm = xpar << 16 >> 58;

						/* If the application is multithreaded, insert call for matching sends/recvs here */
						otherTid = 0;
						if (callbacks.RecvMessage!=null) 
							callbacks.RecvMessage.RecvMessage(callbacks.UserData, ts, (int)otherNid, 
									(int)otherTid, nid, tid, (int)msgLen, (int)msgTag, (int)comm);
						/* the args are user, time, source nid (my), source tid (my), dest nid (other), dest
						 * tid (other), size, tag */
					}
				}
			}
			if ((parameter == 0) && (eventDescr.EventName != null) &&(eventDescr.EventName.equals("\"FLUSH_CLOSE\""))) {
				/* reset the flag in NidTidMap to 0 (from 1) */
				tFile.NidTidMap.put(nidtid,zero);
				/* setting this flag to 0 tells us that a flush close has taken place 
				 * on this node, thread */
			} else {
				/* see if it is a WALL_CLOCK record */
				if ((parameter != 1) && (parameter != -1) && (eventDescr.EventName != null) 
						&& (eventDescr.EventName.equals("\"WALL_CLOCK\""))) {
			/* ok, it is a wallclock event alright. But is it the *last* wallclock event?
			 * We can confirm that it is if the NidTidMap flag has been set to 0 by a 
			 * previous FLUSH_CLOSE call */
			
					if (tFile.NidTidMap.containsKey(nidtid))
					{/*printf("LAST WALL_CLOCK! End of trace file detected \n");*/
						/* see if an end of the trace callback is registered and 
						 * if it is, invoke it.*/
					if(tFile.NidTidMap.get(nidtid).equals(zero))
						if (callbacks.EndTrace!=null) 
							callbacks.EndTrace.EndTrace(callbacks.UserData, nid, tid);
					}
				}
			} /* is it a WALL_CLOCK record? */      
		} /* cycle through all records */
		/* return the number of event records read */
		return recordsRead;
	}

	/* close a trace file */
	public static Ttf_file Ttf_CloseFile( Ttf_file fileHandle )
	{
		if(fileHandle==null)
			return null;
		try {
			fileHandle.Fiid.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return fileHandle;
	}	
}
