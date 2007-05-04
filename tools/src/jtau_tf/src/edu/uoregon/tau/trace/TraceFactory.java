package edu.uoregon.tau.trace;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.TreeMap;

public class TraceFactory {
	
	private final static int PCXX_EV_INIT = 60000;
	private final static int PCXX_EV_CLOSE = 60003;
	private final static int PCXX_EV_WALL_CLOCK = 60005;
	private final static int TAU_MESSAGE_SEND	= 60007;
	private final static int TAU_MESSAGE_RECV	= 60008;
	
	//private static final int TAU_MAX_RECORDS = 64*1024;
	
	
	/* open a trace file for reading */
	public static TraceReader OpenFileForInput( String name, String edf){
		TraceReader tFile= new TraceReader();
		try {	  
			tFile.subtractFirstTimestamp = true;
			tFile.nonBlocking = false;

			//tFile.intBB = ByteBuffer.allocate(4);
			//tFile.charBB = ByteBuffer.allocate(2);
			//tFile.longBB = ByteBuffer.allocate(8);
			//tFile.longBB.order( ByteOrder.LITTLE_ENDIAN );
			
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
			//tFile.NidTidMap=new HashMap();
			/* Allocate space for event id map */
			tFile.EventIdMap = new HashMap();
			/* Allocate space for group id map */
			tFile.GroupIdMap = new HashMap();
			/* initialize clock */
			tFile.ClkInitialized = false;
			/* initialize the first timestamp for the trace */
			//tFile.FirstTimestamp = 0;
			/* determine the format */
			//determineFormat (tFile);
			/* return file handle */
			tFile.format=TraceReader.determineFormat(tFile.Fiid);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return tFile;
	}
	
	
	public static TraceWriter OpenFileForOutput( String name, String edf){
		FileOutputStream ostream;
		try {
			ostream = new FileOutputStream(name);

		BufferedOutputStream bw = new BufferedOutputStream(ostream);
		DataOutputStream p = new DataOutputStream(bw);

		TraceWriter tFile = new TraceWriter();
		//tFile.traceBuffer = new Event[TAU_MAX_RECORDS];
		//tFile.traceBuffer[0]=new Event();
		//tFile.tracePosition = 0; // 0 will be the EV_INIT record
		//tFile.initialized = false;

		tFile.Foid=p;


		BufferedWriter out = new BufferedWriter(new FileWriter(edf));
		out.close();

		/* make a copy of the EDF file name */
		tFile.EdfFile = edf;

		//tFile.NidTidMap = new HashMap();

		/* Allocate space for maps */
		tFile.EventIdMap = new TreeMap();//Tree map for output ordered by id.

		tFile.GroupIdMap = new HashMap();

		//tFile.groupNameMap = new HashMap();

		//tFile.needsEdfFlush = true;

		/* initialize clock */
		//tFile.ClkInitialized = false;
		
		/* initialize the first timestamp for the trace */
		//tFile.FirstTimestamp = 0;

		/* define some events */

		EventDescr newEventDesc = new EventDescr(PCXX_EV_INIT,"TRACER","EV_INIT",0,"none");
		/*newEventDesc.Eid = PCXX_EV_INIT;
		newEventDesc.Group = "TRACER";
		newEventDesc.EventName = "EV_INIT";
		newEventDesc.Tag = 0;
		newEventDesc.Param = "none";*/
		tFile.EventIdMap.put(new Integer(PCXX_EV_INIT),newEventDesc);//[PCXX_EV_INIT] = newEventDesc;
		
		newEventDesc = new EventDescr(PCXX_EV_CLOSE,"TRACER","FLUSH_CLOSE",0,"none");
		/*newEventDesc.Eid = PCXX_EV_CLOSE;
		newEventDesc.Group = "TRACER";
		newEventDesc.EventName = "FLUSH_CLOSE";
		newEventDesc.Tag = 0;
		newEventDesc.Param = "none";*/
		tFile.EventIdMap.put(new Integer(PCXX_EV_CLOSE),newEventDesc);//[PCXX_EV_CLOSE] = newEventDesc;
		
		newEventDesc = new EventDescr(PCXX_EV_WALL_CLOCK,"TRACER","WALL_CLOCK",0,"none");
		/*newEventDesc.Eid = PCXX_EV_WALL_CLOCK;
		newEventDesc.Group = "TRACER";
		newEventDesc.EventName = "WALL_CLOCK";
		newEventDesc.Tag = 0;
		newEventDesc.Param = "none";*/
		tFile.EventIdMap.put(new Integer(PCXX_EV_WALL_CLOCK),newEventDesc);//[PCXX_EV_WALL_CLOCK] = newEventDesc;
		
		newEventDesc = new EventDescr(TAU_MESSAGE_SEND,"TAU_MESSAGE","MESSAGE_SEND",-7,"par");
		/*newEventDesc.Eid = TAU_MESSAGE_SEND;
		newEventDesc.Group = "TAU_MESSAGE";
		newEventDesc.EventName = "MESSAGE_SEND";
		newEventDesc.Tag = -7;
		newEventDesc.Param = "par";*/
		
		tFile.EventIdMap.put(new Integer(TAU_MESSAGE_SEND),newEventDesc);//[TAU_MESSAGE_SEND] = newEventDesc;
		newEventDesc = new EventDescr(TAU_MESSAGE_RECV,"TAU_MESSAGE","MESSAGE_RECV",-8,"par");
		/*newEventDesc.Eid = TAU_MESSAGE_RECV;
		newEventDesc.Group = "TAU_MESSAGE";
		newEventDesc.EventName = "MESSAGE_RECV";
		newEventDesc.Tag = -8;
		newEventDesc.Param = "par";*/
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
}
