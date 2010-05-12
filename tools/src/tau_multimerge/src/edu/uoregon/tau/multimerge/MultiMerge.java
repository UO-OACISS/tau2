package edu.uoregon.tau.multimerge;

import java.awt.Point;
import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.uoregon.tau.trace.TraceReader;
import edu.uoregon.tau.trace.TraceReaderCallbacks;
import edu.uoregon.tau.trace.TraceWriter;

/**
 * Contains the information relevant to a single thread of execution as contained in a single unmerged trace file
 * @author wspear
 *
 */
class TotID{
	/**
	 * The name of the trace file
	 */
	public String filename;
	/**
	 * The id of the thread as contained in the trace file
	 */
	public int thread;
	/**
	 * The id of the node as contained in the trace file
	 */
	public int node;

	/**
	 * The map from the local state id (the id in this trace file) to the global state id (the id used in the merged trace)
	 */
	public Map<Integer, Integer> locToGlobStates;

	//	/**
	//	 * The number of send events seen on this thread.  Used to detect communication triplets used by CUDA traces
	//	 */
	//	public int sSeen=0;

	/**
	 * The number of CUDA communication use events seen on this thread.  Used to detect communication triplets used by CUDA traces
	 */
	//public boolean cSeen=false;
	//public boolean send=true;
	public int oneSideType=-1;
	//public static final int ONESIDE_SEND=0;
	//public static final int ONESIDE_RECV=1;
	public int size=0;
	
	public long offset=0;

	//	/**
	//	 * Most recent valid CUDA mem-copy size, set in the second of 3 communication triplets.
	//	 */
	//	int truSize=0;

	//	/**
	//	 * The unique CUDA communication id composed of the first and last tags of the first and last triplets for communication between two threads
	//	 */
	//	StringBuilder uniC=new StringBuilder();//null;

	DoublePair dp=new DoublePair(-1,-1);

	//	/**
	//	 * The unique CUDA communication id composed of the first and last tags of the first and last triplets for communication between two threads, on the sending side
	//	 */
	//	StringBuilder uniS=new StringBuilder();//null;

	/**
	 * The combination of filename and timestamp for the current communication.  If is not CUDA communication it will be stored in a global list for quick differentiation
	 */
	StringBuilder mpi=new StringBuilder();//null;

	//public int cudaSendID=-1;
	//public int cudaRecvID=-1;
	//public int cudaSizeID=-1;

	TotID(String fname){
		filename=fname;
		locToGlobStates=new HashMap<Integer, Integer>();
		thread=-1;
		node=-1;
	}
}

/**
 * When comparing two tracefile names it will order by node and threads under the node, with CUDA traces under the same node after cpu threads in that node.
 * @author wspear
 *
 */
class TraceNameComparitor implements Comparator<String>{

	public int compare(String o1, String o2) {

		if(o1.equals(o2))
			return 0;

		int[] a1 = MultiMerge.getNCT(o1);
		int[] a2 = MultiMerge.getNCT(o2);

		if(a1[0]<a2[0])
			return -1;
		if(a1[0]>a2[0])
			return 1;


		if(o1.contains("cuda")&&!o2.contains("cuda"))
			return 1;
		if(!o1.contains("cuda")&&o2.contains("cuda"))
			return -1;

		return compareNCT(a1,a2);
	}

	/**
	 * Given arrays of ints n/c/t a and n/c/t b decides which comes first, with node taking precedence over context and context over thread.
	 * @param a
	 * @param b
	 * @return
	 */
	private int compareNCT(int[] a, int[] b){

		if(a.equals(b)||(a[0]==b[0]&&a[1]==b[1]&&a[2]==b[2])){
			return 0;
		}
		if(a[0]<b[0])
			return -1;
		if(a[0]>b[0])
			return 1;
		if(a[1]<b[1])
			return -1;
		if(a[1]>b[1])
			return 1;
		if(a[2]<b[2])
			return -1;
		if(a[2]>b[2])
			return 1;

		System.out.println("Failed to compare NCT");
		return 0;
	}

}

/**
 * Accepts only tau and taucuda trace files.
 * @author wspear
 *
 */
class TraceFilter implements FilenameFilter{

	public boolean accept(File arg0, String arg1) {
		if((arg1.startsWith("tautrace.")||arg1.startsWith("taucudatrace."))&&arg1.endsWith(".trc"))
			return true;
		return false;
	}

}

class DoublePair{
	public double l1;
	public double l2;
	public DoublePair(double la, double lb){
		l1=la;
		l2=lb;
	}
	
	public DoublePair(DoublePair orig){
		l1=orig.l1;
		l2=orig.l2;
	}
	
	public boolean equals(Object o){
		DoublePair dp=(DoublePair)o;
		return(dp!=null&&this.l1==dp.l1&&this.l2==dp.l2);
	}
	public String toString(){
		return l1+"."+l2;
	}
	public int hashCode(){
		System.out.println(this.toString().hashCode());
		return this.toString().hashCode();
	}
}

public class MultiMerge {

	private static final int ONESIDED_MESSAGE_SEND=70000;
	private static final int ONESIDED_MESSAGE_RECV=70001;
	private static final int ONESIDED_MESSAGE_ID_TriggerValueT1=70002;
	private static final int ONESIDED_MESSAGE_ID_TriggerValueT2=70003;
	
	
	/**
	 * The map between state ids provided by different threads, and the global id used in the merged trace
	 */
	static Map<String,Integer> stateMap;
	/**
	 * The map between user event ids provided by different threads, and the global ids used in the merged trace
	 */
	static Map<String,Integer> ueMap;
	/**
	 * Maps the name of the trace file to the number of the thread that will be used in the merged trace, allowing for adding CUDA threads as interleaved nodes
	 */
	static Map<String, Integer> threadMap;
	/**
	 * Keeps track of the number of state and user-events seen so far.  The global ids are set by the current value (ids are sequential in the order seen)
	 */
	static int numStates=0;
	/**
	 * Keeps track of the last global node id recorded.  This allows the global node ids to be generated sequentially.
	 */
	static int lastNode=-1;
	/**
	 * The map from the node id associated with the old trace file to the new global node id that will be used in the merged trace
	 */
	static Map<Integer,Integer> oldNewDest;

	/**
	 * The map from the unique cuda communication id string to the point (node/thread id pair) that is associated with that unique id
	 */
	static Map<String,Point> idNodes;

	//	/**
	//	 * This set contains all filename/timestamp combinations that are associated with mpi communication events.  If the string is not in here then it is cuda communication.
	//	 */
	//	static Set<String> mpiCom;

	/**
	 * The array of TotID objects, index-paired with the list of trace files
	 */
	static TotID[] totIDs;

	/**
	 * The tau trace writer object which will write the merged trace
	 */
	static TraceWriter tw;

	private static boolean synch = false;
	
	/**
	 * Given an array of timestamps returns the index of the smallest timestamp greater than or equal to 0;
	 * @param times an array of timestamps
	 * @return
	 */
	static int minTime(long[] times){
		int least=-1;
		long min=-1;

		for(int i=0;i<times.length;i++){
			if(times[i]>=0){
				min=times[i];
				break;
			}
		}
		for(int i=0; i<times.length;i++){
			if(times[i]>=0&&times[i]<=min){
				min=times[i];
				least=i;
			}
		}
		return least;
	}

	/**
	 * Given a tracefile name returns the node/context/thread values included in the name as an integer array.
	 * @param name
	 * @return
	 */
	static int[] getNCT(String name){

		int[] out ={-1,-1,-1};
		if(name!=null&&name.length()>0)
		{
			String[] a =name.split("\\.");
			if(a.length==5){
			//In some cases the 'node' id contains invalid characters so it must be parsed one character at a time
				if(a[0].contains("cuda")){
					//Pattern p = Pattern.compile("([^\\d]+)(\\d+)(.*)");
					//Matcher m = p.matcher(a[1]);
					String m = "";
					//Char c=null;
					for(int i=0;i<a[1].length();i++){
						char c = a[1].charAt(i);
						if(Character.isDigit(c)){
							m+=c;
						}
					}
					out[0]=Integer.parseInt(m);
					//out[1]=Integer.parseInt(p.matcher(a[2]).group());
					//out[2]=Integer.parseInt(p.matcher(a[3]).group());
				}else{
				out[0]=Integer.parseInt(a[1]);
				//out[1]=Integer.parseInt(a[2]);
				//out[2]=Integer.parseInt(a[3]);
				}
				out[1]=Integer.parseInt(a[2]);
				out[2]=Integer.parseInt(a[3]);
			}
		}

		return out;
	}

	/**
	 * Given the name of a trace file returns the name of the associated .edf file
	 * @param trcName
	 * @return
	 */
	static String getEDFName(String trcName){
		String newname=null;
		String[] a =trcName.split("\\.");
		if(trcName.contains("cuda"))
		{
			newname = "taucudaevents."+a[1]+"."+a[2]+"."+a[3]+".edf";
		}
		else{
			newname="events."+a[1]+".edf";
		}


		return newname;
	}

	/**
	 * Does extra initialization on the given TotID object
	 * @param tot
	 * @return
	 */
	private static TotID initTotLoc(TotID tot){

		//boolean isCuda=tot.filename.contains("cuda");

		int[]fn=getNCT(tot.filename);
		int threadToken=-1;
//		if(isCuda)
//		{
//			threadToken=0;
//		}else{
			threadToken=fn[2];
		//}

		/*
		 * If this is a threaded trace file we are on the same node as before, just update the thread
		 */
		/*if(threadToken>0){
			tot.node=lastNode;
			tot.thread=threadToken;
		}*/
		/*
		 * If this is not a threaded entry we're on a new node so increment and set thread to 0.  If this is not a cuda tracefile we need to map the local node ID to the global nodeID.
		 */
		//else{
			lastNode++;
			tot.node=lastNode;
			tot.thread=0;
			if(threadToken==0){
				oldNewDest.put(new Integer(fn[0]), new Integer(lastNode));
			//}	
		}


		/*
		 * Identify the threads based on the n/c/t in the file names.
		 */
		if(threadToken==1){//!tot.filename.startsWith("tautrace")){

			String threadName="Node"+fn[0]+" CUDA Stream "+fn[1]+" Device "+fn[2];
			tw.defThread(tot.node, tot.thread, threadName);
		}else{
			String threadName="Node"+fn[0]+" Thread "+fn[2];
			tw.defThread(tot.node, tot.thread, threadName);
		}

		return tot;
	}

	/**
	 * Returns a sorted list of all trace files in the current directory
	 * @return
	 */
	private static List<String> listTraces(){
		List<String> traces = new ArrayList<String>();
		File curDir=new File(".");
		File[] tFiles=curDir.listFiles(new TraceFilter());

		for(int i=0;i<tFiles.length;i++){
			traces.add(tFiles[i].getName());
		}
		Collections.sort(traces, new TraceNameComparitor());
		return traces;
	}

	/**
	 * Performs the event initializations and reads all communication events to determine which are MPI vs. CUDA and which CUDA events communicate between which processes
	 * @param traces
	 */
	private static void initializeMerge(List<String> traces){

		TraceReader[] initReaders = new TraceReader[traces.size()];
		totIDs=new TotID[traces.size()];
		TraceReaderCallbacks init_cb = new TAUReaderInit();
		int recs_read;
		for(int rs=0;rs<initReaders.length;rs++)
		{	
			String edf = getEDFName(traces.get(rs));
			initReaders[rs]=new TraceReader(traces.get(rs),edf);
			initReaders[rs].setDefsOnly(false);
			initReaders[rs].setSubtractFirstTimestamp(false);
			TotID t = new TotID(initReaders[rs].getTraceFile());
			t=initTotLoc(t);
			totIDs[rs]=t;
			recs_read=0;
			do{
				recs_read=initReaders[rs].readNumEvents(init_cb, -1,t);//1024
			}while(recs_read!=0&&!initReaders[rs].isDone());
			if(initReaders[rs]!=null)
			{
				initReaders[rs].closeTrace();
			}else System.out.println("Warning: Tried to close null trace");
			initReaders[rs]=null;
		}
		if(synch){
		long base=0;
		for(int i=0;i<totIDs.length;i++){
			if(i==0){
				base=totIDs[i].offset;
				totIDs[i].offset=0;
			}
			else{
				totIDs[i].offset=base-totIDs[i].offset;
			}
		}
		}
	}

	/**
	 * Merges the data in the provided trace files into a single trace.
	 * @param traces
	 */
	private static void dataMerge(List<String> traces){
		TraceReader[] readers = new TraceReader[traces.size()];
		long[] sorter = new long[readers.length];
		TraceReaderCallbacks read_cb = new TAUReaderWriteall();
		long totalRecords=0;

		/*
		 * Create one reader for each trace file
		 */
		for(int rs=0;rs<readers.length;rs++)
		{
			readers[rs]=new TraceReader(traces.get(rs),getEDFName(traces.get(rs)));
			readers[rs].setDefsOnly(false);
			readers[rs].setSubtractFirstTimestamp(false);//TODO: Why is this needed only for cuda output?
			totalRecords+=readers[rs].getNumRecords();
			sorter[rs]=readers[rs].peekTime();
			if(synch){sorter[rs]+=totIDs[rs].offset;}
		}

		System.out.println(totalRecords+" records to merge.");
		long stepsize=totalRecords/50;
		long countRecords=0;
		if(stepsize==0){
			stepsize=1;
		}

		/*
		 * While there are records left in any trace write out the record with the lowest timestamp to the merged trace
		 */
		int minDex=minTime(sorter);
		while(minDex>=0){
			int read=readers[minDex].readNumEvents(read_cb, 1, totIDs[minDex]);
			if(read==0){
				sorter[minDex]=-1;
				readers[minDex].closeTrace();
				readers[minDex]=null;
			}else{

				countRecords++;
				if(countRecords%stepsize==0){
					System.out.println(countRecords+" Records read. "+(int)(100*((double)countRecords/(double)totalRecords))+"% converted");
				}

				sorter[minDex]=readers[minDex].peekTime();
				if(synch){sorter[minDex]+=totIDs[minDex].offset;}
				if(sorter[minDex]==-1)
				{
					readers[minDex].closeTrace();
					readers[minDex]=null;
				}
			}

			minDex=minTime(sorter);
		}

		for(int rs=0;rs<readers.length;rs++){
			if(readers[rs]!=null)
				readers[rs].closeTrace();
		}
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		//mpiCom=new HashSet<String>();
		oldNewDest=new HashMap<Integer,Integer>();
		idNodes=new HashMap<String,Point>();
		stateMap=new HashMap<String,Integer>();
		ueMap=new HashMap<String,Integer>();
		tw = new TraceWriter("tau.trc", "tau.edf");

		List<String> traces = listTraces();

		initializeMerge(traces);
		System.out.println("Initilization complete");
		dataMerge(traces);

		tw.closeTrace();
		System.out.println("The merging is complete.");
	}

	/**
	 * Callbacks for getting event initializations and setting up cuda communication events
	 * @author wspear
	 *
	 */
	private static class TAUReaderInit implements TraceReaderCallbacks{

		public int defClkPeriod(Object userData, double clkPeriod) {
			tw.defClkPeriod(clkPeriod);
			return 0;
		}

		public int defThread(Object userData, int nodeToken, int threadToken, String threadName){
			return 0;
		}

		public int defStateGroup(Object userData, int stateGroupToken, String stateGroupName){
			tw.defStateGroup(stateGroupName, stateGroupToken);
			return 0;
		}

		public int defState(Object userData, int stateToken, String stateName, int stateGroupToken){
			/*
			 * See if we've already initialized this event
			 */
			Integer globstate=stateMap.get(stateName);

			/*
			 * If not write a new state definition, using the number of states seen so far as an ID.  
			 * Map this state name to a global state ID used in the merged trace
			 */
			if(globstate==null){
				tw.defState(numStates, stateName, stateGroupToken);
				globstate=new Integer(numStates);
				stateMap.put(stateName, globstate);
				numStates++;
			}

			/*
			 * Map this local state ID for this thread to the global id used in the merged trace
			 */
			TotID tot = (TotID)userData;
			tot.locToGlobStates.put(new Integer(stateToken), globstate);

			return 0;
		}

		public int defUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing){

			TotID tot = (TotID)userData;

			/*if(userEventName.equals("TAUCUDA_MEM_SEND")){
				tot.cudaSendID=userEventToken;
				return 0;
			} 
			if(userEventName.equals("TAUCUDA_MEM_RCV")){
				tot.cudaRecvID=userEventToken;
				return 0;
			} 
			if(userEventName.equals("TAUCUDA_COPY_MEM_SIZE")){
				tot.cudaSizeID=userEventToken;
				return 0;
			} */
			
			if(userEventToken>=70000)
				return 0;


			/*
			 * As for defState above.  Note that states and user events use the same pool for unique global ids but different maps.  This may be unnecessary.
			 */
			Integer globevts=ueMap.get(userEventName);

			if(globevts==null){

				tw.defUserEvent(numStates, userEventName, monotonicallyIncreasing);
				globevts=new Integer(numStates);
				ueMap.put(userEventName, globevts);
				numStates++;
			}

			/*
			 * Map this local state ID for this thread to the global id used in the merged trace
			 */

			tot.locToGlobStates.put(new Integer(userEventToken), globevts);

			return 0;
		}

		public int enterState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			return 0;
		}
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			TotID tot = (TotID)userData;
			tot.offset=time;
			return 0;
		}

		/*Message registration.  (Message sending is defined in TAUReader below)*/
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){

			return 0;}

		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom) {
			return 0;
		}

		public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken, double userEventValue) {
			TotID tot = (TotID)userData;

			if(userEventToken==ONESIDED_MESSAGE_ID_TriggerValueT1){
				tot.dp.l1=userEventValue;
			}
			
			if(userEventToken==ONESIDED_MESSAGE_ID_TriggerValueT2){
				tot.dp.l2=userEventValue;
				
				Point p =idNodes.get(tot.dp.toString());
				if(p==null){
					p=new Point(tot.node,-1);
					idNodes.put(tot.dp.toString(), p);
				}else if(p.x!=tot.node){
					p.y=tot.node;
				}
			}
			
			/*
			if(userEventToken==ONESIDED_MESSAGE_SEND||userEventToken==ONESIDED_MESSAGE_RECV)
			{

				if(!tot.cSeen){

					//tot.uniC=commID1(tot.uniR,sourceNodeToken,messageSize,messageTag,messageCom);
					tot.dp.l1=userEventValue;
					tot.cSeen=true;
					//System.out.println(userEventToken);
					return 0;
				}
				else{
					tot.cSeen=false;
					//tot.uniR=commID2(tot.uniR,sourceNodeToken,messageSize,messageTag,messageCom);
					tot.dp.l2=userEventValue;
					//System.out.println(tot.dp);
					Point p =idNodes.get(tot.dp.toString());
					if(p==null){
						p=new Point(tot.node,-1);
						idNodes.put(tot.dp.toString(), p);
					}else if(p.x!=tot.node){
						p.y=tot.node;
					}
					
					//System.out.println(" tot "+p);
					
//					if(p.x==tot.node||p.y==tot.node){
//
//					}
//
//					if(p.y!=-1){
//						System.out.println("Warning, doubling up node identifiers!");
//
//					}
					//Does Uni need to be duplicated?
					//tot.uniC.setLength(0);

				}
			}*/

			return 0;
		}

		public int endTrace(Object userData, int nodeToken, int threadToken){
			return 0;
		}
	}


	/**
	 * Callbacks for performing writes of the merged trace file
	 * @author wspear
	 *
	 */
	private static class TAUReaderWriteall implements TraceReaderCallbacks{

		public int defClkPeriod(Object userData, double clkPeriod) {
			return 0;
		}

		public int defThread(Object userData, int nodeToken, int threadToken, String threadName){
			return 0;
		}

		public int defStateGroup(Object userData, int stateGroupToken, String stateGroupName){
			return 0;
		}

		public int defState(Object userData, int stateToken, String stateName, int stateGroupToken){
			return 0;
		}

		public int defUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing){
			return 0;
		}

		public int enterState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}
			int actualID=tot.locToGlobStates.get(new Integer(stateToken)).intValue();

			tw.enterState(time, tot.node, tot.thread, actualID);
			return 0;
		}
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}
			tw.leaveState(time, tot.node, tot.thread, tot.locToGlobStates.get(new Integer(stateToken)).intValue());
			return 0;
		}

		/*Message registration.  (Message sending is defined in TAUReader below)*/
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){

			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}

			Integer transNode=oldNewDest.get(new Integer(destinationNodeToken));
			if(transNode==null)
			{
				System.out.println("Bad Dest Node ID: "+destinationNodeToken);
				return 0;
			}

			tw.sendMessage(time, tot.node, tot.thread, transNode.intValue(), destinationThreadToken, messageSize, messageTag, messageComm);

			return 0;
		}



		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom) {
			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}


			Integer transNode=oldNewDest.get(new Integer(sourceNodeToken));
			if(transNode==null)
			{
				System.out.println("Bad Source Node ID: "+sourceNodeToken);
				return 0;
			}
			tw.recvMessage(time, transNode.intValue(), sourceThreadToken, tot.node, tot.thread, messageSize, messageTag, messageCom);

			return 0;
		}


		public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken, double userEventValue) {
			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}
			if(userEventToken==ONESIDED_MESSAGE_SEND||userEventToken==ONESIDED_MESSAGE_RECV){
				tot.oneSideType=userEventToken;
				tot.size=(int)userEventValue;
				return 0;
			}
			if(userEventToken==ONESIDED_MESSAGE_ID_TriggerValueT1)
			{
				tot.dp.l1=userEventValue;
				return 0;
			}
			if(userEventToken==ONESIDED_MESSAGE_ID_TriggerValueT2)
			{
				tot.dp.l2=userEventValue;
				Point p = idNodes.get(tot.dp.toString());
				if(p==null){
					System.out.println("Bad Recv: "+tot.dp);
					return 0;
				}

				if(p.x!=tot.node&&p.y!=tot.node){
					System.out.println(tot.dp+" not for "+tot.node);
				}
				//tot.uniR.setLength(0);
				int remote;
				if(p.x==tot.node)
				{
					remote=p.y;
				}else{
					remote=p.x;
				}
				if(tot.oneSideType==ONESIDED_MESSAGE_SEND)
				{
					tw.sendMessage(time,  tot.node, tot.thread, remote, 0, (int)userEventValue, 0, 0);
				}
				else{
					tw.recvMessage(time,  remote, 0, tot.node, tot.thread, (int)userEventValue, 0, 0);
				}

				return 0;
			}

			if(tot.thread>0)
			{
				System.out.println("Event from node!");
			}

			tw.eventTrigger(time, tot.node, tot.thread, tot.locToGlobStates.get(new Integer(userEventToken)).intValue(), (long)userEventValue);
			return 0;}

		public int endTrace(Object userData, int nodeToken, int threadToken){
			System.out.println("ENDED!");
			return 0;
		}
	}
}
