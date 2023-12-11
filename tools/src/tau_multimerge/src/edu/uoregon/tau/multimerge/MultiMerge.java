package edu.uoregon.tau.multimerge;



import java.io.File;
import java.io.FilenameFilter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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
	 * The map from the local state id (the id in this trace file) to the global state id (the id used in the merged trace)
	 */
	public Map<Integer, Integer> locToGlobStates;

	/**
	 * The type observed for the one-sided triplet we're investigating
	 */
	public int oneSideType=-1;
	/**
	 * The identifying attributes (message tags etc) of the one sided triplet we're investigating
	 */
	DoublePair dp=new DoublePair(-1,-1,false);
	
	public int size=0;
	
	public long offset=0;


	


	/**
	 * The combination of filename and timestamp for the current communication.  If is not CUDA communication it will be stored in a global list for quick differentiation
	 */
	StringBuilder mpi=new StringBuilder();//null;


	TotID(String fname){
		filename=fname;
		locToGlobStates=new HashMap<Integer, Integer>();

	}
}

/**
 * When comparing two tracefile names it will order by node and threads under the node, with CUDA traces under the same node after cpu threads in that node.
 * @author wspear
 *
 */
class TraceNameComparitor implements Comparator<File>{

	public int compare(File of1, File of2) {

		String o1=of1.getName();
		String o2=of2.getName();
		
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
		if(arg1.startsWith("tautrace.")&&arg1.endsWith(".trc"))
			return true;
		return false;
	}

}

class DoublePair{
	public double l1;
	public double l2;
	//public boolean reciprocal=false;
	public DoublePair(double la, double lb,boolean reciprocal){
		l1=la;
		l2=lb;
		//this.reciprocal=reciprocal;
	}
	
	public DoublePair(DoublePair orig){
		l1=orig.l1;
		l2=orig.l2;
		//reciprocal=orig.reciprocal;
	}
	
	public boolean equals(Object o){
		DoublePair dp=(DoublePair)o;
		return(dp!=null&&this.l1==dp.l1&&this.l2==dp.l2);//&&this.reciprocal==dp.reciprocal
	}
	public String toString(){
		String match =l1+"."+l2;
//		if(reciprocal){
//			match=match+".R";
//		}
		return match;
	}
	public int hashCode(){
		System.out.println(this.toString().hashCode());
		return this.toString().hashCode();
	}
}


class ToFrom{
	public int toNode,toThread,fromNode,fromThread, direction;

	public boolean reciprocal;
	
	public ToFrom(int toNode, int toThread, int fromNode, int fromThread) {
		super();
		this.toNode = toNode;
		this.toThread = toThread;
		this.fromNode = fromNode;
		this.fromThread = fromThread;
		reciprocal=false;
	}
	
}

public class MultiMerge {
	
	private static boolean isReciprocal(int evt){
		if(evt==ONESIDED_MESSAGE_RECIPROCAL_SEND||evt==ONESIDED_MESSAGE_RECIPROCAL_RECV){
			return true;
		}
		return false;
	}

	private static final int ONESIDED_MESSAGE_SEND=70000;
	private static final int ONESIDED_MESSAGE_RECV=70001;
	private static final int ONESIDED_MESSAGE_UNKNOWN=70004;
	private static final int ONESIDED_MESSAGE_ID_TriggerValueT1=70002;
	private static final int ONESIDED_MESSAGE_ID_TriggerValueT2=70003;
	/**
	 * Reciprocal send must be matched with a reciprocal. A non-reciprocal may be matched with anything
	 */
	private static final int ONESIDED_MESSAGE_RECIPROCAL_SEND=70005;
	private static final int ONESIDED_MESSAGE_RECIPROCAL_RECV=70006;
	
	
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
	 * The map from the unique cuda communication id string to the point (node/thread id pair) that is associated with that unique id
	 */
	static Map<String,ToFrom> idNodes;


	/**
	 * The array of TotID objects, index-paired with the list of trace files
	 */
	static TotID[] totIDs;

	/**
	 * The tau trace writer object which will write the merged trace
	 */
	static TraceWriter tw;
	
	/**
	 * List of multiple directories of traces to merge into one.
	 */
	static List<File> multiDirs;

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
					String m = "";
					for(int i=0;i<a[1].length();i++){
						char c = a[1].charAt(i);
						if(Character.isDigit(c)){
							m+=c;
						}
					}
					out[0]=Integer.parseInt(m);
				}else{
				out[0]=Integer.parseInt(a[1]);
				}
				out[1]=Integer.parseInt(a[2]);
				out[2]=Integer.parseInt(a[3]);
			}
		}

		return out;
	}

	/**
	 * Given the name of a trace file returns the name of the associated .edf file
	 * @param trcFile
	 * @return
	 */
	static String getEDFName(File trcFile){
		String newname=null;
		String[] a =trcFile.getName().split("\\.");
		
		if(trcFile.getName().contains("cuda"))
		{
			newname = "taucudaevents."+a[1]+"."+a[2]+"."+a[3]+".edf";
		}
		else{
			newname="events."+a[1]+".edf";
		}
		newname=trcFile.getParentFile().getAbsolutePath()+File.separatorChar+newname;

		return newname;
	}

	/**
	 * Returns a sorted list of all trace files in the current directory
	 * @return
	 */
	private static List<File> listTraces(File f){
		List<File> traces = new ArrayList<File>();
		File curDir;
		if(f==null){
		curDir=new File(".");
		}
		else{
			curDir=f;
		}
		File[] tFiles=curDir.listFiles(new TraceFilter());

		for(int i=0;i<tFiles.length;i++){
			traces.add(tFiles[i]);
		}
		Collections.sort(traces, new TraceNameComparitor());
		return traces;
	}

	/**
	 * Counts all of the trace files in a list of lists of tracefiles
	 * @param traces
	 * @return
	 */
	private static int countTraces(List<List<File>> traces){
		int count = 0;
		for(List<File> sList:traces){
			count+=sList.size();
		}
		return count;
	}
	
	private static char numNid(List<File> traces){
		
		Set<Integer> counter=new HashSet<Integer>();
		
		for(File f:traces){
			String[] a =f.getName().split("\\.");
			counter.add(Integer.parseInt(a[1]));
		}
		
		return (char) counter.size();
	}
	
	/**
	 * Performs the event initializations and reads all communication events to determine which are MPI vs. CUDA and which CUDA events communicate between which processes
	 * @param traces
	 */
	private static void initializeMerge(List<List<File>> traces){

		int totalTraces=countTraces(traces);
		
		TraceReader[] initReaders = new TraceReader[totalTraces];
		totIDs=new TotID[totalTraces];
		TraceReaderCallbacks init_cb = new TAUReaderInit();
		int recs_read;
		char nid_offset=0;
		char rs=0;
		for(List<File> sList:traces)
		{
		for(File trace:sList)
		{	
			String edf = getEDFName(trace);
			//System.out.println("nid_offset: "+(int)nid_offset);
			initReaders[rs]=new TraceReader(trace.getAbsolutePath(),edf,nid_offset);
			initReaders[rs].setDefsOnly(false);
			initReaders[rs].setSubtractFirstTimestamp(false);
			TotID t = new TotID(initReaders[rs].getTraceFile());
			
//			t=initTotLoc(t);
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
			rs++;
		}
		nid_offset=(char) (numNid(sList)+nid_offset);
		//System.out.println("nidoffset-set: "+(int)nid_offset);
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
	private static void dataMerge(List<List<File>> traces){
		int totalTraces=countTraces(traces);
		TraceReader[] readers = new TraceReader[totalTraces];
		long[] sorter = new long[readers.length];
		TraceReaderCallbacks read_cb = new TAUReaderWriteall();
		long totalRecords=0;

		/*
		 * Create one reader for each trace file
		 */
		char rs=0;
		char nid_offset=0;
		for(List<File> sList:traces)
		{
		for(File trace:sList)
		{	
			readers[rs]=new TraceReader(trace.getAbsolutePath(),getEDFName(trace),nid_offset);
			readers[rs].setDefsOnly(false);
			readers[rs].setSubtractFirstTimestamp(false);//TODO: Why is this needed only for cuda output?
			totalRecords+=readers[rs].getNumRecords();
			sorter[rs]=readers[rs].peekTime();
			if(synch){sorter[rs]+=totIDs[rs].offset;}
			rs++;
		}
		nid_offset=(char) (numNid(sList)+nid_offset);
		System.out.println("nidoffset-set: "+(int)nid_offset);
		}

		if(!quiet)
		{
			System.out.println(totalRecords+" records to merge.");
		}
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
				if(!quiet&&countRecords%stepsize==0){
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

		for(int i=0;i<rs;i++){
			if(readers[i]!=null)
				readers[i].closeTrace();
		}
	}

	static boolean quiet = false;
	static boolean nocomm = false;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {

		//mpiCom=new HashSet<String>();
		//oldNewDest=new HashMap<Integer,Integer>();
		idNodes=new HashMap<String,ToFrom>();
		stateMap=new HashMap<String,Integer>();
		ueMap=new HashMap<String,Integer>();
		tw = new TraceWriter("tau.trc", "tau.edf");
		
		 for(int i=0;i<args.length;i++){
			if(args[i].equals("-q"))
			{quiet=true;
			continue;
			}
			if(args[i].equals("--nocomm"))
                        {nocomm=true;
                        continue;
                        }

			if(args[i].equals("--multidir")){
				multiDirs=new ArrayList<File>();
				while(i<args.length){
					if(i<args.length-1&&!args[i+1].startsWith("-")){
						i++;
					}
					else{
						break;
					}
					multiDirs.add(new File(args[i]));
				}
			}
			
		}
		 
		 

		 List<List<File>> traces=new ArrayList<List<File>>();
		 if(multiDirs==null)
		 {
			traces.add(listTraces(null));
		 }
		 else{
			 for(File f:multiDirs){
				 traces.add(listTraces(f));
			 }
		 }

		initializeMerge(traces);
		if(!quiet){
			System.out.println("Initilization complete");
		}
		dataMerge(traces);

		tw.closeTrace();
		if(!quiet){
			System.out.println("The merging is complete.");
		}
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
			tw.defThread(nodeToken, threadToken, threadName);
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
				globstate=Integer.valueOf(numStates);
				stateMap.put(stateName, globstate);
				numStates++;
			}

			/*
			 * Map this local state ID for this thread to the global id used in the merged trace
			 */
			TotID tot = (TotID)userData;
			tot.locToGlobStates.put(Integer.valueOf(stateToken), globstate);

			return 0;
		}

		public int defUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing){

			TotID tot = (TotID)userData;
			
			if(userEventToken>=70000)
				return 0;


			/*
			 * As for defState above.  Note that states and user events use the same pool for unique global ids but different maps.  This may be unnecessary.
			 */
			Integer globevts=ueMap.get(userEventName);

			if(globevts==null){

				tw.defUserEvent(numStates, userEventName, monotonicallyIncreasing);
				globevts=Integer.valueOf(numStates);
				ueMap.put(userEventName, globevts);
				numStates++;
			}

			/*
			 * Map this local state ID for this thread to the global id used in the merged trace
			 */

			tot.locToGlobStates.put(Integer.valueOf(userEventToken), globevts);

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
            if(nocomm)return 0;
			/**
			 * Set the message type (first of the 3 event triplet)
			 */
			if(userEventToken==ONESIDED_MESSAGE_SEND||userEventToken==ONESIDED_MESSAGE_RECV||userEventToken==ONESIDED_MESSAGE_UNKNOWN||userEventToken==ONESIDED_MESSAGE_RECIPROCAL_SEND||userEventToken==ONESIDED_MESSAGE_RECIPROCAL_RECV) {
				tot.oneSideType = userEventToken;	
//				if(userEventToken==ONESIDED_MESSAGE_RECIPROCAL_SEND||userEventToken==ONESIDED_MESSAGE_RECIPROCAL_RECV){
//					tot.dp.reciprocal=true;
//				}else{
//					tot.dp.reciprocal=false;
//				}
			  //System.out.println("eventTrigger, recording: " + userEventToken + " on thread: " + threadToken);
			}
			/**
			 * Set the message id part 1 (second of the 3 event triplet)
			 */
			else if(userEventToken==ONESIDED_MESSAGE_ID_TriggerValueT1){
				tot.dp.l1=userEventValue;
			}
			/**
			 * Set the message id part 2 (thrid of the 3 event triplet) and do processing to start or complete the link between the communication points
			 */
			else if(userEventToken==ONESIDED_MESSAGE_ID_TriggerValueT2){
				tot.dp.l2=userEventValue;
				linkOnesidedComm(tot.dp,nodeToken,threadToken,tot.oneSideType);
				
//				//If this message is *not* reciprocal than it could also match with a reciprocal. We need to do the association in this case as well.
//				if(!tot.dp.reciprocal){
//					DoublePair dpr = new DoublePair(tot.dp);
//					dpr.reciprocal=true;
//					linkOnesidedComm(dpr,nodeToken,threadToken,tot.oneSideType);
//				}
				
			}

			return 0;
		}
		
		/**
		 * Updates the map from communication id to an object indicating the source and destination threads for that communication
		 * @param dp DoublePair containing the communication tags, etc, needed to uniquely identify the communication
		 * @param nodeToken node where communication tags were observed
		 * @param threadToken thread where communication tags were observed
		 * @param direction event type of communication
		 */
		private static void linkOnesidedComm(DoublePair dp, int nodeToken, int threadToken, int direction){
			//See if this unique message id has been associated with a send/recieve location
			ToFrom p =idNodes.get(dp.toString());
			boolean reciprocal=isReciprocal(direction);
			if(p==null){
				//If not, create a new object to record the local side of the transaction and store it
				p=new ToFrom(nodeToken,threadToken,-1,-1);
				if(reciprocal){
					p.reciprocal=true;
				}
				idNodes.put(dp.toString(), p);
			}else {
				//Otherwise, if we find it and our side hasn't been recorded yet update the to/from object with our information
				if(reciprocal){
					if(!p.reciprocal){
						p.reciprocal=true;
						p.toNode=nodeToken;
						p.toThread=threadToken;
					}
				}
				else if(!reciprocal){
					if(p.reciprocal)
					{
						return;
					}
				}
				
				p.fromNode=nodeToken;
				p.fromThread=threadToken;
				
			}
			p.direction = direction;
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
			int actualID=tot.locToGlobStates.get(Integer.valueOf(stateToken)).intValue();

			tw.enterState(time, nodeToken, threadToken, actualID);
			return 0;
		}
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}
			tw.leaveState(time, nodeToken, threadToken, tot.locToGlobStates.get(Integer.valueOf(stateToken)).intValue());
			return 0;
		}

		/*Message registration.  (Message sending is defined in TAUReader below)*/
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){
			
			if(nocomm)return 0;	

			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}

			tw.sendMessage(time, sourceNodeToken, sourceThreadToken, destinationNodeToken, destinationThreadToken, messageSize, messageTag, messageComm);

			return 0;
		}



		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom) {
			if(nocomm)return 0;
			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}
			
			tw.recvMessage(time, sourceNodeToken, sourceThreadToken, destinationNodeToken, destinationThreadToken, messageSize, messageTag, messageCom);

			return 0;
		}

		boolean defRemThread=true;
		
		public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken, double userEventValue) {
			TotID tot = (TotID)userData;
			if(synch){
				time+=tot.offset;
			}
			if(userEventToken==ONESIDED_MESSAGE_SEND||userEventToken==ONESIDED_MESSAGE_RECV||userEventToken==ONESIDED_MESSAGE_UNKNOWN||userEventToken==ONESIDED_MESSAGE_RECIPROCAL_SEND||userEventToken==ONESIDED_MESSAGE_RECIPROCAL_RECV){
                if(nocomm)return 0;
				tot.oneSideType=userEventToken;
				tot.size=(int)userEventValue;
				return 0;
			}
			if(userEventToken==ONESIDED_MESSAGE_ID_TriggerValueT1)
			{
                if(nocomm)return 0;
				tot.dp.l1=userEventValue;
				return 0;
			}
			if(userEventToken==ONESIDED_MESSAGE_ID_TriggerValueT2)
			{
                if(nocomm)return 0;
				tot.dp.l2=userEventValue;
				ToFrom p = idNodes.get(tot.dp.toString());
				if(p==null){
					System.out.println("Bad Recv: "+tot.dp);
					return 0;
				}
				/**
				 * if the to/from object indicates we have reciprocal events but this is not reciprocal, don't record the event.
				 */
				if(p.reciprocal&&!isReciprocal(tot.oneSideType)){
					return 0;
				}
				int remoteNode,remoteThread;
				if(p.toNode==nodeToken&&p.toThread==threadToken){
					remoteNode=p.fromNode;
					remoteThread=p.fromThread;
				}
				else if(p.fromNode==nodeToken&&p.fromThread==threadToken){
					remoteNode=p.toNode;
					remoteThread=p.toThread;
				}	
				else{
					System.out.println(tot.dp+" not for node: "+nodeToken+" thread: "+threadToken);
					return 0;
				}
				//tot.uniR.setLength(0);
				if(defRemThread)
				{
					tw.defUserEvent(7004, "RemoteMessageThreadID", 1);
					defRemThread=false;
				}
				tw.eventTrigger(time, nodeToken, threadToken, 7004, remoteThread);
				if(tot.oneSideType==ONESIDED_MESSAGE_SEND||tot.oneSideType==ONESIDED_MESSAGE_RECIPROCAL_SEND)
				{
					tw.sendMessage(time,  nodeToken, threadToken, remoteNode, remoteThread, (int)userEventValue, 0, 0);
					//System.out.println("Send from"+nodeToken+"-"+threadToken+" to "+remoteNode+"-"+remoteThread+" with tag: "+(int)userEventValue);
				}
				else if (tot.oneSideType==ONESIDED_MESSAGE_RECV||tot.oneSideType==ONESIDED_MESSAGE_RECIPROCAL_RECV)
				{
					tw.recvMessage(time,  remoteNode, remoteThread, nodeToken, threadToken, (int)userEventValue, 0, 0);
					//System.out.println("Recv from"+remoteNode+"-"+remoteThread+" to "+nodeToken+"-"+threadToken+" with tag: "+(int)userEventValue);
				}
				else
				{
					if (p.direction == ONESIDED_MESSAGE_RECV||p.direction==ONESIDED_MESSAGE_RECIPROCAL_RECV)
					{
						//System.out.println("Send from"+nodeToken+"-"+threadToken+" to "+remoteNode+"-"+remoteThread+" with tag: "+(int)userEventValue);
						return tw.sendMessage(time,  nodeToken, threadToken, remoteNode, remoteThread, (int)userEventValue, 0, 0);
						
					}
					else if (p.direction == ONESIDED_MESSAGE_SEND||p.direction==ONESIDED_MESSAGE_RECIPROCAL_SEND) {
						//System.out.println("Recv from"+remoteNode+"-"+remoteThread+" to "+nodeToken+"-"+threadToken+" with tag: "+(int)userEventValue);
						return tw.recvMessage(time,  remoteNode, remoteThread, nodeToken, threadToken, (int)userEventValue, 0, 0);
						
					}
					//otherwise undefined.
					System.out.println("eventTrigger 2, recording " + p.direction + " on thread: "+threadToken);
				}

				return 0;
			}

//			if(tot.thread>0)
//			{
//				System.out.println("Event from node!");
//			}
			
			int stateid = tot.locToGlobStates.get(Integer.valueOf(userEventToken)).intValue();

			tw.eventTrigger(time, nodeToken, threadToken, stateid, (long)userEventValue);
			return 0;}

		public int endTrace(Object userData, int nodeToken, int threadToken){
			return 0;
		}
	}
}
