package edu.uoregon.tau.multimerge;

import java.awt.Point;
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

	/**
	 * The number of send events seen on this thread.  Used to detect communication triplets used by CUDA traces
	 */
	public int sSeen=0;

	/**
	 * The number of receive events seen on this thread.  Used to detect communication triplets used by CUDA traces
	 */
	public int rSeen=0;

	/**
	 * Most recent valid CUDA mem-copy size, set in the second of 3 communication triplets.
	 */
	int truSize=0;

	/**
	 * The unique CUDA communication id composed of the first and last tags of the first and last triplets for communication between two threads, on the receiving side
	 */
	StringBuilder uniR=new StringBuilder();//null;

	/**
	 * The unique CUDA communication id composed of the first and last tags of the first and last triplets for communication between two threads, on the sending side
	 */
	StringBuilder uniS=new StringBuilder();//null;

	/**
	 * The combination of filename and timestamp for the current communication.  If is not CUDA communication it will be stored in a global list for quick differentiation
	 */
	StringBuilder mpi=new StringBuilder();//null;

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

public class MultiMerge {

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

	/**
	 * This set contains all filename/timestamp combinations that are associated with mpi communication events.  If the string is not in here then it is cuda communication.
	 */
	static Set<String> mpiCom;

	/**
	 * The map from the tracefile name to the associated TotID object
	 */
	static Map<String, TotID> totMap;

	/**
	 * The tau trace writer object which will write the merged trace
	 */
	static TraceWriter tw;

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
				out[0]=Integer.parseInt(a[1]);
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

		boolean isCuda=tot.filename.contains("cuda");

		int[]fn=getNCT(tot.filename);
		int threadToken=-1;
		if(isCuda)
		{
			threadToken=0;
		}else{
			threadToken=fn[2];
		}

		/*
		 * If this is a threaded trace file we are on the same node as before, just update the thread
		 */
		if(threadToken>0){
			tot.node=lastNode;
			tot.thread=threadToken;
		}
		/*
		 * If this is not a threaded entry we're on a new node so increment and set thread to 0.  If this is not a cuda tracefile we need to map the local node ID to the global nodeID.
		 */
		else{
			lastNode++;
			tot.node=lastNode;
			tot.thread=0;
			if(!isCuda){
				oldNewDest.put(new Integer(fn[0]), new Integer(lastNode));
			}	
		}


		/*
		 * Identify the threads based on the n/c/t in the file names.
		 */
		if(!tot.filename.startsWith("tautrace")){

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
			totMap.put(t.filename, t);
			recs_read=0;
			do{
				recs_read=initReaders[rs].readNumEvents(init_cb, -1,initReaders[rs].getTraceFile());//1024
			}while(recs_read!=0&&!initReaders[rs].isDone());
			initReaders[rs].closeTrace();
			initReaders[rs]=null;
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
		for(int rs=0;rs<readers.length;rs++)
		{
			readers[rs]=new TraceReader(traces.get(rs),getEDFName(traces.get(rs)));
			readers[rs].setDefsOnly(false);
			readers[rs].setSubtractFirstTimestamp(false);//TODO: Why is this needed only for cuda output?
			sorter[rs]=readers[rs].peekTime();
		}

		int minDex=minTime(sorter);
		while(minDex>=0){
			int read=readers[minDex].readNumEvents(read_cb, 1, readers[minDex].getTraceFile());
			if(read==0){
				sorter[minDex]=-1;
				readers[minDex].closeTrace();
				readers[minDex]=null;
			}else{
				sorter[minDex]=readers[minDex].peekTime();
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

		mpiCom=new HashSet<String>();
		oldNewDest=new HashMap<Integer,Integer>();
		idNodes=new HashMap<String,Point>();
		stateMap=new HashMap<String,Integer>();
		ueMap=new HashMap<String,Integer>();
		totMap=new HashMap<String,TotID>();
		
		tw = new TraceWriter("tau.trc", "tau.edf");
		List<String> traces = listTraces();

		initializeMerge(traces);

		dataMerge(traces);

		tw.closeTrace();
	}

	private static StringBuilder mpiID(StringBuilder mpi, long stamp, String fname){
		mpi.setLength(0);
		mpi.append(stamp);
		mpi.append(".");
		mpi.append(fname);
		return mpi;
	}

	private static StringBuilder commID1(StringBuilder id, int remNode, int size, int tag, int commun){
		id.setLength(0);
		id.append(remNode);
		id.append(".");
		id.append(size);
		id.append(".");
		id.append(tag);
		id.append(".");
		id.append(commun);
		return id;
	}

	private static StringBuilder commID2(StringBuilder id, int remNode, int size, int tag, int commun){
		id.append(".");
		id.append(remNode);
		id.append(".");
		id.append(size);
		id.append(".");
		id.append(tag);
		id.append(".");
		id.append(commun);
		return id;
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
			Integer globstate=stateMap.get(stateName);

			if(globstate==null){
				tw.defState(numStates, stateName, stateGroupToken);
				globstate=new Integer(numStates);
				stateMap.put(stateName, globstate);
				numStates++;
			}

			TotID tot = totMap.get((String)userData);

			tot.locToGlobStates.put(new Integer(stateToken), globstate);

			totMap.put((String)userData, tot);

			return 0;
		}

		public int defUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing){
			Integer globevts=ueMap.get(userEventName);

			if(globevts==null){

				tw.defUserEvent(numStates, userEventName, monotonicallyIncreasing);
				globevts=new Integer(numStates);
				ueMap.put(userEventName, globevts);
				numStates++;
			}

			totMap.get((String)userData).locToGlobStates.put(new Integer(userEventToken), globevts);

			return 0;
		}

		public int enterState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			TotID tot = totMap.get((String)userData);
			tot.sSeen=0;
			tot.rSeen=0;
			return 0;
		}
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			TotID tot = totMap.get((String)userData);
			tot.sSeen=0;
			tot.rSeen=0;
			return 0;
		}

		/*Message registration.  (Message sending is defined in TAUReader below)*/
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){
			TotID tot = totMap.get((String)userData);
			tot.rSeen=0;

			if(tot.sSeen==0){
				tot.mpi=mpiID(tot.mpi,time,(String)userData);
				mpiCom.add(tot.mpi.toString());
				tot.sSeen++;
				tot.uniS=commID1(tot.uniS,destinationNodeToken,messageSize,messageTag,messageComm);
				return 0;
			}
			else if(tot.sSeen==1)
			{
				mpiCom.remove(tot.mpi.toString());

				tot.sSeen++;

				return 0;
			}
			else if(tot.sSeen==2){

				tot.uniS=commID2(tot.uniS,destinationNodeToken,messageSize,messageTag,messageComm);
				Point p =idNodes.get(tot.uniS.toString());
				if(p==null){
					p=new Point(tot.node,-1);
				}else{
					if(p.x==tot.node||p.y==tot.node){

					}else{
						if(p.y!=-1){
							System.out.println("Warning, doubling up node identifiers!");
						}
						p.y=tot.node;
					}
				}
				idNodes.put(tot.uniS.toString(), p);//Does Uni need to be duplicated?
				tot.uniS.setLength(0);
				tot.sSeen=0;
			}

			return 0;}

		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom) {

			TotID tot = totMap.get((String)userData);
			tot.sSeen=0;
			if(tot.rSeen==0){
				tot.mpi=mpiID(tot.mpi,time,(String)userData);
				mpiCom.add(tot.mpi.toString());

				tot.uniR=commID1(tot.uniR,sourceNodeToken,messageSize,messageTag,messageCom);
				tot.rSeen++;
				return 0;
			}
			else if(tot.rSeen==1)
			{
				mpiCom.remove(tot.mpi.toString());

				tot.rSeen++;

				return 0;
			}
			else if(tot.rSeen==2) {

				tot.uniR=commID2(tot.uniR,sourceNodeToken,messageSize,messageTag,messageCom);
				Point p =idNodes.get(tot.uniR.toString());
				if(p==null){
					p=new Point(tot.node,-1);
				}else{
					if(p.x==tot.node||p.y==tot.node){

					}else{
						if(p.y!=-1){
							System.out.println("Warning, doubling up node identifiers!");
						}
						p.y=tot.node;

					}
				}
				idNodes.put(tot.uniR.toString(), p);//Does Uni need to be duplicated?
				tot.uniR.setLength(0);
				tot.rSeen=0;
			}

			return 0;
		}

		public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken, double userEventValue) {
			TotID tot = totMap.get((String)userData);
			tot.sSeen=0;
			tot.rSeen=0;

			return 0;}

		public int endTrace(Object userData, int nodeToken, int threadToken){
			TotID tot = totMap.get((String)userData);
			tot.sSeen=0;
			tot.rSeen=0;
			return 0;
		}
	}


	/**
	 * Callbacks for performing writes of the merged trace file
	 * @author wspear
	 *
	 */
	private static class TAUReaderWriteall implements TraceReaderCallbacks{
		StringBuilder tmpSB = new StringBuilder();
		
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
			TotID tot = totMap.get((String)userData);

			int actualID=tot.locToGlobStates.get(new Integer(stateToken)).intValue();

			tw.enterState(time, tot.node, tot.thread, actualID);
			return 0;
		}
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			TotID tot = totMap.get((String)userData);

			tw.leaveState(time, tot.node, tot.thread, tot.locToGlobStates.get(new Integer(stateToken)).intValue());
			return 0;
		}

		/*Message registration.  (Message sending is defined in TAUReader below)*/
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){

			TotID tot = totMap.get((String)userData);

			if(mpiCom.contains(mpiID(tmpSB,time,(String)userData).toString())){

				Integer transNode=oldNewDest.get(new Integer(destinationNodeToken));
				if(transNode==null)
				{
					System.out.println("Bad Dest Node ID: "+destinationNodeToken);
					return 0;
				}

				tw.sendMessage(time, tot.node, tot.thread, transNode.intValue(), destinationThreadToken, messageSize, messageTag, messageComm);

				return 0;
			}

			if(tot.sSeen==0){
				tot.uniS=commID1(tot.uniS,destinationNodeToken,messageSize,messageTag,messageComm);
				tot.sSeen++;
			}
			else if(tot.sSeen==1){
				tot.truSize=destinationThreadToken;
				tot.sSeen++;
			}
			else if(tot.sSeen==2){

				tot.sSeen=0;
				tot.uniS=commID2(tot.uniS,destinationNodeToken,messageSize,messageTag,messageComm);
				Point p = idNodes.get(tot.uniS.toString());
				if(p==null){
					System.out.println("Bad Send: "+tot.uniS);
					return 0;
				}

				if(p.x!=tot.node&&p.y!=tot.node){
					System.out.println(tot.uniS+" send not for "+tot.node);
				}

				int remote;
				if(p.x==tot.node)
				{
					remote=p.y;
				}else{
					remote=p.x;
				}
				tw.sendMessage(time, tot.node, tot.thread, remote, 0, tot.truSize, 0, 0);
				tot.truSize=0;
			}else{
				System.out.println("How?");
			}

			return 0;
		}



		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom) {
			TotID tot = totMap.get((String)userData);

			if(mpiCom.contains(mpiID(tmpSB,time,(String)userData).toString())){

				Integer transNode=oldNewDest.get(new Integer(sourceNodeToken));
				if(transNode==null)
				{
					System.out.println("Bad Source Node ID: "+sourceNodeToken);
					return 0;
				}
				tw.recvMessage(time, transNode.intValue(), sourceThreadToken, tot.node, tot.thread, messageSize, messageTag, messageCom);

				return 0;
			}

			if(tot.rSeen==0){
				tot.uniR=commID1(tot.uniR,sourceNodeToken,messageSize,messageTag,messageCom);
				tot.rSeen++;
			}
			else if(tot.rSeen==1){
				tot.truSize=sourceThreadToken;
				tot.rSeen++;
			}
			else if(tot.rSeen==2){
				tot.rSeen=0;
				tot.uniR=commID2(tot.uniR,sourceNodeToken,messageSize,messageTag,messageCom);
				Point p = idNodes.get(tot.uniR.toString());
				if(p==null){
					System.out.println("Bad Recv: "+tot.uniR);
					return 0;
				}

				if(p.x!=tot.node&&p.y!=tot.node){
					System.out.println(tot.uniR+" not for "+tot.node);
				}
				tot.uniR.setLength(0);
				int remote;
				if(p.x==tot.node)
				{
					remote=p.y;
				}else{
					remote=p.x;
				}
				tw.recvMessage(time,  remote, 0, tot.node, tot.thread, tot.truSize, 0, 0);
				tot.truSize=0;

			}
			else{
				System.out.println("How?");
			}

			return 0;
		}


		public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken, double userEventValue) {
			TotID tot = totMap.get((String)userData);

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
