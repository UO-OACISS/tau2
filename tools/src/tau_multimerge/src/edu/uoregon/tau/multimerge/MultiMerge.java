package edu.uoregon.tau.multimerge;

import java.awt.Point;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.uoregon.tau.trace.TraceFactory;
import edu.uoregon.tau.trace.TraceReader;
import edu.uoregon.tau.trace.TraceReaderCallbacks;
import edu.uoregon.tau.trace.TraceWriter;


class TotID{
	public String filename;
	public int thread;
	public int node;
	
	public Map<Integer, Integer> locToGlobStates;
	//public Map<Integer, Integer> locToGlobEvts;
	
	public String dualID=null;
	
	public int sSeen=0;
	public int rSeen=0;
	
	int truSize=0;
	String uniR=null;
	String uniS=null;
	String mpi=null;
	
	TotID(String fname){
		filename=fname;
		locToGlobStates=new HashMap<Integer, Integer>();
		//locToGlobEvts=new HashMap<Integer, Integer>();
		thread=-1;
		node=-1;
	}
}


public class MultiMerge {

	static Map<String,Integer> stateMap;
	static Map<String,Integer> ueMap;
	static Map<String, Integer> threadMap; //The name of the tracefile maps to the thread we will use, the acutal passed node will stay the same
	static Map<Integer, Integer>lastThread;
	static Map<String,String>cudaCom;
	static int numStates=0;
	//static int numUEs;
	static int lastNode=-1;
	static int tauCpyID;
	static int cdaCpyID;
	static Map<Integer,Integer> oldNewDest;
	
	static Map<String,Point> idNodes;
	
	static Set<String> mpiCom;
	
	static Map<String, TotID> totMap;
	
	static TraceWriter tw;
	
	static int minTime(long[] times){
		int least=-1;
		long min=times[0];
		for(int i=0; i<times.length;i++){
			if(times[i]>=0&&times[i]<=min){
				min=times[i];
				least=i;
			}
		}
		return least;
	}

	
	private static int[] getNCT(String name){
		
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
		
			if(threadToken>0){
				tot.node=lastNode;
				tot.thread=threadToken;
			}
			else{
				
			lastNode++;
			tot.node=lastNode;
			tot.thread=0;
			if(!isCuda){
				oldNewDest.put(new Integer(fn[0]), new Integer(lastNode));
			}
			
		}
		
		
		
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
	 * @param args
	 */
	public static void main(String[] args) {
		
		List<String> traces = new ArrayList<String>();
		List<String> events = new ArrayList<String>();
		mpiCom=new HashSet<String>();
		oldNewDest=new HashMap<Integer,Integer>();
		
		idNodes=new HashMap<String,Point>();
		
		int i=0;
		while(i<args.length){
			if(args[i].endsWith(".trc")){
				traces.add(args[i]);
			}
			else if(args[i].endsWith(".edf")){
				events.add(args[i]);
			}
			i++;
		}
		
		TraceReader[] readers = new TraceReader[traces.size()];
		
		
		
		TraceReaderCallbacks init_cb = new TAUReaderInit();
		
		stateMap=new HashMap<String,Integer>();
		ueMap=new HashMap<String,Integer>();
		totMap=new HashMap<String,TotID>();
		lastThread=new HashMap<Integer, Integer>();
		cudaCom=new HashMap<String,String>();
		
		 tw = TraceFactory.OpenFileForOutput("tau.trc", "tau.edf");
		
		int recs_read;
		for(int rs=0;rs<readers.length;rs++)
		{	
			readers[rs]=TraceFactory.OpenFileForInput(traces.get(rs),events.get(rs));
			readers[rs].setDefsOnly(false);
			readers[rs].setSubtractFirstTimestamp(false);
			TotID t = new TotID(readers[rs].getTraceFile());
			t=initTotLoc(t);
			totMap.put(t.filename, t);
			recs_read=0;
			do{
				recs_read=readers[rs].readNumEvents(init_cb, 1,readers[rs].getTraceFile());
			}while(recs_read!=0);
			
		}
		
		for(int rs=0;rs<readers.length;rs++){
			readers[rs].closeTrace();//.reset();
			readers[rs]=null;
		}
		
		for(int rs=0;rs<readers.length;rs++)
		{
			readers[rs]=TraceFactory.OpenFileForInput(traces.get(rs),events.get(rs));
			readers[rs].setDefsOnly(false);
			readers[rs].setSubtractFirstTimestamp(false);//TODO: Why is this needed only for cuda output?
			recs_read=0;
			
		}
		
		long[] sorter = new long[readers.length];
		
		init_cb = new TAUReaderWriteall();
		
		for(int rs=0;rs<readers.length;rs++)
		{
			sorter[rs]=readers[rs].peekTime();
		}
		
		int minDex=minTime(sorter);
		while(minDex>=0){
			int read=readers[minDex].readNumEvents(init_cb, 1, readers[minDex].getTraceFile());
			if(read==0){
				sorter[minDex]=-1;
				readers[minDex].closeTrace();
				readers[minDex]=null;
			}else{
				sorter[minDex]=readers[minDex].peekTime();
			}

			minDex=minTime(sorter);
		}
		
		for(int rs=0;rs<readers.length;rs++){
			if(readers[rs]!=null)
				readers[rs].closeTrace();
		}
		
		tw.closeTrace();
	}
	
	
	
	private static class TAUReaderInit implements TraceReaderCallbacks{


		public int defClkPeriod(Object userData, double clkPeriod) {
			tw.defClkPeriod(clkPeriod);
			//sSeen=0;
			//rSeen=0;
			return 0;
		}
		
		public int defThread(Object userData, int nodeToken, int threadToken, String threadName){
			//sSeen=0;
			//rSeen=0;
			return 0;
		}
		
		public int defStateGroup(Object userData, int stateGroupToken, String stateGroupName){
			tw.defStateGroup(stateGroupName, stateGroupToken);
			//sSeen=0;
			//rSeen=0;
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
			
			//sSeen=0;
			//rSeen=0;
			
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

			//sSeen=0;
			//rSeen=0;
			
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
				tot.mpi=time+"."+(String)userData;
				mpiCom.add(tot.mpi);
				tot.sSeen++;
				tot.uniS=destinationNodeToken+"."+messageSize+"."+messageTag+"."+messageComm;
				return 0;
			}
			else if(tot.sSeen==1)
			{
				mpiCom.remove(tot.mpi);

					tot.sSeen++;

				return 0;
			}
			else if(tot.sSeen==2){

				tot.uniS+="."+destinationNodeToken+"."+messageSize+"."+messageTag+"."+messageComm;
				Point p =idNodes.get(tot.uniS);
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
				idNodes.put(tot.uniS, p);//Does Uni need to be duplicated?
				tot.uniS=null;
				tot.sSeen=0;

			}
			
			return 0;}

		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom) {
			

			TotID tot = totMap.get((String)userData);
			tot.sSeen=0;
			if(tot.rSeen==0){
				tot.mpi=time+"."+(String)userData;
				mpiCom.add(tot.mpi);

				tot.uniR=sourceNodeToken+"."+messageSize+"."+messageTag+"."+messageCom;
				tot.rSeen++;
				return 0;
			}
			else if(tot.rSeen==1)
			{
				mpiCom.remove(tot.mpi);

					tot.rSeen++;

				return 0;
			}
			else if(tot.rSeen==2) {
				
				tot.uniR+="."+sourceNodeToken+"."+messageSize+"."+messageTag+"."+messageCom;
				Point p =idNodes.get(tot.uniR);
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
				idNodes.put(tot.uniR, p);//Does Uni need to be duplicated?
				tot.uniR=null;
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
			TotID tot = totMap.get((String)userData);
			
			int actualID=tot.locToGlobStates.get(new Integer(stateToken)).intValue();
			
			if(actualID==tauCpyID){
				//tot.
			}
			else if(actualID==cdaCpyID){
				
			}
			
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
			
			if(mpiCom.contains(time+"."+(String)userData)){
				
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
				tot.uniS=destinationNodeToken+"."+messageSize+"."+messageTag+"."+messageComm;
				tot.sSeen++;
			}
			else if(tot.sSeen==1){
				tot.truSize=destinationThreadToken;
				tot.sSeen++;
			}
			else if(tot.sSeen==2){

				tot.sSeen=0;
				tot.uniS+="."+destinationNodeToken+"."+messageSize+"."+messageTag+"."+messageComm;
				Point p = idNodes.get(tot.uniS);
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
			
			
			if(mpiCom.contains(time+"."+(String)userData)){
				
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
				tot.uniR=sourceNodeToken+"."+messageSize+"."+messageTag+"."+messageCom;
				tot.rSeen++;
			}
			else if(tot.rSeen==1){
				tot.truSize=sourceThreadToken;
				tot.rSeen++;
			}
			else if(tot.rSeen==2){
				tot.rSeen=0;
				tot.uniR+="."+sourceNodeToken+"."+messageSize+"."+messageTag+"."+messageCom;
				Point p = idNodes.get(tot.uniR);
				if(p==null){
					System.out.println("Bad Recv: "+tot.uniR);
					return 0;
				}
				
				if(p.x!=tot.node&&p.y!=tot.node){
					System.out.println(tot.uniR+" not for "+tot.node);
				}
				tot.uniR=null;
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
