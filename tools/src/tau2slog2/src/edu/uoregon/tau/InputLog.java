/*
 *    See TAU License file
/*
 * @author  Wyatt Spear
 * Derived from code by Anthony Chan
 */

package edu.uoregon.tau;

import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.EmptyStackException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import logformat.trace.DobjDef;

import base.drawable.Category;
import base.drawable.Composite;
import base.drawable.Kind;
import base.drawable.Primitive;
import base.drawable.Topology;
import base.drawable.YCoordMap;
import base.io.BufArrayOutputStream;
import edu.uoregon.tau.trace.TraceReader;
import edu.uoregon.tau.trace.TraceReaderCallbacks;


class MessageEvent{
	MessageEvent(long time, int tag, int src, int dst, int siz, int com){
		this.time=time;
		this.tag=tag;
		this.src=src;
		this.dst=dst;
		this.siz=siz;
		this.com=com;
	}
	long time;
	int tag;
	int src;
	int dst;
	int siz;
	int com;
}

// A stack that uses primitive arrays to avoid object allocation
class PrimitiveStack {
    private long[] times;
    private int[] tokens;
    private int size = 0;

    public PrimitiveStack(int initialCapacity) {
        times = new long[initialCapacity];
        tokens = new int[initialCapacity];
    }

    public void push(long time, int token) {
        ensureCapacity(size + 1);
        times[size] = time;
        tokens[size] = token;
        size++;
    }

    public long peekTime() {
        if (size == 0) throw new java.util.EmptyStackException();
        return times[size - 1];
    }
    
    public int peekToken() {
        if (size == 0) throw new java.util.EmptyStackException();
        return tokens[size - 1];
    }

    // Pop effectively just decrements the pointer
    public void pop() {
        if (size == 0) throw new java.util.EmptyStackException();
        size--; 
    }
    
    // Helper to retrieve data before popping (replaces the 'leave' object)
    public long getLastTime() { return times[size - 1]; }
    public int getLastToken() { return tokens[size - 1]; }

    public boolean isEmpty() { return size == 0; }
    
    public int size() { return size; }

    private void ensureCapacity(int minCapacity) {
        if (minCapacity > times.length) {
            int newCapacity = times.length * 2;
            times = java.util.Arrays.copyOf(times, newCapacity);
            tokens = java.util.Arrays.copyOf(tokens, newCapacity);
        }
    }
	
	public int getTokenAt(int index) { 
		return tokens[index]; 
	}

	public long getTimeAt(int index) { 
		return times[index]; 
	}
}


class ComSource {
	int source;
	int dest;
	int tag;
	
	

	public ComSource(int source, int dest, int tag) {
		super();
		this.source = source;
		this.dest = dest;
		this.tag = tag;
	}



	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + dest;
		result = prime * result + source;
		result = prime * result + tag;
		return result;
	}



	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		ComSource other = (ComSource) obj;
		if (dest != other.dest)
			return false;
		if (source != other.source)
			return false;
		if (tag != other.tag)
			return false;
		return true;
	}
}

/*
   This class provides the Java version of TRACE-API.
*/
public class InputLog implements base.drawable.InputAPI
{
	//private static String filespec;
	
	private static Integer ZERO = Integer.valueOf(0);
	private static Integer ONE = Integer.valueOf(1);
	private static Integer TWO = Integer.valueOf(2);
	
	private TraceReaderCallbacks ev_cb;
	private static long filehandle;
	private static int num_topology_returned;

	private TraceReader tFileEvRead;
	private static double clockP;
	//private static boolean eventReady;
	private static boolean doneReading;
	
	private static Set<Integer> papiEvents = new HashSet<Integer>();
	
	private static ArrayList<DobjDef> categories;
	//private static int numcats;
	//private static int maxcats;
	//private static base.drawable.YCoordMap ymap;
	private static java.util.LinkedList<base.drawable.Primitive> primeQueue = new java.util.LinkedList<base.drawable.Primitive>();
	private static base.drawable.Primitive lastPrime;
	private base.drawable.Category statedef;
	
	/**
	 * Per-node data structure using dynamic HashMaps for thread-specific state.
	 * Similar pattern to ThreadTracker in TraceReader - avoids fixed allocation
	 * and global ID computation by using two-level node->thread lookups.
	 */
	public static class NodeData {
		public final Map<Integer, PrimitiveStack> eventstacks = new HashMap<>();
		public final Map<Integer, Integer> noMonCycles = new HashMap<>();
	}
	
	private static int maxnode;
	private static int maxthread;
	private static HashMap<Integer, NodeData> nodeData;
	private static HashMap<ComSource, ArrayDeque<MessageEvent>> msgRecStack;
	private static HashMap<ComSource, ArrayDeque<MessageEvent>> msgSenStack;
	
	// Cache last accessed node to reduce HashMap lookups in hot path
	private static int cachedNodeID = -1;
	private static NodeData cachedNodeData = null;
	
	private static boolean papiEnabled = false;
	
	private static Random getcolors;
	private static Set<Integer> msgtags;
	private static Set<Integer> noMonEvents;
	
	private static int maxEvtId;
	
	private long arch_read;
	private long count_read = 0;
	private long stepsize = 0;
	
	public void enablePAPI(boolean enable) {
		papiEnabled = enable;
	}
	
	// Combine node and thread IDs into single integer
	private static int GlobalID(int tid, int nid) {
		return (nid << 16) + tid;
	}
	
//	private static long SourceDest(int source,int dest, int tag)
//	{
//		return ((long)source << 32)+dest;
//	}
	
	/*private static int GlobalID(int localnodeid, int localthreadid)
	{
	  if (maxthread>0)
	  {
	    if (offset == null)
	    {
	      //printf("Error: offset vector is NULL in GlobalId()\n");
	      return localnodeid;
	    }
	    
	    // for multithreaded programs, modify this routine /
	    return offset[localnodeid]+localthreadid;  // for single node program /
	  }
	  else
	  { 
	    return localnodeid;
	  }
	}  */

	public InputLog( String tautrc, String tauedf )
	{
		boolean isOK;
		//filespec = spec_str;
		isOK = this.open(tautrc,tauedf);        // set filehandle
		if ( filehandle == 0 ) {
			if ( isOK )
			{
				//System.out.println( "InputLog.open() exits normally!" );
				//System.exit( 0 );
			}
			else
			{
				System.err.println( "InputLog.open() fails!\n"
						+ "No slog2 file is generated due to previous errors." );
				System.exit( 1 );
			}
		}
		// Initialize Topology name return counter
		num_topology_returned = 0;
	}

/*	private static void initIDs(){
		
	}*/

	private boolean open(String tautrc, String tauedf)
	{
		//private 
		
		categories = new ArrayList<DobjDef>();//TreeMap();//base.drawable.Category[128];
		//logformat.trace.DobjDef newobj= new logformat.trace.DobjDef(1, "message", Topology.ARROW_ID, 
			//	255,255,255, 255, 3, "msg_tag=%d, msg_size=%d", null);

		//categories.add(newobj);//put(new Integer(numcats), newobj);
		//numcats=0;
		//maxcats=0;
		//prime=null;
		primeQueue.clear();
		lastPrime=null;
		clockP=1;
		//eventReady=false;
		doneReading=false;
		
		//numthreads=0;
		maxnode=0;
		maxthread=0;
		maxEvtId=0;
		msgtags=new HashSet<Integer>();
		getcolors = new Random(0);
		noMonEvents=new HashSet<Integer>();
		
		TraceReaderCallbacks def_cb = new TAUReaderInit();
		TraceReader tFileDefRead=new TraceReader(tautrc,tauedf);
		//System.out.println()
		tFileDefRead.setSubtractFirstTimestamp(true);
		tFileDefRead.setDefsOnly(true);
		int recs_read=0;
		do{
			recs_read=tFileDefRead.readNumEvents(def_cb, 1024,null);
			//if(recs_read>0)
			//System.out.println("Read "+recs_read+" records");
		}while(recs_read!=0);//&&((Integer)tb.UserData).intValue()!=0
		tFileDefRead.closeTrace();
		arch_read=tFileDefRead.getNumRecords();
		System.out.println(arch_read+" records initialized.  Processing.");
		
		stepsize=arch_read/50;
		if(stepsize==0){
			stepsize=1;
		}
		
		/*if(maxthread>0)
		{
			offset=new int[maxnode+2];
			offset[0]=0;
			if(maxnode>0)
			{
				for(int i=0;i<maxnode+1;i++)
					offset[i+1]=offset[i]+maxthread+1;
			}
		}*/
		nodeData = new HashMap<Integer, NodeData>();
		msgRecStack = new HashMap<ComSource, ArrayDeque<MessageEvent>>();// Stack[maxnode+1][maxthread+1];
		msgSenStack = new HashMap<ComSource, ArrayDeque<MessageEvent>>();
		cachedNodeID = -1;
		cachedNodeData = null;
		
		ev_cb = new TAUReader();
		tFileEvRead=new TraceReader(tautrc,tauedf);
		tFileEvRead.setSubtractFirstTimestamp(true);
		//tFileDefRead.setSubtractFirstTimestamp(false);
		return true;
	}

	public boolean close()
	{
		ev_cb.endTrace(null, -1, -1);
		tFileEvRead.closeTrace();
		return true;
	}
	
	private int popCategory(){
		statedef = categories.remove(categories.size()-1);// .get(new Integer(maxcats));//[maxcats];
		//maxcats++;
		return Kind.CATEGORY_ID;
	}

	public int peekNextKindIndex()
	{
		
		if(categories.size()>0)//maxcats<categories.size())//numcats)
		{
			return popCategory();
		}

		if(!doneReading)
		{
			while(primeQueue.isEmpty())
			{
				
				if(categories.size()>0)//maxcats<categories.size())//numcats)
				{
					return popCategory();
				}
				
				if(tFileEvRead.readNumEvents(ev_cb, 1024, null)==0)
				{
					doneReading=true;
					if(maxthread>0)//maxnode>=0&&
					{
						//System.out.println("ymap"+maxnode +" "+maxthread);
						return Kind.YCOORDMAP_ID;
					}
					else
						return Kind.EOF_ID;
				}
				else {
					count_read++;
					if(count_read%stepsize==0){
						System.out.println(count_read+" Records read. "+(int)(100*((double)count_read/(double)arch_read))+"% converted");
					}
				}
			}
			//eventReady=false;
			//System.out.println("returning prime");
			return Kind.PRIMITIVE_ID;
		}
		return Kind.EOF_ID;
	}

	public Category getNextCategory()
	{
		return statedef;
	}

	public YCoordMap getNextYCoordMap()
	{
		System.out.println("Getting YMap, Maxnode: "+maxnode+", Maxthread: "+maxthread);
		String colnames[] = new String[]{"NodeID","ThreadID"};
		int elems[] = new int[(maxnode+1)*(maxthread+1)*3];
		int idx = 0;
		for(int i =0; i<=maxnode;i++)
			for(int j=0;j<=maxthread;j++)
			{
				elems[idx++]=(GlobalID(i,j));//(Integer)global.get(new Point(i,j))).intValue();//
				elems[idx++]=i;
				elems[idx++]=j;
				//System.out.println(idx);
			}
		//int elems[]=new int[]{GlobalID(maxnode,maxthread),maxnode,maxthread};
		return new YCoordMap((maxthread+1)*(maxnode+1), 3, "Thread View", colnames, elems, null);
	}

	public Primitive getNextPrimitive()
	{
		if (!primeQueue.isEmpty()) {
			return primeQueue.removeFirst();
		}
		return null;
	}

	public Composite getNextComposite()
	{
		return new Composite();
	}

	public Kind peekNextKind()
	{
		// Return all the Topology names.
		if ( num_topology_returned < 3 )
			return Kind.TOPOLOGY;

		int next_kind_index  = this.peekNextKindIndex();
		switch ( next_kind_index ) {
			case Kind.TOPOLOGY_ID :
				return Kind.TOPOLOGY;
			case Kind.EOF_ID :
				return Kind.EOF;
			case Kind.PRIMITIVE_ID :
				return Kind.PRIMITIVE;
			case Kind.COMPOSITE_ID :
				return Kind.COMPOSITE;
			case Kind.CATEGORY_ID :
				return Kind.CATEGORY;
			case Kind.YCOORDMAP_ID :
				return Kind.YCOORDMAP;
			default :
				System.err.println( "trace.InputLog.peekNextKind(): "
						+ "Unknown value, " + next_kind_index );
		}
		return null;
	}

	public Topology getNextTopology()
	{
		switch ( num_topology_returned ) {
			case 0:
				num_topology_returned = 1;
				return Topology.EVENT;
			case 1:
				num_topology_returned = 2;
				return Topology.STATE;
			case 2:
				num_topology_returned = 3;
				return Topology.ARROW;
			default:
				System.err.println( "All Topology Names have been returned" );
		}
		return null;
	}
	
	protected static byte[] getInfoVals(int[] vals)
	{
		BufArrayOutputStream bary_outs  = new BufArrayOutputStream( vals.length*4 );//8
		DataOutputStream     data_outs  = new DataOutputStream( bary_outs);
		try {
			for(int i=0; i< vals.length;i++)
			{
				data_outs.writeInt(vals[i]);
			}
		} catch ( java.io.IOException ioerr ) {
			ioerr.printStackTrace();
			System.exit( 1 );
		}
		return bary_outs.getByteArrayBuf();
	}	
	
	protected static byte[] getInfoVals(double[] vals)
	{
		BufArrayOutputStream bary_outs  = new BufArrayOutputStream( vals.length*8 );//8
		DataOutputStream     data_outs  = new DataOutputStream( bary_outs);
		try {
			for(int i=0; i< vals.length;i++)
			{
				data_outs.writeDouble(vals[i]);
			}
		} catch ( java.io.IOException ioerr ) {
			ioerr.printStackTrace();
			System.exit( 1 );
		}
		return bary_outs.getByteArrayBuf();
	}	
	
	private static class TAUReaderInit implements TraceReaderCallbacks{
		
		public int defClkPeriod(Object userData, double clkPeriod) {
			clockP =clkPeriod;
			return 0;
		}
		
		public int defThread(Object userData, int nodeToken, int threadToken, String threadName){
			//System.out.println("DefThread nid "+nodeToken+" tid "+threadToken+", thread name "+threadName);
			//global.put(new Point(nodeToken,threadToken), new Integer(numthreads));
			//numthreads++;
			
			return 0;
		}
		
		public int defStateGroup(Object userData, int stateGroupToken, String stateGroupName){return 0;}
		
		public int defState(Object userData, int stateToken, String stateName, int stateGroupToken){
			//System.out.println("DefState stateid "+stateToken+" stateName "+stateName+" stategroup id "+stateGroupToken);
			String name = stateName;
			if(name.charAt(0)=='"' && name.charAt(name.length()-1)=='"')
				name = name.substring(1,name.length()-1);
			logformat.trace.DobjDef newobj= new logformat.trace.DobjDef(stateToken, name, Topology.STATE_ID, 
					getcolors.nextInt(256),getcolors.nextInt(256),getcolors.nextInt(256), 255, 1, null, null);
			categories.add(newobj);///put(new Integer(numcats), newobj);//[numcats]=newcat;
			//numcats++;
			if(stateToken>maxEvtId)
				maxEvtId=stateToken;
			return 0;
		}
		
		static final String PAPI="PAPI_";
		
		public int defUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing){
			
			if(!papiEnabled&&userEventName.startsWith(PAPI)){
				papiEvents.add(userEventToken);
				return 0;
			}
			
			if(userEventToken==7004){
				return 0;
			}
			
			if(monotonicallyIncreasing==0)
			{
				noMonEvents.add(Integer.valueOf(userEventToken));
			}
			
			String name = userEventName;
			if(name.charAt(0)=='"' && name.charAt(name.length()-1)=='"')
				name = name.substring(1,name.length()-1);
			
			logformat.trace.DobjDef newobj= new logformat.trace.DobjDef(userEventToken, name, Topology.EVENT_ID, 
					getcolors.nextInt(256),getcolors.nextInt(256),getcolors.nextInt(256), 255, 20, "Count=%E", null);
			categories.add(newobj);//put(new Integer(numcats), newobj);
			//numcats++;
			if(userEventToken>maxEvtId)
				maxEvtId=userEventToken;
			return 0;
		}
		
		public int enterState(Object userData, long time, int nodeToken, int threadToken, int stateToken){return 0;}
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){return 0;}
		
		/*Message registration.  (Message sending is defined in TAUReader below)*/
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){
			

			
			return 0;}
		public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken, double userEventValue) {return 0;}

		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom) {return 0;}
		
		public int endTrace(Object userData, int nodeToken, int threadToken){return 0;}
	}
	
	private static class TAUReader implements TraceReaderCallbacks{
		
		public int enterState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			//System.out.println("Entered state "+stateToken+" time "+time+" nid "+nodeToken+" tid "+threadToken);
			//eventstack[nodeToken][threadToken];
			//System.out.println("enter "+stateToken);
			
			NodeData node;
			if (cachedNodeID == nodeToken) {
				node = cachedNodeData;
			} else {
				node = nodeData.get(nodeToken);
				if (node == null) {
					node = new NodeData();
					nodeData.put(nodeToken, node);
				}
				cachedNodeID = nodeToken;
				cachedNodeData = node;
			}
			
			PrimitiveStack stack = node.eventstacks.get(threadToken);
			if (stack == null) {
				stack = new PrimitiveStack(128);
				node.eventstacks.put(threadToken, stack);
			}
			
			stack.push(time,stateToken);
			return 0;
		}
		
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			//System.out.println("Leaving state "+stateToken+" time "+time+" nid "+nodeToken+" tid "+threadToken);
			//System.out.println("exit "+stateToken);
			
			NodeData node;
			if (cachedNodeID == nodeToken) {
				node = cachedNodeData;
			} else {
				node = nodeData.get(nodeToken);
			}
			
			PrimitiveStack stack = (node != null) ? node.eventstacks.get(threadToken) : null;
			
			if(stack == null || stack.isEmpty())
			{
				System.out.println("node "+ nodeToken+" thd "+threadToken);
				if(stack != null)
					System.out.println("size: " + stack.size());
				
				System.err.println("Fault: Exit from empty or uninitialized thread");
				return -1;
				//System.exit(1);
			}
    
    if (stack.peekToken() != stateToken) {
        // The top of the stack is NOT what we want to close.
        // Check if the state we want to close is buried deeper in the stack.
        boolean found = false;
       
        for (int i = stack.size() - 1; i >= 0; i--) {
            if (stack.getTokenAt(i) == stateToken) {
                found = true;
                break;
            }
        }

        if (found) {
            // CASE A: The state exists deeper. The top states are "orphaned" children.
            // We must implicitly close them to reach our target.
            System.err.println("Fixing mismatch: Implicitly closing children to reach state: " + stateToken+ " on node: "+nodeToken+", thread: "+threadToken);
            
            while (!stack.isEmpty()) {
                if (stack.peekToken() == stateToken) {
                    break; // We found our target! Stop unwinding.
                }
                
                // Pop and auto-close the orphan
				long orphanTime = stack.getLastTime();
                int orphanToken = stack.getLastToken();
                stack.pop();
                
                // Record the orphaned event using the current time as its end time.
                // This ensures the bar shows up in the viewer, even if it runs a bit long.
                base.drawable.Primitive p = new base.drawable.Primitive(
                    orphanToken, orphanTime * clockP, time * clockP,
                    new double[]{orphanTime * clockP, time * clockP},
                    new int[]{GlobalID(nodeToken, threadToken), GlobalID(nodeToken, threadToken)}, null
                );
                if (lastPrime == null || comparePrimatives(p, lastPrime)) {
                    primeQueue.add(p);
                    lastPrime = p;
                }
            }
        } else {
            // CASE B: The state is NOT in the stack. This is a spurious exit event.
            // We should ignore it to protect the stack integrity.
            System.err.println("Skipping spurious exit for state " + stateToken);
            return 0;
        }
    }
			
			
			
		long entryTime = stack.getLastTime();
		stack.pop();
		
			base.drawable.Primitive p = new base.drawable.Primitive(
			stateToken,entryTime*clockP,time*clockP,
					new double[]{entryTime*clockP,time*clockP} ,new int[]{GlobalID(nodeToken,threadToken),GlobalID(nodeToken,threadToken)},null); // create local 'p'
			
			if(lastPrime==null||(comparePrimatives(p,lastPrime))) {
				primeQueue.add(p); // Add to queue
				lastPrime=p;
			}
			
			//System.out.println(nodeToken+" "+threadToken+" vs. "+((Integer)global.get(new Point(nodeToken,threadToken))).intValue()+" "+((Integer)global.get(new Point(nodeToken,threadToken))).intValue());
			return 0;
		}
		
		//TODO: Use event comparison
//		private static boolean compareEvts(MessageEvent a, MessageEvent b){
//			
//			if(a.tag==b.tag&&a.src==b.src&&a.dst==b.dst&&a.siz==b.siz&&a.com==b.com&&a.time!=b.time)
//				return true;
//			
//			return false;
//		}
		
		/**
		 * Returns true if the argument primitives are not too similar to both be printed.  False if they should not both be printed (they are the same category in the same location)
		 * @param prim1
		 * @param prim2
		 * @return
		 */
		private static boolean comparePrimatives(Primitive prim1,Primitive prim2){
			if(prim1.getCategoryIndex()!=prim2.getCategoryIndex()||!prim1.equals(prim2)||(prim1.getStartVertex().lineID!=prim2.getStartVertex().lineID&&prim1.getFinalVertex().lineID!=prim2.getFinalVertex().lineID))
				return true;
			return false;
		}
		
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){

			if(remoteThread>-1)
			{
				destinationThreadToken=remoteThread;
				remoteThread=-1;
			}
			else
				destinationThreadToken=0;
			/*
			 * The 'tag space' and 'eventId space' may overlap and overwrite each other
			 * So add the highest tag ID+1, to be sure all IDs sent to jumpshot are unique.
			 * */
			int messageTagShift=messageTag+maxEvtId+1;
			/*
			 * If we have a new tag ID we need to register a new object type for it
			 * (Is there a faster way to do this?)
			 */
			Integer msgTagKey = Integer.valueOf(messageTag);
			if(msgtags.add(msgTagKey))
			{
				logformat.trace.DobjDef newobj= new logformat.trace.DobjDef(messageTagShift, "message "+messageTag, Topology.ARROW_ID, 
						255,255,255, 255, 3, "msg_tag=%d, msg_size=%d, msg_comm=%d", null);

				categories.add(newobj);
			}

			/*
			 * Get the global source and destination IDs.  From those, get a unique long
			 * that represents the transaction-ID (the source and dest of the message)
			 */
			int destGlob = GlobalID(destinationNodeToken,destinationThreadToken);
			int sourceGlob = GlobalID(sourceNodeToken,sourceThreadToken);
			//Long srcDst=new Long(SourceDest(sourceGlob,destGlob,messageTag));
			ComSource cs = new ComSource(sourceGlob,destGlob,messageTag);
			
			MessageEvent sndE=new MessageEvent(time,messageTag,sourceGlob,destGlob,messageSize,messageComm);
			
			MessageEvent leave = null;
			/*
			 * If we have a message event in the recieved list then there has been a recieve sent before its send.
			 * Match this send with the advanced recieve and submit the event.  (Is this always correct?)
			 */
			ArrayDeque<MessageEvent> recList = msgRecStack.get(cs);
			if(recList != null && !recList.isEmpty() && (leave = recList.removeFirst()) != null)
			{
					System.out.println("Reversed Message: time "+time+", source nid "+ sourceNodeToken +
							" tid "+sourceThreadToken+", destination nid "+destinationNodeToken+" tid "+
							destinationThreadToken+", size "+messageSize+", tag "+messageTag);
					base.drawable.Primitive p = new base.drawable.Primitive(messageTagShift,leave.time*clockP,time*clockP,
							new double[]{time*clockP,leave.time*clockP,} ,
							new int[]{sourceGlob,destGlob},
							getInfoVals(new int[]{messageTag,messageSize,messageComm}));
					
					
					if(lastPrime==null||(
							comparePrimatives(p,lastPrime)
							))
					{	
						primeQueue.add(p);
						lastPrime=p;
					}
					
					//eventReady=true;
					return 0;
			}
			
			/*
			 * If we do not yet have an event source-list list for this transaction ID, create one
			 * Put this event in the source list
			 */
			ArrayDeque<MessageEvent> senList = msgSenStack.get(cs);
			if(senList == null) {
				senList = new ArrayDeque<MessageEvent>();
				msgSenStack.put(cs, senList);
			}
			senList.add(sndE);
			return 0;
		}//sendMessage
		
		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){
			
			int messageTagShift=messageTag+maxEvtId+1;
			
			if(remoteThread>-1)
			{
				sourceThreadToken=remoteThread;
				remoteThread=-1;
			}
			else
				sourceThreadToken=0;
			int sourceGlob = GlobalID(sourceNodeToken,sourceThreadToken);
			int destGlob = GlobalID(destinationNodeToken,destinationThreadToken);
			//Long srcDst=new Long(SourceDest(sourceGlob,destGlob,messageTag));
			ComSource cs = new ComSource(sourceGlob,destGlob,messageTag);
			MessageEvent recE=new MessageEvent(time,messageTag, sourceGlob,destGlob,messageSize,messageComm);
			
			MessageEvent leave = null;
			ArrayDeque<MessageEvent> senList = msgSenStack.get(cs);
			if(senList != null && !senList.isEmpty() && (leave = senList.removeFirst()) != null)
			{
					base.drawable.Primitive p = new base.drawable.Primitive(messageTagShift,leave.time*clockP,time*clockP,
							new double[]{leave.time*clockP,time*clockP},
							new int[]{
							sourceGlob,destGlob
							},
							getInfoVals(new int[]{messageTag,messageSize,messageComm}));
					
					if(lastPrime==null||(
							comparePrimatives(p,lastPrime)
							))
					{	
						primeQueue.add(p);
						lastPrime=p;
					}
					
					//eventReady=true;
					return 0;
			}
			
			ArrayDeque<MessageEvent> recList = msgRecStack.get(cs);
			if(recList == null) {
				recList = new ArrayDeque<MessageEvent>();
				msgRecStack.put(cs, recList);
			}
			recList.add(recE);
			return 0;
		}//recvMessage
		
		/*
		 * If a message is sent from a thread (7004) we need to get its value from the send since we don't get thread values in messages naturally.
		 */
		int remoteThread=-1;
		
		public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken,
				double userEventValue){
			
			if(!papiEnabled&&papiEvents.contains(userEventToken)){
				return 0;
			}
			
			if(userEventToken==7004){
				remoteThread=(int) userEventValue;
				return 0;
			}
			
			//System.out.println("EventTrigger: time "+time+", nid "+nodeToken+" tid "+threadToken+" event id "+userEventToken+" triggered value "+userEventValue);
			Integer userEventKey = Integer.valueOf(userEventToken);
			if(noMonEvents.contains(userEventKey))
			{
				//System.out.println(noMonEventCycle[nodeToken][threadToken]);
				
				NodeData node = nodeData.get(nodeToken);
				if (node == null) {
				node = new NodeData();
					nodeData.put(nodeToken, node);
				}
				
Integer cycle = node.noMonCycles.get(threadToken);
		if (cycle == null) cycle = Integer.valueOf(0);
				
				if(cycle == 0)
				{
					node.noMonCycles.put(threadToken, 1);
					return 0;
				}
				else if(cycle == 1)
				{
					node.noMonCycles.put(threadToken, 2);
				}
				else if(cycle == 2)
				{
					node.noMonCycles.put(threadToken, 0);
					return 0;
				}
			}
			//System.out.println("Test Evt Double: "+userEventValue+" vs: "+(long)userEventValue);
			base.drawable.Primitive p = new base.drawable.Primitive(userEventToken,time*clockP,time*clockP,
					new double[]{time*clockP} ,new int[]{
					GlobalID(nodeToken,threadToken)
					//((Integer)global.get(new Point(nodeToken,threadToken))).intValue()
					},
					getInfoVals(new double[]{userEventValue}));
			if(lastPrime==null||(
					comparePrimatives(p,lastPrime)
					))
			{	
				primeQueue.add(p);
				lastPrime=p;
			}
			//eventReady=true;
			return 0;
		}

		public int defClkPeriod(Object userData, double clkPeriod) {return 0;}

		public int defState(Object userData, int stateToken, String stateName, int stateGoupToken) {return 0;}

		public int defStateGroup(Object userData, int stateGroupToken, String stateGroupName) {return 0;}

		public int defThread(Object userData, int nodeToken, int threadToken, String threadName) {
			
			if(nodeToken>maxnode)maxnode=nodeToken;
			if(threadToken>maxthread)maxthread=threadToken;
			
			return 0;}

		public int defUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing) {return 0;}

		public int endTrace(Object userData, int nodeToken, int threadToken) {
			return 0;
		}
	}
}
