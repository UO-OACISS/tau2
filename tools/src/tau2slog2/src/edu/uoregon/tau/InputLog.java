/*
 *    See TAU License file
/*
 * @author  Wyatt Spear
 * Derived from code by Anthony Chan
 */

package edu.uoregon.tau;

import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.EmptyStackException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Random;
import java.util.Set;

import base.drawable.Category;
import base.drawable.Composite;
import base.drawable.Kind;
import base.drawable.Primitive;
import base.drawable.Topology;
import base.drawable.YCoordMap;
import base.io.BufArrayOutputStream;
import edu.uoregon.tau.trace.TraceFactory;
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

class PrimEvent
{
	PrimEvent(long t,int s){
		time = t;
		stateToken = s;
	}
	long time;
	int stateToken;
}

class EZStack extends ArrayList{
	/**
	 * 
	 */
	private static final long serialVersionUID = 4135823375949466058L;

	public EZStack() {
    }
	public void push(Object o){
		add(o);
	}
	
	public Object peek(){
		int	len = size();

		if (len == 0)
		    throw new EmptyStackException();
		return get(len - 1);
	}
	
	public Object pop(){
		Object obj;
		
		int	len = size();

		obj = peek();
		remove(len - 1);

		return obj;
	}
	
	public boolean empty() {
		return size() == 0;
	    }
}

/*
   This class provides the Java version of TRACE-API.
*/
public class InputLog implements base.drawable.InputAPI
{
	//private static String filespec;
	
	private static Integer ZERO = new Integer(0);
	private static Integer ONE = new Integer(1);
	private static Integer TWO = new Integer(2);
	
	private TraceReaderCallbacks ev_cb;
	private static long filehandle;
	private static int num_topology_returned;

	private TraceReader tFileEvRead;
	private static double clockP;
	private static boolean eventReady;
	private static boolean doneReading;
	
	private static ArrayList categories;
	//private static int numcats;
	//private static int maxcats;
	//private static base.drawable.YCoordMap ymap;
	private static base.drawable.Primitive prime;
	private base.drawable.Category statedef;
	
	//private static Map global;
//	private static Map threadspernode;
	//private static int numthreads;
	private static int maxnode;
	private static int maxthread;
	private static HashMap eventstack;
	private static HashMap msgRecStack;
	private static HashMap msgSenStack;
	//private static int[] offset;
	
	private static Random getcolors;
	private static Set msgtags;
	/**
	 * Associates each thread with a state of 0,1 or 2 with 1 indicating an actual event containing data, 
	 * and 0 or 2 indicating leading and trailing 'empty' events.
	 */
	private static HashMap noMonEventCycle;
	private static Set noMonEvents;
	
	private static int maxEvtId;
	
	private long arch_read;
	private long count_read=0;
	private long stepsize=0;
	
	private static int GlobalID(int tid, int nid){
		//System.out.println("n: "+nid+ " t: "+tid);
		return (nid << 16)+tid;
	}
	
	private static long SourceDest(int source,int dest)
	{
		return ((long)source << 32)+dest;
	}
	
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
		
		categories = new ArrayList();//TreeMap();//base.drawable.Category[128];
		//logformat.trace.DobjDef newobj= new logformat.trace.DobjDef(1, "message", Topology.ARROW_ID, 
			//	255,255,255, 255, 3, "msg_tag=%d, msg_size=%d", null);

		//categories.add(newobj);//put(new Integer(numcats), newobj);
		//numcats=0;
		//maxcats=0;
		prime=null;
		clockP=1;
		eventReady=false;
		doneReading=false;
		
		//numthreads=0;
		maxnode=0;
		maxthread=0;
		maxEvtId=0;
		msgtags=new HashSet();
		getcolors = new Random(0);
		noMonEvents=new HashSet();
		
		TraceReaderCallbacks def_cb = new TAUReaderInit();
		TraceReader tFileDefRead=TraceFactory.OpenFileForInput(tautrc,tauedf);
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
		arch_read=TraceFactory.getNumRecords(tautrc);
		System.out.println(arch_read+" records initialized.  Processing.");
		
		stepsize=arch_read/50;
		
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
		eventstack = new HashMap();// Stack[maxnode+1][maxthread+1];
		msgRecStack = new HashMap();// Stack[maxnode+1][maxthread+1];
		msgSenStack = new HashMap();
		noMonEventCycle = new HashMap();//[maxnode+1][maxthread+1];
		
		ev_cb = new TAUReader();
		tFileEvRead=TraceFactory.OpenFileForInput(tautrc,tauedf);
		tFileEvRead.setSubtractFirstTimestamp(true);
		//tFileDefRead.setSubtractFirstTimestamp(false);
		return true;
	}

	public boolean close()
	{
		tFileEvRead.closeTrace();
		return true;
	}

	public int peekNextKindIndex()
	{
		if(categories.size()>0)//maxcats<categories.size())//numcats)
		{
			statedef = (Category)categories.remove(categories.size()-1);// .get(new Integer(maxcats));//[maxcats];
			//maxcats++;
			return Kind.CATEGORY_ID;
		}

		if(!doneReading)
		{
			while(!eventReady)
			{
				if(tFileEvRead.readNumEvents(ev_cb, 1, null)==0)
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
				else if(stepsize>0){
					count_read++;
					if(count_read%stepsize==0){
						System.out.println(count_read+" Records read. "+(int)(100*((double)count_read/(double)arch_read))+"% converted");
					}
				}
			}
			eventReady=false;
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
		return prime;
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
		
		public int defUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing){
			
			if(monotonicallyIncreasing==0)
			{
				noMonEvents.add(new Integer(userEventToken));
			}
			
			String name = userEventName;
			if(name.charAt(0)=='"' && name.charAt(name.length()-1)=='"')
				name = name.substring(1,name.length()-1);
			
			logformat.trace.DobjDef newobj= new logformat.trace.DobjDef(userEventToken, name, Topology.EVENT_ID, 
					getcolors.nextInt(256),getcolors.nextInt(256),getcolors.nextInt(256), 255, 20, "Count=%d", null);
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
			Integer glob = new Integer(GlobalID(nodeToken,threadToken));
			if(!eventstack.containsKey(glob))//  eventstack[nodeToken][threadToken]==null
				eventstack.put(glob, new EZStack());  //[nodeToken][threadToken]=new Stack();
			((EZStack)eventstack.get(glob)).push(new PrimEvent(time,stateToken));//eventstack[nodeToken][threadToken].push(new PrimEvent(time,stateToken));
			return 0;
		}
		
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){
			//System.out.println("Leaving state "+stateToken+" time "+time+" nid "+nodeToken+" tid "+threadToken);
			Integer glob = new Integer(GlobalID(nodeToken,threadToken));
			if(!eventstack.containsKey(glob)||((EZStack)eventstack.get(glob)).size()==0)//if(eventstack[nodeToken][threadToken]==null||eventstack[nodeToken][threadToken].size()==0)
			{
				System.err.println("Fault: Exit from empty or uninitialized thread");
				System.exit(1);
			}
			PrimEvent leave = (PrimEvent) (((EZStack)eventstack.get(glob)).pop());
			if(leave.stateToken!=stateToken)
			{
				System.err.println("Fault: Event order failure.");
				System.exit(1);
			}
			prime = new base.drawable.Primitive(stateToken,leave.time*clockP,time*clockP,
					new double[]{leave.time*clockP,time*clockP} ,new int[]{GlobalID(nodeToken,threadToken),GlobalID(nodeToken,threadToken)},null);//((Integer)global.get(new Point(nodeToken,threadToken))).intValue(),((Integer)global.get(new Point(nodeToken,threadToken))).intValue()
			eventReady=true;
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
		
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){

			/*
			 * The 'tag space' and 'eventId space' may overlap and overwrite each other
			 * So add the highest tag ID+1, to be sure all IDs sent to jumpshot are unique.
			 * */
			int messageTagShift=messageTag+maxEvtId+1;
			/*
			 * If we have a new tag ID we need to register a new object type for it
			 * (Is there a faster way to do this?)
			 */
			if(msgtags.add(new Integer(messageTag)))
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
			Long srcDst=new Long(SourceDest(sourceGlob,destGlob));
			
			MessageEvent sndE=new MessageEvent(time,messageTag,sourceGlob,destGlob,messageSize,messageComm);
			
			MessageEvent leave;
			/*
			 * If we have a message event in the recieved list then there has been a recieve sent before its send.
			 * Match this send with the advanced recieve and submit the event.  (Is this always correct?)
			 */
			if(msgRecStack.containsKey(srcDst) && ((LinkedList)msgRecStack.get(srcDst)).size()>0 &&(leave=(MessageEvent)((LinkedList)msgRecStack.get(srcDst)).removeFirst())!=null)
			{
					System.out.println("Reversed Message: time "+time+", source nid "+ sourceNodeToken +
							" tid "+sourceThreadToken+", destination nid "+destinationNodeToken+" tid "+
							destinationThreadToken+", size "+messageSize+", tag "+messageTag);
					prime = new base.drawable.Primitive(messageTagShift,leave.time*clockP,time*clockP,
							new double[]{time*clockP,leave.time*clockP,} ,
							new int[]{sourceGlob,destGlob},
							getInfoVals(new int[]{messageTag,messageSize,messageComm}));
					eventReady=true;
					return 0;
			}
			
			/*
			 * If we do not yet have an event source-list list for this transaction ID, create one
			 * Put this event in the source list
			 */
			if(!msgSenStack.containsKey(srcDst))
				msgSenStack.put(srcDst, new LinkedList());
			((LinkedList)msgSenStack.get(srcDst)).add(sndE);
			return 0;
		}
		
		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){
			
			int messageTagShift=messageTag+maxEvtId+1;
			int sourceGlob = GlobalID(sourceNodeToken,sourceThreadToken);
			int destGlob = GlobalID(destinationNodeToken,destinationThreadToken);
			Long srcDst=new Long(SourceDest(sourceGlob,destGlob));
			
			MessageEvent recE=new MessageEvent(time,messageTag, sourceGlob,destGlob,messageSize,messageComm);
			
			MessageEvent leave;
			if(msgSenStack.containsKey(srcDst) && ((LinkedList)msgSenStack.get(srcDst)).size()>0 && (leave=(MessageEvent)((LinkedList)msgSenStack.get(srcDst)).removeFirst())!=null)
			{
					prime = new base.drawable.Primitive(messageTagShift,leave.time*clockP,time*clockP,
							new double[]{leave.time*clockP,time*clockP},
							new int[]{
							sourceGlob,destGlob
							},
							getInfoVals(new int[]{messageTag,messageSize,messageComm}));
					eventReady=true;
					return 0;
			}
			
			if(!msgRecStack.containsKey(srcDst))
				msgRecStack.put(srcDst, new LinkedList());
			((LinkedList)msgRecStack.get(srcDst)).add(recE);
			return 0;
		}
		
		public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken,
				double userEventValue){
			//System.out.println("EventTrigger: time "+time+", nid "+nodeToken+" tid "+threadToken+" event id "+userEventToken+" triggered value "+userEventValue);
			if(noMonEvents.contains(new Integer(userEventToken)))
			{
				//System.out.println(noMonEventCycle[nodeToken][threadToken]);
				
				Integer glob = new Integer(GlobalID(nodeToken,threadToken));
				
				if(!noMonEventCycle.containsKey(glob))noMonEventCycle.put(glob, ZERO);
				
				if(noMonEventCycle.get(glob).equals(ZERO))
				{
					noMonEventCycle.put(glob, ONE);
					return 0;
				}
				else
				if(noMonEventCycle.get(glob).equals(ONE))
				{
					noMonEventCycle.put(glob, TWO);
				}
				else
				if(noMonEventCycle.get(glob).equals(TWO))
				{
					noMonEventCycle.put(glob, ZERO);
					return 0;
				}
			}
			prime = new base.drawable.Primitive(userEventToken,time*clockP,time*clockP,
					new double[]{time*clockP} ,new int[]{
					GlobalID(nodeToken,threadToken)
					//((Integer)global.get(new Point(nodeToken,threadToken))).intValue()
					},
					getInfoVals(new int[]{(int)userEventValue}));
			eventReady=true;
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

		public int endTrace(Object userData, int nodeToken, int threadToken) {return 0;}
	}
}
