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
import java.util.Random;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import base.drawable.*;
import base.io.BufArrayOutputStream;
import edu.uoregon.tau.trace.*;

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
	
	private static Map categories;
	private static int numcats;
	private static int maxcats;
	//private static base.drawable.YCoordMap ymap;
	private static base.drawable.Primitive prime;
	private base.drawable.Category statedef;
	
	//private static Map global;
//	private static Map threadspernode;
	//private static int numthreads;
	private static int maxnode;
	private static int maxthread;
	private static HashMap eventstack;
	private static HashMap msgstack;
	//private static int[] offset;
	
	private static Random getcolors;
	private static String msgtags;
	private static HashMap noMonEventCycle;
	private static Set noMonEvents;
	
	private static int GlobalID(int tid, int nid){
		//System.out.println("n: "+nid+ " t: "+tid);
		return (nid << 16)+tid;
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
		
		categories = new TreeMap();//base.drawable.Category[128];
		numcats=0;
		maxcats=0;
		prime=null;
		clockP=1;
		eventReady=false;
		doneReading=false;
		
		//numthreads=0;
		maxnode=0;
		maxthread=0;
		msgtags="";
		getcolors = new Random(0);
		noMonEvents=new HashSet();
		
		TraceReaderCallbacks def_cb = new TAUReaderInit();
		TraceReader tFileDefRead=TraceFactory.OpenFileForInput(tautrc,tauedf);
		//System.out.println()
		tFileDefRead.setSubtractFirstTimestamp(false);
		tFileDefRead.setDefsOnly(true);
		int recs_read=0;
		int arch_read=0;
		do{
			recs_read=tFileDefRead.readNumEvents(def_cb, 1024,null);
			arch_read+=recs_read;
			//if(recs_read>0)
			//System.out.println("Read "+recs_read+" records");
		}while(recs_read!=0);//&&((Integer)tb.UserData).intValue()!=0
		tFileDefRead.closeTrace();
		
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
		msgstack = new HashMap();// Stack[maxnode+1][maxthread+1];
		noMonEventCycle = new HashMap();//[maxnode+1][maxthread+1];
		
		ev_cb = new TAUReader();
		tFileEvRead=TraceFactory.OpenFileForInput(tautrc,tauedf);
		tFileDefRead.setSubtractFirstTimestamp(false);
		return true;
	}

	public boolean close()
	{
		tFileEvRead.closeTrace();
		return true;
	}

	public int peekNextKindIndex()
	{
		if(maxcats<numcats)
		{
			statedef = (Category)categories.get(new Integer(maxcats));//[maxcats];
			maxcats++;
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
			categories.put(new Integer(numcats), newobj);//[numcats]=newcat;
			numcats++;
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
			categories.put(new Integer(numcats), newobj);
			numcats++;
			return 0;
		}
		
		public int enterState(Object userData, long time, int nodeToken, int threadToken, int stateToken){return 0;}
		public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken){return 0;}
		
		/*Message registration.  (Message sending is defined in TAUReader below)*/
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){return 0;}
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
		
		public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){
			
			

			String tag=Integer.toString(messageTag);
			tag="."+tag+".";
			if(msgtags.indexOf(tag)<0)//msgtags.contains(tag)
			{
				msgtags+=tag;
				logformat.trace.DobjDef newobj= new logformat.trace.DobjDef(messageTag, "message", Topology.ARROW_ID, 
						255,255,255, 255, 3, "msg_tag=%d, msg_size=%d", null);

				categories.put(new Integer(numcats), newobj);
				numcats++;
				//return 0;
			}
			
			Integer sourceGlob = new Integer(GlobalID(sourceNodeToken,sourceThreadToken));
			
			if(!msgstack.containsKey(sourceGlob))msgstack.put(sourceGlob, new EZStack());//[sourceNodeToken][sourceThreadToken]=new Stack();
			if(((EZStack)msgstack.get(sourceGlob)).size()>0)
			{
				if(((PrimEvent)((EZStack)msgstack.get(sourceGlob)).peek()).stateToken==messageTag)
				{
					//We've already seen a recieve for this event!
					PrimEvent leave = (PrimEvent)((EZStack)msgstack.get(sourceGlob)).pop();
					System.out.println("Reversed Message: time "+time+", source nid "+ sourceNodeToken +
							" tid "+sourceThreadToken+", destination nid "+destinationNodeToken+" tid "+
							destinationThreadToken+", size "+messageSize+", tag "+messageTag);
					prime = new base.drawable.Primitive(messageTag,leave.time*clockP,time*clockP,
							new double[]{time*clockP,leave.time*clockP,} ,
							new int[]{GlobalID(sourceNodeToken,sourceThreadToken),GlobalID(destinationNodeToken,destinationThreadToken)},//((Integer)global.get(new Point(sourceNodeToken,sourceThreadToken))).intValue(),((Integer)global.get(new Point(destinationNodeToken,destinationThreadToken))).intValue()
							getInfoVals(new int[]{messageTag,messageSize}));
					eventReady=true;
					return 0;
				}
			}
			((EZStack)msgstack.get(sourceGlob)).push(new PrimEvent(time,messageTag));
			return 0;
		}
		
		public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageComm){
			
			Integer sourceGlob = new Integer(GlobalID(sourceNodeToken,sourceThreadToken));
			
			if(!msgstack.containsKey(sourceGlob))msgstack.put(sourceGlob, new EZStack());//[sourceNodeToken][sourceThreadToken]=new Stack();
			PrimEvent leave;
			if(((EZStack)msgstack.get(sourceGlob)).size()>0)
			{
				if(((PrimEvent)(((EZStack)msgstack.get(sourceGlob)).peek())).stateToken==messageTag)
				{
					leave = (PrimEvent)((EZStack)msgstack.get(sourceGlob)).pop();
					prime = new base.drawable.Primitive(messageTag,leave.time*clockP,time*clockP,
							new double[]{leave.time*clockP,time*clockP},
							new int[]{
							GlobalID(sourceNodeToken,sourceThreadToken),GlobalID(destinationNodeToken,destinationThreadToken)
							//((Integer)global.get(new Point(sourceNodeToken,sourceThreadToken))).intValue(),((Integer)global.get(new Point(destinationNodeToken,destinationThreadToken))).intValue()
							},
							getInfoVals(new int[]{messageTag,messageSize}));
					eventReady=true;
					return 0;
				}
			}
			((EZStack)msgstack.get(sourceGlob)).push(new PrimEvent(time,messageTag));
			
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
