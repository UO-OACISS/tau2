/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

/*This object contains the callback classes required by the trace reader*/
package edu.uoregon.tau.trace;
public interface TraceReaderCallbacks {
	
	public int defClkPeriod(Object userData, double clkPeriod);
	
	public int defThread(Object userData, int nodeToken, int threadToken, String threadName);
	
	public int defStateGroup(Object userData, int stateGroupToken, String stateGroupName);
	
	public int defState(Object userData, int stateToken, String stateName, int stateGoupToken);
	
	public int defUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing);

	public int enterState(Object userData, long time, int nodeToken, int threadToken, int stateToken);

	public int leaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken);
	
	public int sendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom);
	
	public int recvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom);
	
	public int eventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken,
				double userEventValue);
	
	public int endTrace(Object userData, int nodeToken, int threadToken);
}
