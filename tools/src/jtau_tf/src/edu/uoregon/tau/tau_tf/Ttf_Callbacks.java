/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

/*This object contains the callback classes required by the trace reader*/
package edu.uoregon.tau.tau_tf;
public interface Ttf_Callbacks {
	
	public int DefClkPeriod(Object userData, double clkPeriod);
	
	public int DefThread(Object userData, int nodeToken, int threadToken, String threadName);
	
	public int DefStateGroup(Object userData, int stateGroupToken, String stateGroupName);
	
	public int DefState(Object userData, int stateToken, String stateName, int stateGoupToken);
	
	public int DefUserEvent(Object userData, int userEventToken, String userEventName, int monotonicallyIncreasing);

	public int EnterState(Object userData, long time, int nodeToken, int threadToken, int stateToken);

	public int LeaveState(Object userData, long time, int nodeToken, int threadToken, int stateToken);
	
	public int SendMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom);
	
	public int RecvMessage(Object userData, long time, int sourceNodeToken, int sourceThreadToken, 
				int destinationNodeToken, int destinationThreadToken, int messageSize, int messageTag, int messageCom);
	
	public int EventTrigger(Object userData, long time, int nodeToken, int threadToken, int userEventToken,
				double userEventValue);
	
	public int EndTrace(Object userData, int nodeToken, int threadToken);
}
