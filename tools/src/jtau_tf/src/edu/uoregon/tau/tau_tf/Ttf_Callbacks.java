/*
 *  See TAU License file
 */

/*
 * @author  Wyatt Spear
 */

/*This object contains the callback classes required by the trace reader*/
package edu.uoregon.tau.tau_tf;
public class Ttf_Callbacks {
	public Object UserData; 
	//Ttf_DefClkPeriodT  
	public TAU_tf_reader.Ttf_DefClkPeriod DefClkPeriod;
	//Ttf_DefThreadT     
	public TAU_tf_reader.Ttf_DefThread DefThread;
	//Ttf_DefStateGroupT 
	public TAU_tf_reader.Ttf_DefStateGroup DefStateGroup;
	//Ttf_DefStateT      
	public TAU_tf_reader.Ttf_DefState DefState;
	//Ttf_EndTraceT      
	public TAU_tf_reader.Ttf_EndTrace EndTrace;

	//Ttf_EnterStateT    
	public TAU_tf_reader.Ttf_EnterState EnterState;
	//Ttf_LeaveStateT    
	public TAU_tf_reader.Ttf_LeaveState LeaveState;
	//Ttf_SendMessageT   
	public TAU_tf_reader.Ttf_SendMessage SendMessage;
	//Ttf_RecvMessageT   
	public TAU_tf_reader.Ttf_RecvMessage RecvMessage;
	//Ttf_DefUserEventT   
	public TAU_tf_reader.Ttf_DefUserEvent DefUserEvent;
	//Ttf_EventTriggerT   
	public TAU_tf_reader.Ttf_EventTrigger EventTrigger;
}
