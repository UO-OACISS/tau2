 /* 
   ParaProfLispPrimitives.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  Lisp primatives for Jatha's lisp interpreter.
*/

package edu.uoregon.tau.paraprof;

import java.io.*;
import java.lang.*;
import java.util.*;
import org.jatha.Jatha;
import org.jatha.dynatype.*;
import org.jatha.compile.*;
import org.jatha.machine.*;
import edu.uoregon.tau.dms.dss.*;

public class ParaProfLispPrimitives{

    public ParaProfLispPrimitives(boolean debug){
	super();}

    public static DataSessionIterator getPrimitiveList(Jatha lisp, boolean debug){
	Vector primatives = new Vector();
	primatives.add(new showThreadDataWindow(lisp, debug));
	primatives.add(new GetExclusiveValues(lisp, debug));

	return new DataSessionIterator(primatives);
    }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}

    //####################################
    //Instance data.
    //####################################
    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}

class showThreadDataWindow extends LispPrimitive{
    public showThreadDataWindow(Jatha lisp, boolean debug){
	super(lisp, "SHOWTHREADDATAWINDOW", 6);
	this.lisp = lisp;
    }

    public void Execute(SECDMachine machine){
	LispValue arg6 = machine.S.pop();
	LispValue arg5 = machine.S.pop();
	LispValue arg4 = machine.S.pop();
	LispValue arg3 = machine.S.pop();
	LispValue arg2 = machine.S.pop();
	LispValue arg1 = machine.S.pop();

	machine.S.push(result(arg1, arg2, arg3, arg4, arg5, arg6));
	machine.C.pop();
    }

    LispValue result(LispValue arg1, LispValue arg2, LispValue arg3, LispValue arg4, LispValue arg5, LispValue arg6){
	System.out.println("Applying showThreadDataWindow. Args: " + arg1 + "," + arg2 + "," + arg3 + "," + arg4 + "," + arg5 + "," + arg6);
	ParaProfTrial trial = ParaProf.applicationManager.getTrial(Integer.parseInt(arg1.toString()),
									   Integer.parseInt(arg2.toString()),
									   Integer.parseInt(arg3.toString()));
	System.out.println("Got trial: " + trial);

	ThreadDataWindow  threadDataWindow = new ThreadDataWindow(trial, Integer.parseInt(arg4.toString()),
								  Integer.parseInt(arg5.toString()),
								  Integer.parseInt(arg6.toString()),
								  new StaticMainWindowData(trial, this.debug()),
								  1, this.debug());

	trial.getSystemEvents().addObserver(threadDataWindow);
	threadDataWindow.show();



	/*
	if(arg1.basic_integerp())
	    System.out.println("Integer");
	if(arg1.basic_numberp())
	    System.out.println("Number");
	*/

	
    
	return lisp.makeInteger(1);
    }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}
 
class GetExclusiveValues extends LispPrimitive{
    public GetExclusiveValues(Jatha lisp, boolean debug){
	super(lisp, "GETEXCLUSIVEVALUES", 4);
	this.lisp = lisp;
    }

    public void Execute(SECDMachine machine){
	LispValue arg4 = machine.S.pop();
	LispValue arg3 = machine.S.pop();
	LispValue arg2 = machine.S.pop();
	LispValue arg1 = machine.S.pop();

	machine.S.push(result(arg1, arg2, arg3, arg4));
	machine.C.pop();
    }

    LispValue result(LispValue arg1, LispValue arg2, LispValue arg3, LispValue arg4){
	System.out.println("Applying getTrial. Args: " + arg1 + "," + arg2 + "," + arg3 + "," + arg4);
	Vector v = new Vector();
	
	try{
	    //Get the exclusive values for the given metric.  Put the result in a list, and pass it back.
	    ParaProfTrial trial = ParaProf.applicationManager.getTrial(Integer.parseInt(arg1.toString()),
								       Integer.parseInt(arg2.toString()),
								       Integer.parseInt(arg3.toString()));
	    
	    int metric = Integer.parseInt(arg4.toString());
	    if(trial!=null){
		DataSession dataSession = trial.getDataSession();
		for(Enumeration e1 = dataSession.getNCT().getNodes().elements(); e1.hasMoreElements() ;){
		    Node node = (Node) e1.nextElement();
		    for(Enumeration e2 = node.getContexts().elements(); e2.hasMoreElements() ;){
			Context context = (Context) e2.nextElement();
			for(Enumeration e3 = context.getThreads().elements(); e3.hasMoreElements() ;){
			    edu.uoregon.tau.dms.dss.Thread thread = (edu.uoregon.tau.dms.dss.Thread) e3.nextElement();
			    ListIterator l = thread.getFunctionListIterator();
			    while(l.hasNext()){
				GlobalThreadDataElement globalThreadDataElement = (GlobalThreadDataElement) l.next();
				if(globalThreadDataElement!=null){
				    v.add(lisp.makeReal(globalThreadDataElement.getExclusiveValue(metric)));
				    System.out.println(globalThreadDataElement.getMappingName()+": "+globalThreadDataElement.getExclusiveValue(metric));
				}
			    }
			}
		    }
		}
	    }
	}
	catch(Exception e){
	    System.out.println("An exception was caught in: GetTrial(...)");
	    e.printStackTrace();
	}
	return lisp.makeList(v);
    }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}

    //####################################
    //Instance data.
    //####################################
    Jatha lisp = null;

    private boolean debug = false; //Off by default.
    //####################################
    //End - Instance data.
    //####################################

}   
    
