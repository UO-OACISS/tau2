/* 
   Interpret.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:
   To do: A lot.  This class is still in the design phase.
*/

package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.*;
import java.util.*;
import edu.uoregon.tau.dms.dss.*;

public class Interpreter implements Runnable{

    //####################################
    //Public Section.
    //####################################

    //######
    //Contructors.
    //######
    public Interpreter(){
	super();
    }

    public Interpreter(Vector expressions, boolean graphicsEnvironment){
	super();
	this.expressions = expressions;
    }
    //######
    //End - Contructors.
    //######

    public synchronized void evalExpression(String expression, boolean graphicsEnvironment){
	InterpreterThread interpreterThread = new InterpreterThread(this, expression, threadCount);
	java.lang.Thread thread = new java.lang.Thread(interpreterThread);
	thread.setName(this.threadCount + " - " + expression);
	this.threadCount++;
	thread.start();
    }

    public void evalExpressions(Vector expressions, boolean graphicsEnvironment){
	this.expressions = expressions;
	this.graphicsEnvironment = graphicsEnvironment;
	for(Enumeration e = expressions.elements(); e.hasMoreElements() ;){
	    if(UtilFncs.debug)
		System.out.println("Thread(" + this.threadCount + "). --- Trying to get the lock .... ");
	    
	    /*
	    if(EventQueue.isDispatchThread())
		System.out.println("In method evalExpressions(...) - Is the event dispatch thread? yes");
	    else
		System.out.println("In method evalExpressions(...) - Is the event dispatch thread? no");
	    */
	    
	    this.getLock();
	    if(UtilFncs.debug)
		System.out.println("Thread(" + this.threadCount  + "). --- Lock obtained!");
	    String expression = (String) e.nextElement();
	    InterpreterThread interpreterThread = new InterpreterThread(this, expression, threadCount);
	    java.lang.Thread thread = new java.lang.Thread(interpreterThread);
	    thread.setName(this.threadCount + " - " + expression);
	    this.threadCount++;
	    thread.start();

	    //EventQueue.invokeLater(thread);

	}
    }

    public void run(){
	this.evalExpressions(this.expressions, this.graphicsEnvironment);}

    public synchronized void getLock(){
	while(!this.lock()){
	    try{
		wait();
	    }
	    catch(Exception e){
	    }
	}
    }

    public synchronized boolean lock(){
	boolean lockObtained = false;
	if(this.lock==null){
	    this.lock = java.lang.Thread.currentThread();
	    lockObtained = true;
	}
	return lockObtained;
    }

    public synchronized void unlock(){
	try{
	    if(UtilFncs.debug)
		System.out.println("Setting the lock to null and calling notify!");
	    this.lock = null;
	    notify();
	}
	catch(Exception e){
	}
    }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}
    //####################################
    //End - Public Section.
    //####################################

    //####################################
    //Protected Section.
    //####################################
    //####################################
    //End - Protected Section.
    //####################################

    //####################################
    //Private Section.
    //####################################
    private Vector expressions = null;
    private boolean graphicsEnvironment = false;

    private int threadCount = 0;

    private java.lang.Thread lock = null;
    private boolean debug = false;
    //####################################
    //End - Private Section.
    //####################################
}

class InterpreterThread implements Runnable{
    //####################################
    //Public Section.
    //####################################

    //######
    //Contructors.
    //######
    public InterpreterThread(){
	super();
    }

    public InterpreterThread(Interpreter interpreter, String expression, int id){
	super();
	this.interpreter = interpreter;
	this.expression = expression;
	this.id = id;
    }
    //######
    //End - Contructors.
    //######

    public void run(){
	ParaProf.paraProfLisp.eval(this.expression);
	interpreter.unlock();
    }

    public void setDebug(boolean debug){
	this.debug = debug;}
    
    public boolean debug(){
	return debug;}

    public String toString(){
	return this.expression;}
    //####################################
    //End - Public Section.
    //####################################

    //####################################
    //Protected Section.
    //####################################
    //####################################
    //End - Protected Section.
    //####################################

    //####################################
    //Private Section.
    //####################################
    Interpreter interpreter = null;
    private String expression = null;
    private int id = 0;
    boolean graphicsEnvironment = false;

    private boolean debug = false;
    //####################################
    //End - Private Section.
    //####################################
}
