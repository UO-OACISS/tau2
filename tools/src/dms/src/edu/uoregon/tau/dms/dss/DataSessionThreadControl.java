/* 
   DataSessionThreadControl.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:
   To do:
*/

package edu.uoregon.tau.dms.dss;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import java.io.*;
import java.util.*;

public class DataSessionThreadControl implements Runnable{

    //####################################
    //Public Section.
    //####################################

    //######
    //Contructors.
    //######
    public DataSessionThreadControl(){
	super();
    }
    //######
    //End - Contructors.
    //######


    public void initialize(DataSession dataSession, Object dataSessionInitializeObject, boolean graphicsEnvironment){
	this.dataSession = dataSession;
	this.dataSessionInitializeObject = dataSessionInitializeObject;
	this.graphicsEnvironment = graphicsEnvironment;
	java.lang.Thread thread = new java.lang.Thread(this);
	thread.start();
    }

    public void run(){

	this.dataSession.initialize(dataSessionInitializeObject);

	if(graphicsEnvironment){
	    //Need to notify observers that we are done.  Be careful here.
	    //It is likely that they will modify swing elements.  Make sure
	    //to dump request onto the event dispatch thread to ensure
	    //safe update of said swing elements.  Remember, swing is not thread
	    //safe for the most part.
	    EventQueue.invokeLater(new Runnable(){
		    public void run(){
			DataSessionThreadControl.this.notifyObservers();
		    }
		});
	}
	else
	    this.notifyObservers();
    }

    //######
    //Methods that manage the ParaProfObservers.
    //######
    public void addObserver(ParaProfObserver observer){
	observers.add(observer);}

    public void removeObserver(ParaProfObserver observer){
	observers.remove(observer);}

    public void notifyObservers(){
	if(this.debug()){
	    System.out.println("######");
	    System.out.println("ParaProfDataSession.notifyObservers()");
	    System.out.println("Listening classes ...");
	    for(Enumeration e = observers.elements(); e.hasMoreElements() ;)
		System.out.println(e.nextElement().getClass());
	    System.out.println("######");
	}
	for(Enumeration e = observers.elements(); e.hasMoreElements() ;)
	    ((ParaProfObserver) e.nextElement()).update(this);
    }
    //######
    //End - Methods that manage the ParaProfObservers.
    //######

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
    DataSession dataSession = null;
    Object dataSessionInitializeObject = null;
    boolean graphicsEnvironment = false;

    private Vector observers = new Vector();
    private boolean debug = false;
    //####################################
    //End - Private Section.
    //####################################
}
