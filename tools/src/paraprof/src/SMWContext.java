/* 
   SMWContext.java
   
   Title:      ParaProf
   Author:     Robert Bell
   Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;

public class SMWContext{
    
    public SMWContext(){}

    public SMWContext(int contextID){
	this.contextID = contextID;}

    public SMWContext(SMWServer parentNode, int contextID){
	this.parentNode = parentNode;
	this.contextID = contextID;
    }

    public void setParentNode(SMWServer parentNode){
	this.parentNode = parentNode;}

    public SMWServer getParentNode(){
	return parentNode;}

    public void setContextID(int contextID){
	this.contextID = contextID;}

    public int getContextID(){
	return contextID;}
    
    public void addThread(SMWThread inSMWThread){
	//Add the thread to the end of the list ... the default
	//for addElement in a Vector.
	threadList.addElement(inSMWThread);
    }
  
    public Vector getThreadList(){
	return threadList;}
    
    public void setYDrawCoord(int inYDrawCoord){
	yDrawCoord = inYDrawCoord;
    }
    
    public int getYDrawCoord(){
	return yDrawCoord;
    }
    
    //####################################
    //Instance data.
    //####################################
    SMWServer parentNode = null;
    int contextID = -1;
    Vector threadList = new Vector();
    //To aid with drawing searches.
    int yDrawCoord = -1;
    //####################################
    //End - Instance data.
    //####################################
  
}
