/* 
   SMWThread.java

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

public class SMWThread{

    public SMWThread(){}

    public SMWThread(int threadID){
	this.threadID = threadID;}

    public SMWThread(SMWContext parentContext, int threadID){
	this.parentContext = parentContext;
	this.threadID = threadID;
    }

    public void setParentContext(SMWContext parentContext){
	this.parentContext = parentContext;}

    public SMWContext getParentContext(){
	return parentContext;}

    public void setThreadID(int threadID){
	this.threadID = threadID;}
    
    public int getThreadID(){
	return this.threadID;}
  
    public void addFunction(SMWThreadDataElement sMWThreadDataElement){
	functions.addElement(sMWThreadDataElement);}

    public void addUserevent(SMWThreadDataElement sMWThreadDataElement){
	userevents.addElement(sMWThreadDataElement);}

    public Vector getFunctionList(){
	return functions;}

    public ListIterator getFunctionListIterator(){
	return new ParaProfIterator(functions);}
  
    public Vector getUsereventList(){
	return userevents;}

    public ListIterator getUsereventListIterator(){
	return new ParaProfIterator(userevents);}
  
    //Rest of the public functions.
    public void setYDrawCoord(int inYDrawCoord){
	yDrawCoord = inYDrawCoord;
    }
  
    public int getYDrawCoord(){
	return yDrawCoord;
    }
    
    //####################################
    //Instance data.
    //####################################
    SMWContext parentContext = null;
    int threadID = -1;
    Vector functions = new Vector();
    Vector userevents = new Vector();
    //To aid with drawing searches.
    int yDrawCoord = -1;
    //####################################
    //End - Instance data.
    //####################################
}
