/* 
   SMWServer.java
   
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

public class SMWServer
{
    public SMWServer(){}

    public SMWServer(int nodeID){
	this.nodeID = nodeID;}

    public void setNodeID(int nodeID){
	this.nodeID = nodeID;}

    public int getNodeID(){
	return nodeID;}
    
    public void addContext(SMWContext inGlobalSMWContext){
	//Add the context to the end of the list ... the default
	//for addElement in a Vector.
	contextList.addElement(inGlobalSMWContext);
    }
  
    public Vector getContextList(){
	return contextList;}
    
    public void setYDrawCoord(int inYDrawCoord){
	yDrawCoord = inYDrawCoord;}
    
    public int getYDrawCoord(){
	return yDrawCoord;}
  
    //####################################
    //Instance data.
    //####################################
    int nodeID = -1;
    Vector contextList = new Vector();
    //To aid with drawing searches.
    int yDrawCoord = -1;
    //####################################
    //End  - Instance data.
    //####################################
}
