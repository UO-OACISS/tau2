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
  //Constructors.
  public SMWServer()
  {
    contextList = new Vector();
    yDrawCoord = -1;
  }
  
  public void addContext(SMWContext inGlobalSMWContext)
  {
    //Add the context to the end of the list ... the default
    //for addElement in a Vector.
    contextList.addElement(inGlobalSMWContext);
  }
  
  public Vector getContextList() //Called by ListModel routines.
  {
    return contextList;
  }
  
  public void setYDrawCoord(int inYDrawCoord)
  {
    yDrawCoord = inYDrawCoord;
  }
  
  public int getYDrawCoord()
  {
    return yDrawCoord;
  }
  
  //Instance data.
  Vector contextList;
  //To aid with drawing searches.
  int yDrawCoord;
  
}
