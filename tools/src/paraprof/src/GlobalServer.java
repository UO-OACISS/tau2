/* 
  GlobalServer.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;

public class GlobalServer implements Serializable 
{
  //Constructor.
  public GlobalServer()
  {
    contextList = new Vector();
    numberOfContexts = 0;
  }
  
  //Rest of the public functions.
  public void addContext(GlobalContext inGlobalContextObject)
  {
    //Keeping track of the number of contexts on this server.
    numberOfContexts++;
    //Now add the context to the end of the list ... the default
    //for addElement in a Vector.
    contextList.addElement(inGlobalContextObject);
  }

  public Vector getContextList() //Called by ListModel routines.
  {
    return contextList;
  }
  
  //Instance data.
  Vector contextList;
  int numberOfContexts;
  
}
