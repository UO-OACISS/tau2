/*  
  StaticMainWindowData.java


  Title:      ParaProf
  Author:     Robert Bell
  Description:  This object file controls the ordering of all data relating to the
          StaticMainWindow, and its subwindows.  It obtains its data from the
          global StaticSystemData object.  Thus, all requests for data by the
          StaticMainWindow, and its subwindows, are handled by this object only.  
*/

package paraprof;

import java.util.*;
import java.lang.*;

public class StaticMainWindowData
{

  private Trial trial = null;
  
  //The sorted system data lists.  It makes more sense to sort at the beginning.
  //F:Function;N:Name;M:Millisecond;D:Descending;A:Ascending;E:Exclusive;I:Inclusive;Id:Function ID
  
  private Vector sMWGeneralData = new Vector();
  private Vector sMWMeanData = new Vector();
  
  private String currentlySortedAsMeanTotalStatWindow;
  
  public StaticMainWindowData(Trial inTrial)
  {
    trial = inTrial;
  }
  
  //********************************
  //
  //Functions that create the StaticMainWindowData lists.
  //
  //********************************
  
  public void buildStaticMainWindowDataLists()
  {
    buildSMWGeneralData();
    buildSMWMeanList();
  }
  
  
  private void buildSMWGeneralData()
  {
    
    //********************************
    //This function builds the server, context, and thread list for
    //the default static main window displays.
    //
    //
    //Note:
    //The extensions of the global server, global context, and global thread
    //objects are specific to this display structure.  Unless you are happy
    //with how the drawing data is stored in these extensions, you should
    //use your own custom ones.
    //********************************
    
    
    //Copy data to the appropriate list with the appropriate sorting.
    GlobalServer tmpGlobalServer;
    GlobalContext tmpGlobalContext;
    GlobalThread tmpGlobalThread;
    GlobalThreadDataElement tmpGlobalThreadDataElement;
    
    SMWServer tmpSMWServer;
    SMWContext tmpSMWContext;
    SMWThread tmpSMWThread;
    SMWThreadDataElement tmpSMWThreadDataElement;
    SMWThreadDataElement tmpSMWUserThreadDataElement;
    
    
    Vector tmpContextList;
    Vector tmpThreadList;
    Vector tmpThreadDataList;
    
    //Get a reference to the global data.
    Vector tmpVector = trial.getStaticServerList();
    
    //Clear the sMWGeneralData list for safety.
    sMWGeneralData.removeAllElements();
    
    for(Enumeration e1 = tmpVector.elements(); e1.hasMoreElements() ;)
    {
      tmpGlobalServer = (GlobalServer) e1.nextElement();
      //Create a new sMWServer object and set the name properly.
      tmpSMWServer = new SMWServer();
      //Add the server.
      sMWGeneralData.addElement(tmpSMWServer);
      
      //Enter the context loop for this server.
      tmpContextList = tmpGlobalServer.getContextList();
        
      for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
      {
        tmpGlobalContext = (GlobalContext) e2.nextElement();
        
        //Create a new context object and set the name properly.
        tmpSMWContext = new SMWContext();
        //Add to the server.
        tmpSMWServer.addContext(tmpSMWContext);
        
          
        //Enter the thread loop for this context.
        tmpThreadList = tmpGlobalContext.getThreadList();
        for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
        {
          tmpGlobalThread = (GlobalThread) e3.nextElement();
          
          //Create a new thread object.
          tmpSMWThread = new SMWThread();
          //Add to the context.
          tmpSMWContext.addThread(tmpSMWThread);
          
          //Now enter the thread loop for this thread.
          tmpThreadDataList = tmpGlobalThread.getThreadDataList();
          for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
          {
            tmpGlobalThreadDataElement = (GlobalThreadDataElement) e4.nextElement();
            
            //Only want to add an element if this mapping existed on this thread.
            //Check for this.
            if(tmpGlobalThreadDataElement != null)
            {
              //Create a new thread data object.
              tmpSMWThreadDataElement = new SMWThreadDataElement(trial, tmpGlobalThreadDataElement);
              
              tmpSMWThreadDataElement.setMappingID(tmpGlobalThreadDataElement.getMappingID());
              
              //Add to the thread data object.
              tmpSMWThread.addThreadDataElement(tmpSMWThreadDataElement);
            }
          }
        }
      }
    }
  }
  
  private void buildSMWMeanList()
  {
    //First, grab the global mapping element list.
    GlobalMapping tmpGlobalMapping = trial.getGlobalMapping();
    
    Vector tmpVector = tmpGlobalMapping.getMapping(0);
    
    //Clear the sMWMeanData for safety.
    sMWMeanData.removeAllElements();
    
    //Now cycle through, building our new list.
    for(Enumeration e1 = tmpVector.elements(); e1.hasMoreElements() ;)
    {
      GlobalMappingElement tmpGME = (GlobalMappingElement) e1.nextElement();
      
      if(tmpGME.getMeanValuesSet())
      {
        //Create a new mean data element.
        SMWMeanDataElement tmpSMWMeanDataElement = new SMWMeanDataElement(trial);

        tmpSMWMeanDataElement.setMappingID(tmpGME.getGlobalID());
        tmpSMWMeanDataElement.setValue(tmpGME.getMeanExclusiveValue(trial.getCurValLoc()));
        tmpSMWMeanDataElement.setSortByValue();
        tmpSMWMeanDataElement.setSortByReverse(true);
        
        sMWMeanData.addElement(tmpSMWMeanDataElement);
      }   
    }
    
    //Now sort it.
    Collections.sort(sMWMeanData);
  
  }
  
  //********************************
  //
  //End - Functions that create the StaticMainWindowData lists.
  //
  //********************************
  
  
  //********************************
  //
  //Functions that return various sorted version of the data lists.
  //
  //********************************
  public Vector getSMWMappingData(int inMappingID)
  {
    Vector tmpVector = new Vector();
    Vector rtnVector = new Vector();
  
    try{
      GlobalServer tmpGlobalServer;
      GlobalContext tmpGlobalContext;
      GlobalThread tmpGlobalThread;
      GlobalThreadDataElement tmpGlobalThreadDataElement;
      
      SMWServer tmpSMWServer;
      SMWContext tmpSMWContext;
      SMWThread tmpSMWThread;
      SMWThreadDataElement tmpSMWThreadDataElement;
      SMWThreadDataElement tmpSMWUserThreadDataElement;
      
      
      Vector tmpContextList;
      Vector tmpThreadList;
      Vector tmpThreadDataList;
      
      
      //Get a reference to the global data.
      tmpVector = trial.getStaticServerList();
      
      for(Enumeration e1 = tmpVector.elements(); e1.hasMoreElements() ;)
      {
        tmpGlobalServer = (GlobalServer) e1.nextElement();
        //Create a new sMWServer object and set the name properly.
        tmpSMWServer = new SMWServer();
        //Add the server.
        rtnVector.addElement(tmpSMWServer);
        
        //Enter the context loop for this server.
        tmpContextList = tmpGlobalServer.getContextList();
          
        for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
        {
          tmpGlobalContext = (GlobalContext) e2.nextElement();
          
          //Create a new context object and set the name properly.
          tmpSMWContext = new SMWContext();
          //Add to the server.
          tmpSMWServer.addContext(tmpSMWContext);
          
            
          //Enter the thread loop for this context.
          tmpThreadList = tmpGlobalContext.getThreadList();
          for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
          {
            tmpGlobalThread = (GlobalThread) e3.nextElement();
            
            //Create a new thread object.
            tmpSMWThread = new SMWThread();
            //Add to the context.
            tmpSMWContext.addThread(tmpSMWThread);
            
            //Now enter the thread loop for this thread.
            tmpThreadDataList = tmpGlobalThread.getThreadDataList();
            
            //Only want to add an the element with the correct mapping id.
            tmpGlobalThreadDataElement = (GlobalThreadDataElement) tmpThreadDataList.elementAt(inMappingID);
            
            if(tmpGlobalThreadDataElement != null)
            {
              //Create a new thread data object.
              tmpSMWThreadDataElement = new SMWThreadDataElement(trial, tmpGlobalThreadDataElement);
              
              tmpSMWThreadDataElement.setMappingID(tmpGlobalThreadDataElement.getMappingID());
              
              //Add to the thread data object.
              tmpSMWThread.addThreadDataElement(tmpSMWThreadDataElement);
            }
          }
        }
      }
    }
    catch(Exception e)
    {
      ParaProf.systemError(e, null, "SMWD01");
    }
    return rtnVector;
  }
  
  public Vector getSMWUserEventData(int inMappingID)
  {
    Vector tmpVector = new Vector();
    Vector rtnVector = new Vector();
    
    
    GlobalServer tmpGlobalServer;
    GlobalContext tmpGlobalContext;
    GlobalThread tmpGlobalThread;
    GlobalThreadDataElement tmpGlobalThreadDataElement;
    
    SMWServer tmpSMWServer;
    SMWContext tmpSMWContext;
    SMWThread tmpSMWThread;
    SMWThreadDataElement tmpSMWThreadDataElement;
    SMWThreadDataElement tmpSMWUserThreadDataElement;
    
    
    Vector tmpContextList;
    Vector tmpThreadList;
    Vector tmpThreadDataList;
    
    
    //Get a reference to the global data.
    tmpVector = trial.getStaticServerList();
    
    for(Enumeration e1 = tmpVector.elements(); e1.hasMoreElements() ;)
    {
      tmpGlobalServer = (GlobalServer) e1.nextElement();
      //Create a new sMWServer object and set the name properly.
      tmpSMWServer = new SMWServer();
      //Add the server.
      rtnVector.addElement(tmpSMWServer);
      
      //Enter the context loop for this server.
      tmpContextList = tmpGlobalServer.getContextList();
        
      for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
      {
        tmpGlobalContext = (GlobalContext) e2.nextElement();
        
        //Create a new context object and set the name properly.
        tmpSMWContext = new SMWContext();
        //Add to the server.
        tmpSMWServer.addContext(tmpSMWContext);
        
          
        //Enter the thread loop for this context.
        tmpThreadList = tmpGlobalContext.getThreadList();
        for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
        {
          tmpGlobalThread = (GlobalThread) e3.nextElement();
          
          //Create a new thread object.
          tmpSMWThread = new SMWThread();
          //Add to the context.
          tmpSMWContext.addThread(tmpSMWThread);
          
          //Now enter the thread loop for this thread.
          tmpThreadDataList = tmpGlobalThread.getUserThreadDataList();
          
          //Only want to add an the element with the correct mapping id.
          tmpGlobalThreadDataElement = (GlobalThreadDataElement) tmpThreadDataList.elementAt(inMappingID);
          
          if(tmpGlobalThreadDataElement != null)
          {
            //Create a new thread data object.
            tmpSMWThreadDataElement = new SMWThreadDataElement(trial, tmpGlobalThreadDataElement);
            
            tmpSMWThreadDataElement.setMappingID(tmpGlobalThreadDataElement.getUserEventID());
            
            //Add to the thread data object.
            tmpSMWThread.addThreadDataElement(tmpSMWThreadDataElement);
          }
        }
      }
    }
    
    return rtnVector;
  }
  
  public Vector getSMWThreadData(int inServer, int inContext, int inThread, String inString)
  {
    //Return a copy of the requested data, sorted in the appropriate manner.
    int sortSetting = 0;
    int metric = 0;
    boolean isExclusive = true;
    //Check to see if selected groups only are being displayed.
    GlobalMapping tmpGM = trial.getGlobalMapping();
    
    boolean isSelectedGroupOn = false;
    int selectedGroupID = 0;
    
    if(tmpGM.getIsSelectedGroupOn()){
      isSelectedGroupOn = true;
      selectedGroupID = tmpGM.getSelectedGroupID();
    }
    
    //First, obtain the appropriate server.
    SMWServer tmpSMWServer = (SMWServer) sMWGeneralData.elementAt(inServer);
    Vector tmpContextList = tmpSMWServer.getContextList();
    SMWContext tmpSMWContext = (SMWContext) tmpContextList.elementAt(inContext);
    Vector tmpThreadList = tmpSMWContext.getThreadList();
    SMWThread tmpSMWThread = (SMWThread) tmpThreadList.elementAt(inThread);
    Vector tmpThreadDataList = tmpSMWThread.getThreadDataList();
    
    //Ok, now that I have the appropriate thread, copy it and then sort the copy in the appropriate manner.
    Vector tmpVector = new Vector();
    SMWThreadDataElement tmpSMWThreadDataElement;
    SMWThreadDataElement tmpSMWThreadDataElementCopy;
    
      
    //This section needs some work!!  Should be able to find a better system.  Perhaps additive or something.
    
    if(inString.equals("FIdDE")){
      sortSetting = 1;
      metric = 1;}
    else if(inString.equals("FIdDI")){
      sortSetting = 1;
      metric = 2;}
    else if(inString.equals("FIdAE")){
      sortSetting = 2;
      metric = 1;}
    else if(inString.equals("FIdAI")){
      sortSetting = 2;
      metric = 2;}
    else if(inString.equals("NDE")){
      sortSetting = 3;
      metric = 1;}
    else if(inString.equals("NDI")){
      sortSetting = 3;
      metric = 2;}
    else if(inString.equals("NAE")){
      sortSetting = 4;
      metric = 1;}
    else if(inString.equals("NAI")){
      sortSetting = 4;
      metric = 2;}
    else if(inString.equals("MDE")){
      sortSetting = 5;
      metric = 1;}
    else if((inString.equals("MDI"))){
      sortSetting = 5;
      metric = 2;}
    else if((inString.equals("MAE"))){
      sortSetting = 6;
      metric = 1;}
    else if((inString.equals("MAI"))){
      sortSetting = 6;
      metric = 2;}
    else if(inString.equals("FIdDNC")){
      sortSetting = 1;
      metric = 3;}
    else if(inString.equals("FIdDNS")){
      sortSetting = 1;
      metric = 4;}
    else if(inString.equals("FIdDUS")){
      sortSetting = 1;
      metric = 5;}
    else if(inString.equals("FIdANC")){
      sortSetting = 2;
      metric = 3;}
    else if(inString.equals("FIdANS")){
      sortSetting = 2;
      metric = 4;}
    else if(inString.equals("FIdAUS")){
      sortSetting = 2;
      metric = 5;}
    else if(inString.equals("NDNC")){
      sortSetting = 3;
      metric = 3;}
    else if(inString.equals("NDNS")){
      sortSetting = 3;
      metric = 4;}
    else if(inString.equals("NDUS")){
      sortSetting = 3;
      metric = 5;}
    else if(inString.equals("NANC")){
      sortSetting = 4;
      metric = 3;}
    else if(inString.equals("NANS")){
      sortSetting = 4;
      metric = 4;}
    else if(inString.equals("NAUS")){
      sortSetting = 4;
      metric = 5;}
    else if(inString.equals("MDNC")){
      sortSetting = 5;
      metric = 3;}
    else if((inString.equals("MDNS"))){
      sortSetting = 5;
      metric = 4;}
    else if((inString.equals("MDUS"))){
      sortSetting = 5;
      metric = 5;}
    else if((inString.equals("MANC"))){
      sortSetting = 6;
      metric = 3;}
    else if((inString.equals("MANS"))){
      sortSetting = 6;
      metric = 4;}
    else{
      sortSetting = 6;
      metric = 5;}
    
    if(!isSelectedGroupOn){
        for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
        {
          tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
          //Create a new thread data object.
          tmpSMWThreadDataElementCopy = new SMWThreadDataElement(trial, tmpSMWThreadDataElement.getGTDE());
          
          tmpSMWThreadDataElementCopy.setMappingID(tmpSMWThreadDataElement.getMappingID());
          
          switch(metric){
            case(1):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getExclusiveValue());
              break;
            case(2):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getInclusiveValue());
              break;
            case(3):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getNumberOfCalls());
              break;
            case(4):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getNumberOfSubRoutines());
              break;
            case(5):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getUserSecPerCall());
              break;
            default:
              tmpSMWThreadDataElementCopy.setValue(0);
              break;
          }
          tmpSMWThreadDataElementCopy.setSortSetting(sortSetting);
          tmpVector.addElement(tmpSMWThreadDataElementCopy);
        }
      }
      else{
        for(Enumeration e1 = tmpThreadDataList.elements(); e1.hasMoreElements() ;)
        {
          tmpSMWThreadDataElement = (SMWThreadDataElement) e1.nextElement();
          if(tmpSMWThreadDataElement.isGroupMember(selectedGroupID)){
            //Create a new thread data object.
            tmpSMWThreadDataElementCopy = new SMWThreadDataElement(trial, tmpSMWThreadDataElement.getGTDE());
          
            tmpSMWThreadDataElementCopy.setMappingID(tmpSMWThreadDataElement.getMappingID());
            switch(metric){
            case(1):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getExclusiveValue());
              break;
            case(2):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getInclusiveValue());
              break;
            case(3):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getNumberOfCalls());
              break;
            case(4):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getNumberOfSubRoutines());
              break;
            case(5):
              tmpSMWThreadDataElementCopy.setValue(tmpSMWThreadDataElement.getUserSecPerCall());
              break;
            default:
              tmpSMWThreadDataElementCopy.setValue(0);
              break;
          }
            tmpSMWThreadDataElementCopy.setSortSetting(sortSetting);
            tmpVector.addElement(tmpSMWThreadDataElementCopy);
          }
        }
      }
    
    Collections.sort(tmpVector);
    return tmpVector;
  }
  
  public Vector getSMWUEThreadData(int inServer, int inContext, int inThread)
  {
    //First, obtain the appropriate server.
    Vector tmpVector = trial.getStaticServerList();
    
    //Find the correct global thread data element.
    GlobalServer tmpGSUE = null;
    Vector tmpGlobalContextListUE = null;
    GlobalContext tmpGCUE = null;
    Vector tmpGlobalThreadListUE = null;
    GlobalThread tmpGTUE = null;
    Vector tmpGlobalThreadDataElementListUE = null;
    
    //Find the correct global thread data element.
    tmpGSUE = (GlobalServer) tmpVector.elementAt(inServer);
    tmpGlobalContextListUE = tmpGSUE.getContextList();
    tmpGCUE = (GlobalContext) tmpGlobalContextListUE.elementAt(inContext);
    tmpGlobalThreadListUE = tmpGCUE.getThreadList();
    tmpGTUE = (GlobalThread) tmpGlobalThreadListUE.elementAt(inThread);
    
    tmpGlobalThreadDataElementListUE = tmpGTUE.getUserThreadDataList();
    
    
    //Ok, now that I have the appropriate thread, copy it and then sort the copy in the appropriate manner.
    Vector returnVector = new Vector();
    SMWThreadDataElement tmpSMWThreadDataElement;
    GlobalThreadDataElement tmpGlobalThreadDataElement = null;
    
    for(Enumeration e1 = tmpGlobalThreadDataElementListUE.elements(); e1.hasMoreElements() ;)
    {
      tmpGlobalThreadDataElement = (GlobalThreadDataElement) e1.nextElement();
      if(tmpGlobalThreadDataElement != null)
      {
        //Create a new thread data object.
        tmpSMWThreadDataElement = new SMWThreadDataElement(trial, tmpGlobalThreadDataElement);
        tmpSMWThreadDataElement.setMappingID(tmpGlobalThreadDataElement.getUserEventID());
        //Add to the thread data object.
        returnVector.add(tmpSMWThreadDataElement);
      }
    }
    
    return returnVector;
  }
  
  public Vector getSMWGeneralData(String inString){
    
    int sortSetting = 0;
    boolean isExclusive = true;
    
    SMWServer tmpSMWServer;
    SMWContext tmpSMWContext;
    SMWThread tmpSMWThread;
    SMWThreadDataElement tmpSMWThreadDataElement;
    
    Vector tmpContextList;
    Vector tmpThreadList;
    Vector tmpThreadDataList;
    
    if(inString == null)
    {
      //Just return the current list as the caller does not care as to the order.
      return sMWGeneralData;
    }
    
    if(inString.equals("FIdDE")){
      sortSetting = 1;}
    else if(inString.equals("FIdDI")){
      sortSetting = 1;
      isExclusive = false;}
    else if(inString.equals("FIdAE")){
      sortSetting = 2;}
    else if(inString.equals("FIdAI")){
      sortSetting = 2;
      isExclusive = false;}
    else if(inString.equals("NDE")){
      sortSetting = 3;}
    else if(inString.equals("NDI")){
      sortSetting = 3;
      isExclusive = false;}
    else if(inString.equals("NAE")){
      sortSetting = 4;}
    else if(inString.equals("NAI")){
      sortSetting = 4;
      isExclusive = false;}
    else if(inString.equals("MDE")){
      sortSetting = 5;}
    else if((inString.equals("MDI"))){
      sortSetting = 5;
      isExclusive = false;}
    else if((inString.equals("MAE"))){
      sortSetting = 6;}
    else{
      sortSetting = 6;
      isExclusive = false;}
    
    for(Enumeration e1 = sMWGeneralData.elements(); e1.hasMoreElements() ;){
      tmpSMWServer = (SMWServer) e1.nextElement();
      
      //Enter the context loop for this server.
      tmpContextList = tmpSMWServer.getContextList();
        
      for(Enumeration e2 = tmpContextList.elements(); e2.hasMoreElements() ;)
      {
        tmpSMWContext = (SMWContext) e2.nextElement();
          
        //Enter the thread loop for this context.
        tmpThreadList = tmpSMWContext.getThreadList();
        
        for(Enumeration e3 = tmpThreadList.elements(); e3.hasMoreElements() ;)
        {
           tmpSMWThread = (SMWThread) e3.nextElement();
           
          //Now enter the thread loop for this thread.
          tmpThreadDataList = tmpSMWThread.getThreadDataList();
          
          for(Enumeration e4 = tmpThreadDataList.elements(); e4.hasMoreElements() ;)
          {
            tmpSMWThreadDataElement = (SMWThreadDataElement) e4.nextElement();
            if(isExclusive)
              tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getExclusiveValue());
            else
              tmpSMWThreadDataElement.setValue(tmpSMWThreadDataElement.getInclusiveValue());
            tmpSMWThreadDataElement.setSortSetting(sortSetting);
          }
          
          //Now, sort this thread list.
          Collections.sort(tmpThreadDataList);
        }
      }
    }
          
    return sMWGeneralData;
  }
  
  public Vector getSMWMeanData(String inString)
  { 
    Vector tmpVector = new Vector();
    SMWMeanDataElement tmpSMWMeanDataElement;
    SMWMeanDataElement tmpSMWMeanDataElementCopy;
    
    int sortSetting = 0;
    int metric = 0;
    
    //Check to see if selected groups only are being displayed.
    GlobalMapping tmpGM = trial.getGlobalMapping();
    
    boolean isSelectedGroupOn = false;
    int selectedGroupID = 0;
    
    if(tmpGM.getIsSelectedGroupOn()){
      isSelectedGroupOn = true;
      selectedGroupID = tmpGM.getSelectedGroupID();
    }
    
    if(inString == null)
    {
      //Just return the current list as the caller does not care as to the order.
      return sMWMeanData;
    }
    
    //This section needs some work!!  Should be able to find a better system.  Perhaps additive or something.
    
    if(inString.equals("FIdDE")){
      sortSetting = 1;
      metric = 1;}
    else if(inString.equals("FIdDI")){
      sortSetting = 1;
      metric = 2;}
    else if(inString.equals("FIdAE")){
      sortSetting = 2;
      metric = 1;}
    else if(inString.equals("FIdAI")){
      sortSetting = 2;
      metric = 2;}
    else if(inString.equals("NDE")){
      sortSetting = 3;
      metric = 1;}
    else if(inString.equals("NDI")){
      sortSetting = 3;
      metric = 2;}
    else if(inString.equals("NAE")){
      sortSetting = 4;
      metric = 1;}
    else if(inString.equals("NAI")){
      sortSetting = 4;
      metric = 2;}
    else if(inString.equals("MDE")){
      sortSetting = 5;
      metric = 1;}
    else if((inString.equals("MDI"))){
      sortSetting = 5;
      metric = 2;}
    else if((inString.equals("MAE"))){
      sortSetting = 6;
      metric = 1;}
    else if((inString.equals("MAI"))){
      sortSetting = 6;
      metric = 2;}
    else if(inString.equals("FIdDNC")){
      sortSetting = 1;
      metric = 3;}
    else if(inString.equals("FIdDNS")){
      sortSetting = 1;
      metric = 4;}
    else if(inString.equals("FIdDUS")){
      sortSetting = 1;
      metric = 5;}
    else if(inString.equals("FIdANC")){
      sortSetting = 2;
      metric = 3;}
    else if(inString.equals("FIdANS")){
      sortSetting = 2;
      metric = 4;}
    else if(inString.equals("FIdAUS")){
      sortSetting = 2;
      metric = 5;}
    else if(inString.equals("NDNC")){
      sortSetting = 3;
      metric = 3;}
    else if(inString.equals("NDNS")){
      sortSetting = 3;
      metric = 4;}
    else if(inString.equals("NDUS")){
      sortSetting = 3;
      metric = 5;}
    else if(inString.equals("NANC")){
      sortSetting = 4;
      metric = 3;}
    else if(inString.equals("NANS")){
      sortSetting = 4;
      metric = 4;}
    else if(inString.equals("NAUS")){
      sortSetting = 4;
      metric = 5;}
    else if(inString.equals("MDNC")){
      sortSetting = 5;
      metric = 3;}
    else if((inString.equals("MDNS"))){
      sortSetting = 5;
      metric = 4;}
    else if((inString.equals("MDUS"))){
      sortSetting = 5;
      metric = 5;}
    else if((inString.equals("MANC"))){
      sortSetting = 6;
      metric = 3;}
    else if((inString.equals("MANS"))){
      sortSetting = 6;
      metric = 4;}
    else{
      sortSetting = 6;
      metric = 5;}
      
    if(!isSelectedGroupOn){
      for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
      {
        tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
        
        tmpSMWMeanDataElementCopy = new SMWMeanDataElement(trial);
        tmpSMWMeanDataElementCopy.setMappingID(tmpSMWMeanDataElement.getMappingID());       
        //Set the sorting method.
        switch(metric){
          case(1):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanExclusiveValue());
            break;
          case(2):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanInclusiveValue());
            break;
          case(3):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanNumberOfCalls());
            break;
          case(4):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanNumberOfSubRoutines());
            break;
          case(5):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanUserSecPerCall());
            break;
          default:
            tmpSMWMeanDataElementCopy.setValue(0);
            break;
        }
        tmpSMWMeanDataElementCopy.setSortSetting(sortSetting);
        
        tmpVector.addElement(tmpSMWMeanDataElementCopy);
        
      }
    }
    else{
      for(Enumeration e1 = sMWMeanData.elements(); e1.hasMoreElements() ;)
      {
        tmpSMWMeanDataElement = (SMWMeanDataElement) e1.nextElement();
        
        tmpSMWMeanDataElementCopy = new SMWMeanDataElement(trial);
        tmpSMWMeanDataElementCopy.setMappingID(tmpSMWMeanDataElement.getMappingID());       
        //Set the sorting method.
        switch(metric){
          case(1):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanExclusiveValue());
            break;
          case(2):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanInclusiveValue());
            break;
          case(3):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanNumberOfCalls());
            break;
          case(4):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanNumberOfSubRoutines());
            break;
          case(5):
            tmpSMWMeanDataElementCopy.setValue(tmpSMWMeanDataElementCopy.getMeanUserSecPerCall());
            break;
          default:
            tmpSMWMeanDataElementCopy.setValue(0);
            break;
        }
        tmpSMWMeanDataElementCopy.setSortSetting(sortSetting);
        
        tmpVector.addElement(tmpSMWMeanDataElementCopy);
      }
    }
    Collections.sort(tmpVector);
    return tmpVector;
  }
}

//********************************
//
//End - Functions that return various sorted version of the data lists.
//
//********************************
