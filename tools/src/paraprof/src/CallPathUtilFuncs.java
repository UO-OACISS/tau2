/* 
  CallPathUtilFuncs.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  
*/

package paraprof;

import java.util.*;

public class CallPathUtilFuncs{
    public CallPathUtilFuncs(){}

    public static boolean isAvailable(ListIterator l){
	GlobalMappingElement gme = null;
	String s = null;
	boolean result = false;
	while(l.hasNext()){
	    gme = (GlobalMappingElement) l.next();
	    s = gme.getMappingName();
	    if(s.indexOf("=>")>0){
		gme.setCallPathObject(true);
		result = true;
	    }
	}
	return result;
    }

    public static void buildRelations(GlobalMapping gm){
	try{
	    GlobalMappingElement gme1 = null;
	    GlobalMappingElement gme2 = null;
	    GlobalMappingElement gme3 = null;
	    String s = null;
	    String parent = null;
	    String child = null;
	    int location = -1;
	    ParaProfIterator l = null;
	    
	    l = (ParaProfIterator)gm.getMappingIterator(0);
	    while(l.hasNext()){
		gme1 = (GlobalMappingElement) l.next();
		s = gme1.getMappingName();
		location = s.lastIndexOf("=>");
		if(location>0){
		    child = s.substring(location+3,(s.length()-2));
		    s = s.substring(0,location-1);
		    location = s.lastIndexOf("=>");
		    if(location>0){
			parent = s.substring(location+3); 
		    }
		    else
			parent = s;
		    //Update parent/child relationships.
		    gme2 = gm.getGlobalMappingElement(parent,0); 
		    gme3 = gm.getGlobalMappingElement(child,0);
		    gme2.addChild(gme3.getGlobalID(),gme1.getGlobalID());
		    gme3.addParent(gme2.getGlobalID(),gme1.getGlobalID());
		}
	    }
	}
	catch(Exception e){
	    System.out.println(e.toString());
	    e.printStackTrace();
	}
    }
}
