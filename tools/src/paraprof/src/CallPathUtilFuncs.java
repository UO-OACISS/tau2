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
	while(l.hasNext()){
	    gme = (GlobalMappingElement) l.next();
	    s = gme.getMappingName();
	    System.out.println(s);
	    if(s.indexOf("=>") > 0)
		return true;
	}
	return false;
    }
}
