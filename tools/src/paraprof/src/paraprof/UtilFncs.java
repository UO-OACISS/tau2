/* 
  UtilFncs.java

  Title:      ParaProf
  Author:     Robert Bell
  Description:  Some useful functions for the system.
*/

package ParaProf;
import java.util.*;
import java.lang.*;
import java.io.*;
import java.text.*;

public class UtilFncs{
  
  public static double adjustDoublePresision(double inDouble, int precision){
    String result = null;
    
    try{
      String formatString = "##0.0";
      for(int i=0;i<(precision-1);i++){
          formatString = formatString+"0";
        }
        if(inDouble < 0.001){
          for(int i=0;i<4;i++){
          formatString = formatString+"0";
          }
        }
        
        DecimalFormat dF = new DecimalFormat(formatString);
        result = dF.format(inDouble);
      }
      catch(Exception e)
    {
      ParaProf.systemError(e, null, "UF01");
    }
    return Double.parseDouble(result);
    }
    
    public static String getTestString(double inDouble, int precision){
      
      //This method comes up with a rough estimation.  The drawing windows do not
      //need to be absolutely accurate.
      
      String returnString = "";
      for(int i=0;i<precision;i++){
        returnString = returnString + " ";
      }
      
      long tmpLong = Math.round(inDouble);
      returnString = Long.toString(tmpLong) + returnString;
      
      return returnString;
    }
}
