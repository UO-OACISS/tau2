/* 
   ParaProfLisp.java

   Title:      ParaProf
   Author:     Robert Bell
   Description:  This class handles all lisp activity with help from Jatha.
   Ref: http://jatha.sourceforge.net/
*/

/*
  To do: 
  1) Testing complete.  Now need to integrate the system.
*/

package paraprof;

import java.io.*;
import java.lang.*;
import java.util.*;
import org.jatha.Jatha;
import org.jatha.dynatype.*;
import org.jatha.compile.*;
import org.jatha.machine.*;

public class ParaProfLisp{
    ParaProfList(){
	this.lisp = new Jatha(false, false);
	this.lisp.init();
	this.lisp.start();
	//lisp.COMPILER.Register(new FirstPrimitive(this.lisp));
    }

    public void register(LispPrimative lispPrimative){
	lisp.COMPILER.Register(lispPrimative)}

    public String eval(String s){
	String value = null;
	try{
	    LispValue input = this.lisp.parse(s);
	    LispValue result = this.lisp.eval(input);
	    value = result.toString();
	}
	catch(Exception e){
	    System.err.println("LISP Exception: " + e);
	}
	return value;
    }


    //####################################
    //Instance Data
    //####################################
    Jatha lisp = null;
    //####################################
    //Instance Data
    //####################################
}


/*
public String nativeEval(){
	return null;
    }

    public static void main(String []args){
	SimpleExpressionTest simpleExpressionTest = new SimpleExpressionTest();
	String expression = "(* 5 10)";
	String expression2 = "(FIRST \"0:0:0\" \"0:0:1\" \"PAPI_FP_INS\" \"CPU_TIME\")";
	String value = simpleExpressionTest.eval(expression2);
	System.out.println(value);
	System.exit(0);
    }

    //####################################
    //Instance Data
    //####################################
    Jatha lisp = null;
    //####################################
    //Instance Data
    //####################################
    */

  
/*
class FirstPrimitive extends LispPrimitive{
    public FirstPrimitive(Jatha jatha){
	super(jatha, "FIRST", 4);
	this.jatha = jatha;
    }

    public void Execute(SECDMachine machine){
	LispValue arg4 = machine.S.pop();
	LispValue arg3 = machine.S.pop();
	LispValue arg2 = machine.S.pop();
	LispValue arg1 = machine.S.pop();

	machine.S.push(result(arg1, arg2, arg3, arg4));
	machine.C.pop();
    }

    LispValue result(LispValue arg1, LispValue arg2, LispValue arg3, LispValue arg4){
	System.out.println("Applying FIRST. Args: " + arg1 + "," + arg2 + "," + arg3 + "," + arg4);
	Vector v = new Vector();
	for(int i=0;i<1000000;i++){
	    if((i%1000)==0)
		System.out.println(i);
	    v.add(new Integer(7));
	}
	System.out.println("Done!");
	//int v1 = Integer.parseInt(arg1.toString());
	//int v2 = Integer.parseInt(arg2.toString());
	//return jatha.makeInteger(v1*v2);
	return jatha.makeInteger(1);
    }

    Jatha jatha = null;

}
*/
