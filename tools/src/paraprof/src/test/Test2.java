import java.io.*;
import java.util.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.tree.*;

public class Test extends Thread{

    Test(){
    }

    public Vector mappings = new Vector();
    public Vector userevents = new Vector();

    public static void main(String args[]){

	/*
	System.out.println("------------------------------------------------");
	String s1 = "10,9,123 35 \"main() int (int, char **)\" incl 5008292 100.00 GROUP=\"TAU_DEFAULT\"";
	String s2 = "    0        5,008        5,008           134           24    5008292 main() int (int, char **)";

	String m1 = "m 2 \"main() int (int, char **)\" incl 2504146 50.01 GROUP=\"TAU_DEFAULT\"";
	String m2 = " 50.0        2,504        2,504         0.5           0    5008292 main() int (int, char **)";

	String t1 = "t 323 \"second() int ()\" excl 39 0.00 GROUP=\"TAU_DEFAULT\"";
	String t2 = "  0.0        0.039        0.092           12           1234         92 second() int ()";

	String u = "userevent 0,0,0 34 \"Memory allocated by arrays\" 34.000000000000000 2048.000000000000 2048.000000000000 2048.000000000000 123.000000000000000";

	String N = "NumSamples   MaxValue   MinValue  MeanValue  Std. Dev.  Event Name";

	String noue = "223 userevents";
	*/
	Test t = new Test();
	


	
	/*
	System.out.println("s1: " + s1);
	System.out.println("s2: " + s2);
	System.out.println("m1: " + m1);
	System.out.println("m2: " + m2);
	System.out.println("t1: " + t1);
	System.out.println("t2: " + t2);
	System.out.println("u: " + u);

	TestMapping tm = t.getData(s1, s2);

	System.out.println("ue: " + tm.ue);
	System.out.println("excl: " + tm.excl);
	System.out.println("m: " + tm.mean);
	System.out.println("t: " + tm.total);
	System.out.println("node: " + tm.node);
	System.out.println("context: " + tm.context);
	System.out.println("thread: " + tm.thread);
	System.out.println("mappingID: " + tm.mappingID);
	System.out.println("mappingName: " + tm.mappingName);
	System.out.println("groups: " + tm.groups);
	System.out.println("value: " + tm.val2);
	System.out.println("percent: " + tm.val3);
	System.out.println("noc: " + tm.noc);
	System.out.println("nos: " + tm.nos);
	System.out.println("Num of Samples: " + tm.noc);
	System.out.println("max: " + tm.val1);
	System.out.println("min: " + tm.val2);
	System.out.println("umean: " + tm.val3);
	System.out.println("std: " + tm.val4);
	
	System.out.println("------------------------------------------------");

	tm = t.getData(m1, m2);
	System.out.println("ue: " + tm.ue);
	System.out.println("excl: " + tm.excl);
	System.out.println("m: " + tm.mean);
	System.out.println("t: " + tm.total);
	System.out.println("node: " + tm.node);
	System.out.println("context: " + tm.context);
	System.out.println("thread: " + tm.thread);
	System.out.println("mappingID: " + tm.mappingID);
	System.out.println("mappingName: " + tm.mappingName);
	System.out.println("groups: " + tm.groups);
	System.out.println("value: " + tm.val2);
	System.out.println("percent: " + tm.val3);
	System.out.println("noc: " + tm.noc);
	System.out.println("nos: " + tm.nos);
	System.out.println("Num of Samples: " + tm.noc);
	System.out.println("max: " + tm.val1);
	System.out.println("min: " + tm.val2);
	System.out.println("umean: " + tm.val3);
	System.out.println("std: " + tm.val4);

	System.out.println("------------------------------------------------");

	tm = t.getData(t1, t2);
	System.out.println("ue: " + tm.ue);
	System.out.println("excl: " + tm.excl);
	System.out.println("m: " + tm.mean);
	System.out.println("t: " + tm.total);
	System.out.println("node: " + tm.node);
	System.out.println("context: " + tm.context);
	System.out.println("thread: " + tm.thread);
	System.out.println("mappingID: " + tm.mappingID);
	System.out.println("mappingName: " + tm.mappingName);
	System.out.println("groups: " + tm.groups);
	System.out.println("value: " + tm.val2);
	System.out.println("percent: " + tm.val3);
	System.out.println("noc: " + tm.noc);
	System.out.println("nos: " + tm.nos);
	System.out.println("Num of Samples: " + tm.noc);
	System.out.println("max: " + tm.val1);
	System.out.println("min: " + tm.val2);
	System.out.println("umean: " + tm.val3);
	System.out.println("std: " + tm.val4);

	System.out.println("------------------------------------------------");


	tm = t.getData(u, null);
	System.out.println("ue: " + tm.ue);
	System.out.println("excl: " + tm.excl);
	System.out.println("m: " + tm.mean);
	System.out.println("t: " + tm.total);
	System.out.println("node: " + tm.node);
	System.out.println("context: " + tm.context);
	System.out.println("thread: " + tm.thread);
	System.out.println("mappingID: " + tm.mappingID);
	System.out.println("mappingName: " + tm.mappingName);
	System.out.println("groups: " + tm.groups);
	System.out.println("value: " + tm.val2);
	System.out.println("percent: " + tm.val3);
	System.out.println("noc: " + tm.noc);
	System.out.println("nos: " + tm.nos);
	System.out.println("Num of Samples: " + tm.noc);
	System.out.println("max: " + tm.val1);
	System.out.println("min: " + tm.val2);
	System.out.println("umean: " + tm.val3);
	System.out.println("std: " + tm.val4);

	System.out.println("------------------------------------------------");

	tm = t.getData(N,null);
	if(tm==null)
	    System.out.println("Ignored: " + N);

	tm = t.getData(noue,null);
	if(tm==null)
	    System.out.println("Ignored: " + noue);
	*/
	System.exit(0);
    }

    public void run(){
	try{
	    File f = new File("pprof.dat");
	    FileInputStream fileIn = new FileInputStream(f);
	    ProgressMonitorInputStream progressIn = new ProgressMonitorInputStream(null, "Processing ...", fileIn);
	    InputStreamReader inReader = new InputStreamReader(progressIn);
	    BufferedReader br = new BufferedReader(inReader);

	    TestMapping tm = null;
	    String s = null;

	    s = br.readLine();
	    s = br.readLine();
	    s = br.readLine();

	    while((s = br.readLine()) != null){
		if((!n(s)) && (!noue(s))){
		    tm = getData(s,br.readLine());
		    if(tm.ue)
			userevents.add(tm);
		    else
			mappings.add(tm);
		}
	    }
	}
	catch(Exception e){
	    e.printStackTrace();
	}
    }

    public boolean n(String s){
	if((s.charAt(0))=='N')
	    return true;
	else
	    return false;
    }

    public boolean noue(String s){
	int stringPosition = 0;
	char tmpChar = s.charAt(stringPosition);
	while(tmpChar!='\u0020'){
	    stringPosition++;
	    tmpChar = s.charAt(stringPosition);
	}
	stringPosition++;
	tmpChar = s.charAt(stringPosition);
	if(tmpChar=='u')
	    return true;
	else
	    return false;
    }

    public TestMapping getData(String s1, String s2){
	//I am assuming an quick implimentation of charAt and append for this function.
	TestMapping tm = new TestMapping();
	char lastCharCheck = '\u0020';
	try{
	    char tmpChar = '\u0020';
	    StringBuffer tmpBuffer = new StringBuffer();
	    int stringPosition = 0;

	    //Test for mean or total.
	    tmpChar = s1.charAt(stringPosition);
	    boolean mt = false;
	    boolean u = false;
	    if(tmpChar=='u'){
		tm.ue = true;
		u = true;
	    }
	    else if(tmpChar=='m'){
		tm.mean = true;
		mt = true;
	    }
	    else if(tmpChar=='t'){
		tm.total = true;
		mt = true;
	    }
	    
	    //Process s1.
	    int start = 0;
	    int end = 9;
	    if(mt){
		start = 3;
		stringPosition = 2;
	    }
	    else
		lastCharCheck = ',';

	    if(u)
		stringPosition = 10;

	    for(int i=start;i<9;i++){
		if(i==2)
		    lastCharCheck = '\u0020';
		else if(i==4){
		    lastCharCheck = '"';
		    stringPosition++;
		}
		else if(i==5){
		    lastCharCheck = '\u0020';
		    stringPosition++;
		}
		else if(i==8 && !u){
		    lastCharCheck = '"';
		    stringPosition = stringPosition+7;
		}
		tmpChar = s1.charAt(stringPosition);
		while(tmpChar!=lastCharCheck){
		    tmpBuffer.append(tmpChar);
		    stringPosition++;
		    tmpChar = s1.charAt(stringPosition);
		}
		
		switch(i){
		case 0:
		    tm.node = Integer.parseInt(tmpBuffer.toString());
		    break;
		case 1:
		    tm.context = Integer.parseInt(tmpBuffer.toString());
		    break;
		case 2:
		    tm.thread = Integer.parseInt(tmpBuffer.toString());
		    break;
		case 3:
		    tm.mappingID = Integer.parseInt(tmpBuffer.toString());
		    break;
		case 4:
		    tm.mappingName = tmpBuffer.toString();
		    break;
		case 5:
		    if(u)
			tm.noc = Double.parseDouble(tmpBuffer.toString());
		    else
			tm.excl = (tmpBuffer.toString()).equals("excl");
		    break;
		case 6:
		    tm.val1 = Double.parseDouble(tmpBuffer.toString());
		    break;
		case 7:
		    tm.val2 = Double.parseDouble(tmpBuffer.toString());
		    break;
		case 8:
		    if(u)
			tm.val3 = Double.parseDouble(tmpBuffer.toString());
		    else
			tm.groups = tmpBuffer.toString();
		    break;
		default:
		    throw new UnexpectedStateException(String.valueOf(i));
		}
		//Reset things.
		tmpBuffer.delete(0,tmpBuffer.length());
		tmpChar = '\u0020';
		stringPosition++;
	    }
	    //One more item to pick up if userevent string.
	    if(u){
		int length = s1.length();
		while(stringPosition < length){
		    tmpChar = s1.charAt(stringPosition);
		    tmpBuffer.append(tmpChar);
		    stringPosition++;
		}
		tm.val4 = Double.parseDouble(tmpBuffer.toString());
		//Reset things.
		tmpBuffer.delete(0,tmpBuffer.length());
		tmpChar = '\u0020';
	    }
	
	    //Process s2
	    if(u){
		return tm;
	    }
	    
	    stringPosition = 0;
	    for(int i=0;i<10;i++){
		tmpChar = s2.charAt(stringPosition);
		if(i%2==0){
		    while(tmpChar=='\u0020'){
			stringPosition++;
			tmpChar = s2.charAt(stringPosition);
		    }
		}
		else if(i==7 || i==9){
		    while(tmpChar!='\u0020'){
			tmpBuffer.append(tmpChar);
			stringPosition++;
			tmpChar = s2.charAt(stringPosition);
		    }
		}
		else{
		    while(tmpChar!='\u0020'){
			stringPosition++;
			tmpChar = s2.charAt(stringPosition);
		    }
		}
		
		if(i==7){
		    tm.noc = Double.parseDouble(tmpBuffer.toString());
		}
		else if(i==9){
		    tm.nos = Double.parseDouble(tmpBuffer.toString());
		}
		
		//Reset things.
		tmpBuffer.delete(0,tmpBuffer.length());
		tmpChar = '\u0020';
	    }
	}
	catch(Exception e){
	    System.out.println("An error occured!");
	    e.printStackTrace();
	}
	return tm;
    }
}

class TestMapping{
    
    public boolean ue = false;
    public boolean excl = true;
    public boolean mean = false;
    public boolean total = false;
    public int node = -1;
    public int context = -1;
    public int thread = -1;
    public int mappingID = -1;
    public String mappingName = "not-set";
    public String groups = "not-set";
    public double val1 = -1.0; //First value after the name.
    public double val2 = -1.0; //And so on.
    public double val3 = -1.0;
    public double val4 = -1.0;
    public double val5 = -1.0;
    public double value = -1.0;
    public double percentValue = -1.0;
    public double noc = -1.0;
    public double nos = -1.0;
    public double min = -1.0;
    public double max = -1.0;
    public double umean = -1.0;
    public double std = -1.0;
}

class UnexpectedStateException extends Exception{
    public UnexpectedStateException(){}
    public UnexpectedStateException(String err){
	super("UnexpectedStateException - message: " + err);
    }
}
