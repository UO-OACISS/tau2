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
    public int uENamesSet = 0;

    public static void main(String args[]){

	Test t = new Test();

	t.start();
	try{
	    t.join();
	}
	catch(Exception e){
	    System.out.println("The thread was interupted!");
	}

	System.out.println("Ok, thread is done ... exit the system");
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
	    String s1 = null;
	    String s2 = null;

	    s1 = br.readLine();
	    s1 = br.readLine();
	    s1 = br.readLine();

	    System.out.println("Processing file ...");

	    while((s1 = br.readLine()) != null){
		if((!n(s1)) && (!noue(s1))){
		    s2 = br.readLine();
		    tm = getData(s1,s2);
		    if(tm.ue)
			userevents.add(tm);
		    else
			mappings.add(tm);
		}
		else
		    uENamesSet++; 
	    }

	    System.out.println("Done!");
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
	    boolean m = false;
	    boolean t = false;
	    boolean u = false;
	    if(tmpChar=='u'){
		tm.ue = true;
		u = true;
	    }
	    else if(tmpChar=='m'){
		tm.mean = true;
		m = true;
	    }
	    else if(tmpChar=='t'){
		tm.total = true;
		t = true;
	    }
	    
	    //Process s1.
	    int start = 0;
	    int end = 9;
	    if(m || t){
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
		    if(t)
			tm.mappingName = tmpBuffer.toString();
		    else if(u)
			if(uENamesSet<2)
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
