/**This class starts a new Matlabprocess via Java Runtime class. 
 * We can send commands to this process and receive results back. 
 * 
 */

package edu.uoregon.tau.viewer.iowithMatlab;

import java.lang.Runtime;
import java.io.*;
import javax.swing.*;

public class Engine{

    private Process proc;
    private BufferedWriter out; // output stream
    private BufferedReader in; // input stream
    private BufferedReader err; // error stream
    private boolean isOpen = false;
    
    public Engine(){}

	// open a Matlab process with a command string.
    public void open(String cmd) {
	try {
	    synchronized(this){
		proc = Runtime.getRuntime().exec(cmd);
		out = new BufferedWriter(new OutputStreamWriter(proc.getOutputStream()));
		in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
		err = new BufferedReader(new InputStreamReader(proc.getErrorStream()));
		isOpen = true;
	    }
	}
	catch(IOException e){
		JOptionPane.showMessageDialog(null, "Matlab could not be opened! Please check your Matlab setting.", "Warning!",
										  JOptionPane.ERROR_MESSAGE);		
	}
    }

	// judge whether the matlab process is going on.
    public boolean isOpen() {return isOpen;}

    public void evaluateStr(String str) throws IOException{
	send(str);
    }

	// send command to Matlab
    private void send(String str) throws IOException{
	str += "\n";
	synchronized(this){
	    out.write(str,0,str.length());
	    out.flush();
	}
    }

	// close the process.
    public void close() throws IOException{
	send("exit");
	isOpen = false;
	proc.destroy();
    }
    
    // test...
    public static void main(String[] args) {
	Engine engine = new Engine();
	
	// test
	try{
	    engine.open("matlab -nosplash -nojvm");
	    String str = "x=[2 4 8 16 32 64 128 ];";
	    str += "y=[2.568303124E9 3.252043214E9 3.924968426E9 3.92021509E9 4.91669621E9 6.25140437E9 1.3097782162E10 ];";
	    str+= "plot(x,y,'-s','MarkerSize',4,'MarkerFaceColor','auto');";
	    str += "title('Overhead');";
	    str += "xlabel('process number');";
	    str += "ylabel('overhead');";
	    str += "set(gca, 'xtick',x);";
	    str += "v = axis; ymax = v(4); axis([2 128 0 ymax]);";
	    
	    engine.evaluateStr(str);
	    Thread.sleep(10000);
	    engine.close();
	}
	catch (Exception e){
	}
    }
}
