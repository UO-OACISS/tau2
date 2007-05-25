package edu.uoregon.tau.common;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;

public class Wget {

    public static void wget(String URL, String file) throws IOException {
		wget(URL, file, false);
	}

    public static void wget(String URL, String file, boolean status) throws IOException
    {
    	 wget(URL, file, status, new Progress() {

    		 int size;
    		 
    		 public void reportFininshed(int bytes) {
    			 System.out.println("\r" + bytes / 1000 + "k bytes... done.");
			}

			public void reportProgress(int bytes) {
				System.out.print("\r" + bytes / 1000 + "k bytes... of " + size / 1000 + "k bytes" );
			}
			public void reportSize(int bytes) {
				size = bytes;
				
			}});
    }
    
    public static void wget(String URL, String file, boolean status, Progress prg) throws IOException {
        URLConnection url = new URL(URL).openConnection();
        DataInputStream in = new DataInputStream(url.getInputStream());
        OutputStream out = new FileOutputStream(file);
		int i = 0;
		
		prg.reportSize(url.getContentLength());
		
        try {
            while (true) {
                out.write(in.readUnsignedByte());
				i++;
				if (status && i % 100000 == 0)
					//System.out.print("\r" + i / 1000 + "k bytes...");
					prg.reportProgress(i);
            }
        } catch (EOFException e) {
            out.close();
			if (status)
				//System.out.println("\r" + i / 1000 + "k bytes... done.");
				prg.reportFininshed(i);
            return;
        }
    }

	public static void main(String[] args) {
		if (args.length == 2) {
        	try {
				Wget.wget(args[0], args[1], false);
			} catch (IOException e) {
				System.out.println("Failed getting " + args[0]);
			}
		} else if (args.length == 3) {
        	try {
				Wget.wget(args[0], args[1], args[2].equalsIgnoreCase("true"));
			} catch (IOException e) {
				System.out.println("Failed getting " + args[0]);
			}
		} else {
			System.out.println("Usage: Wget <url> <local filename>");
			System.out.println("Failed getting " + args[0]);
		}
	}
	  public static interface Progress
	  {
	    public void reportProgress(int bytes);
	    public void reportFininshed(int bytes);
	    public void reportSize(int bytes);
	  }
}
