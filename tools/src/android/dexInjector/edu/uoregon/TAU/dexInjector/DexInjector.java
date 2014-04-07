package edu.uoregon.TAU.dexInjector;

import java.io.*;
import java.util.*;
import org.ow2.asmdex.*;

public class DexInjector implements Opcodes {
    static HashMap<String,ArrayList<ArrayList<String>>> methodMap;
    static String packageName;

    public static void main(String[] args) {
	if (args.length != 3) {
	    System.out.println("Usage: DexInjector <in.dex> <out.dex> <PackageName>");
	    return;
	}

	methodMap = new HashMap<String,ArrayList<ArrayList<String>>>();
	packageName = args[2];

	try {
	    ApplicationReader  ar = new ApplicationReader(ASM4, args[0]);
	    ApplicationWriter  aw = new ApplicationWriter();
	    ApplicationVisitor av = new ApplicationAdapter(ASM4, aw);

	    ar.accept(av, 0);

	    byte[] newdex = aw.toByteArray();
	    File f = new File(args[1]);
	    OutputStream s = new FileOutputStream(f);
	    s.write(newdex);

	    FileOutputStream fos = new FileOutputStream("map.ser");
	    ObjectOutputStream oos = new ObjectOutputStream(fos);
	    oos.writeObject(methodMap);
	    oos.close();
	} catch(Exception e) {
	    e.printStackTrace();
	}
    }
}
