package edu.uoregon.TAU.dexInjector;

import java.io.*;
import java.util.*;
import org.ow2.asmdex.*;

public class DexInjector implements Opcodes {
    public static void main(String[] args) {
	if (args.length != 2) {
	    System.out.println("Usage: DexInjector <in.dex> <out.dex>");
	    return;
	}

	try {
	    ApplicationReader  ar = new ApplicationReader(ASM4, args[0]);
	    ApplicationWriter  aw = new ApplicationWriter();
	    ApplicationVisitor av = new ApplicationAdapter(ASM4, aw);

	    ar.accept(av, 0);

	    byte[] newdex = aw.toByteArray();
	    File f = new File(args[1]);
	    OutputStream s = new FileOutputStream(f);
	    s.write(newdex);
	} catch(Exception e) {
	    e.printStackTrace();
	}
    }
}
