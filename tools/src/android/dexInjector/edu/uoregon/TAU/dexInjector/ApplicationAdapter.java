package edu.uoregon.TAU.dexInjector;

import org.ow2.asmdex.*;
import org.ow2.asmdex.structureCommon.Label;

public class ApplicationAdapter extends ApplicationVisitor implements Opcodes {
    public ApplicationAdapter(int api, ApplicationVisitor av) {
	super(api, av);
    }

    public void visit() {
	super.visit();
	ProfileDump.dumpProfile((ApplicationWriter)av);
	ProfilerDump.dumpProfiler((ApplicationWriter)av);
    }

    public ClassVisitor visitClass(int access, String name, String[] signature,
				   String superName, String[] interfaces) {
	Filter.className      = name;
	Filter.classAccess    = access;
	Filter.classSignature = signature;
	Filter.classSuperName = superName;

	ClassVisitor cv = av.visitClass(access, name, signature, superName, interfaces);
	ClassAdapter ca = new ClassAdapter(api, cv);

	return ca;
    }
}
