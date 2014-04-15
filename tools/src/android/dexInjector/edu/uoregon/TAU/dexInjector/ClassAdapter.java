package edu.uoregon.TAU.dexInjector;

import org.ow2.asmdex.*;

public class ClassAdapter extends ClassVisitor implements Opcodes {
    public ClassAdapter(int api, ClassVisitor cv) {
	super(api, cv);
    }

    public MethodVisitor visitMethod(int access, String name, String desc,
				     String[] signature, String[] exceptions) {
	Filter.methodName      = name;
	Filter.methodAccess    = access;
	Filter.methodDesc      = desc;
	Filter.methodSignature = signature;

	MethodVisitor mv = cv.visitMethod(access, name, desc, signature, exceptions);

	if ((access & ACC_ABSTRACT) != 0 ||
	    (access & ACC_NATIVE)   != 0 ||
	    !Filter.accept()) {
	    return mv;
	} else {
	    MethodAdapter ma = new MethodAdapter(api, mv);
	    return ma;
	}
    }
}
