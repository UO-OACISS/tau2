package edu.uoregon.TAU.dexInjector;

import org.ow2.asmdex.*;

public class ClassAdapter extends ClassVisitor implements Opcodes {
    public ClassAdapter(int api, ClassVisitor cv) {
	super(api, cv);
    }

    public MethodVisitor visitMethod(int access, String name, String desc, String[] signature,
				     String[] exceptions) {
	Filter.methodName = name;
	Filter.methodDesc = desc;
	Filter.methodSignature = signature;

	MethodVisitor mv = cv.visitMethod(access, name, desc, signature, exceptions);

	/*
	if (name.equals("main")) {
	    return ma;
	} else {
	    return mv;
	}
	*/

	if ((access & ACC_ABSTRACT) != 0 ||
	    (access & ACC_NATIVE)   != 0 ||
	    !Filter.accept()) {
	    return mv;
	}

	MethodAdapter ma = new MethodAdapter(api, mv);

	return ma;
    }
}
