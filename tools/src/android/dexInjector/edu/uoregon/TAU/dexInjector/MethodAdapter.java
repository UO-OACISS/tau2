package edu.uoregon.TAU.dexInjector;

import java.util.*;
import org.ow2.asmdex.*;
import org.ow2.asmdex.structureCommon.Label;

public class MethodAdapter extends MethodVisitor implements Opcodes {
    private Label handler;

    private Label last_start;
    private Label last_end;

    private int lineBegin = 0;
    private int lineEnd   = 0;

    private boolean insertProfilerStart;

    private List<DelegatedReturn> drList;

    public MethodAdapter(int api, MethodVisitor mv) {
	super(api, mv);
    }

    public void visitLineNumber(int line, Label start) {
	if ((lineBegin == 0) ||
	    (lineBegin > line)) {
	    lineBegin = line;
	}

	if (lineEnd < line) {
	    lineEnd   = line;
	}

	mv.visitLineNumber(line, start);

	// injact Profiler.start() after the first visitLineNumber(), so it
	// can have a line# at the runtime, i.e. Thread.currentThread().getStackTrace()
	if (insertProfilerStart) {
	    mv.visitMethodInsn(INSN_INVOKE_STATIC, "Ledu/uoregon/TAU/Profiler;", "start", "V", new int[] { });
	    insertProfilerStart = false;
	}
    }

    public void visitMaxs(int maxStack, int maxLocals) {
	// we need at least 1 register for global exception handler
	int regCount = maxStack==0 ? 1 : maxStack;

	mv.visitMaxs(regCount, maxLocals);
    }

    private void debugMsg() {
	System.out.println("className:        " + Filter.className);
	System.out.println("classSuperName:   " + Filter.classSuperName);
	if (Filter.classSignature != null ) {
	    System.out.println("classSignatures: ");
	    for (String s: Filter.classSignature) {
		System.out.println(s);
	    }
	} else {
	    System.out.println("classSignatures : NULL");
	}
	System.out.println("methodName:       " + Filter.methodName);
	if (Filter.methodSignature != null) {
	    System.out.println("methodSignatures:");
	    for (String s: Filter.methodSignature) {
		System.out.println(s);
	    }
	} else {
	    System.out.println("methodSignatures: NULL");
	}
	System.out.println("methodDesc:       " + Filter.methodDesc);

	System.out.println("---------------");
    }

    public void visitCode() {
	insertProfilerStart = true;

	handler    = new Label();
	last_start = new Label();
	last_end   = last_start;

	drList = new ArrayList<DelegatedReturn>();

	mv.visitCode();

	mv.visitLabel(last_start);

	//debugMsg();
    }

    public void visitTryCatchBlock(Label start,
				   Label end,
				   Label handler,
				   java.lang.String type) {
	if (!start.equals(last_start) &&  // not the same try-catch block as last one
	    !start.equals(last_end)) {    // not a empty try-catch block
	    mv.visitTryCatchBlock(last_end, start, this.handler, null);
	}

	mv.visitTryCatchBlock(start, end, handler, type);

	last_start = start;
	last_end   = end;
    }

    public void visitEnd() {
	MethodDescriptor desc = new MethodDescriptor(Filter.methodDesc);
	String className = MethodDescriptor.parseClassName(Filter.className);

	String signature = desc.returnType()+" "+className+":"+Filter.methodName+"("+desc.argsList()+")";
	String key = className+":"+Filter.methodName;

	// add global exception handler block
	visitLabel(handler);
	if (!last_end.equals(handler)) {
	    mv.visitTryCatchBlock(last_end, handler, handler, null);
	}
	mv.visitIntInsn(INSN_MOVE_EXCEPTION, 0);
	mv.visitMethodInsn(INSN_INVOKE_STATIC, "Ledu/uoregon/TAU/Profiler;", "stop", "V", new int[] { });
	mv.visitIntInsn(INSN_THROW, 0);

	// add delegated returns
	for (DelegatedReturn dr: drList) {
	    dr.visit();
	}

	// update method-line# map
	ArrayList<String> sigList = new ArrayList<String>();
	sigList.add(Integer.toString(lineBegin));
	sigList.add(Integer.toString(lineEnd));
	sigList.add(signature);
	if (DexInjector.methodMap.containsKey(key)) {
	    ArrayList<ArrayList<String>> val = (ArrayList<ArrayList<String>>)DexInjector.methodMap.get(key);
	    val.add(sigList);
	} else {
	    ArrayList<ArrayList<String>> val = new ArrayList<ArrayList<String>>();
	    val.add(sigList);
	    DexInjector.methodMap.put(key, val);
	}

	mv.visitEnd();
    }

    class DelegatedReturn {
	private int opcode;
	private int register;
	private Label label;

	public DelegatedReturn(int opcode, int register, Label label) {
	    this.opcode   = opcode;
	    this.register = register;
	    this.label    = label;
	}

	public void visit() {
	    mv.visitLabel(label);
	    mv.visitMethodInsn(INSN_INVOKE_STATIC, "Ledu/uoregon/TAU/Profiler;", "stop", "V", new int[] { });

	    switch (opcode) {
	    case INSN_RETURN_VOID:
		mv.visitInsn(opcode);
		break;
	    case INSN_RETURN:
	    case INSN_RETURN_WIDE:
	    case INSN_RETURN_OBJECT:
		mv.visitIntInsn(opcode, register);
		break;
	    /* no default action */
	    }
	}
    }

    public void visitInsn(int opcode) {
	boolean isInTryBlock;

	switch (opcode) {
	case INSN_RETURN_VOID:
	    Label label = new Label();
	    DelegatedReturn dr = this.new DelegatedReturn(opcode, 0, label);
	    drList.add(dr);
	    mv.visitJumpInsn(INSN_GOTO_32, label, 0, 0);
	    break;
	default:
	    mv.visitInsn(opcode);
	    break;
	}
    }

    public void visitIntInsn(int opcode, int register) {
	boolean isInTryBlock;

	switch (opcode) {
	case INSN_RETURN:
	case INSN_RETURN_WIDE:
	case INSN_RETURN_OBJECT:
	    Label label = new Label();
	    DelegatedReturn dr = this.new DelegatedReturn(opcode, register, label);
	    drList.add(dr);
	    mv.visitJumpInsn(INSN_GOTO_32, label, 0, 0);
	    break;
	default:
	    mv.visitIntInsn(opcode, register);
	    break;
	}
    }
}
