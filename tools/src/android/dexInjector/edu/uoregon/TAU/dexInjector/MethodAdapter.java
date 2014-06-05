package edu.uoregon.TAU.dexInjector;

import java.util.*;
import org.ow2.asmdex.*;
import org.ow2.asmdex.structureCommon.Label;

public class MethodAdapter extends MethodVisitor implements Opcodes {
    private final static int EXTRA_REGS = 3;

    private int registerCount;
    private MethodDescriptor methodDesc;

    private Label handler;

    private Label last_start;
    private Label last_end;

    private List<DelegatedReturn> drList;

    public MethodAdapter(int api, MethodVisitor mv) {
	super(api, mv);
	methodDesc = new MethodDescriptor(Filter.className, Filter.methodName, Filter.methodDesc);
    }

    private void shiftRegisters() {
	List<TypeDescriptor> args = methodDesc.getTypeList();

	/*
	 * Note that instance method (i.e. non-STATIC) always explicitly
	 * pass "this" as the first argument
	 */

	/* figure out how many registers are used for argument passing */
	int argRegs = 0;
	if ((Filter.methodAccess & ACC_STATIC) == 0) {
	    argRegs += 1;  // this
	}
	for (int i=1; i<args.size(); i++) {
	    argRegs += args.get(i).size;
	}

	/* now shift the registers */
	int reg = registerCount - argRegs; // first register used for argument passing
	if ((Filter.methodAccess & ACC_STATIC) == 0) {
	    mv.visitVarInsn(INSN_MOVE_OBJECT_16, reg-EXTRA_REGS, reg);
	    reg += 1;
	}
	for (int i=1; i<args.size(); i++) {
	    TypeDescriptor arg = args.get(i);

	    /* Note that VOID never appears in argument list */
	    switch (arg.type) {
	    case TypeDescriptor.BOOL:
	    case TypeDescriptor.BYTE:
	    case TypeDescriptor.SHORT:
	    case TypeDescriptor.CHAR:
	    case TypeDescriptor.INT:
	    case TypeDescriptor.FLOAT:
		mv.visitVarInsn(INSN_MOVE_16, reg-EXTRA_REGS, reg);
		break;

	    case TypeDescriptor.CLASS:
	    case TypeDescriptor.ARRAY:
		mv.visitVarInsn(INSN_MOVE_OBJECT_16, reg-EXTRA_REGS, reg);
		break;

	    case TypeDescriptor.LONG:
	    case TypeDescriptor.DOUBLE:
		mv.visitVarInsn(INSN_MOVE_WIDE_16, reg-EXTRA_REGS, reg);
		break;

	    default:
		/* this should never happen */
		break;
	    }

	    reg += arg.size;
	}
    }

    public void visitMaxs(int maxStack, int maxLocals) {
	registerCount = maxStack + EXTRA_REGS;

	mv.visitMaxs(registerCount, maxLocals);

	mv.visitStringInsn(INSN_CONST_STRING_JUMBO, 0, methodDesc.toString());
	mv.visitMethodInsn(INSN_INVOKE_STATIC, "Ledu/uoregon/TAU/Profiler;", "start", "VLjava/lang/String;", new int[] { 0 });

	shiftRegisters();
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
	MethodDescriptor desc = new MethodDescriptor(Filter.className, Filter.methodName, Filter.methodDesc);

	String key = desc.getClassName() + ":" + desc.getMethodName();

	// add global exception handler block
	visitLabel(handler);
	if (!last_end.equals(handler)) {
	    mv.visitTryCatchBlock(last_end, handler, handler, null);
	}
	mv.visitIntInsn(INSN_MOVE_EXCEPTION, 0);
	mv.visitStringInsn(INSN_CONST_STRING_JUMBO, 1, methodDesc.toString());
	mv.visitMethodInsn(INSN_INVOKE_STATIC, "Ledu/uoregon/TAU/Profiler;", "stop", "VLjava/lang/String;", new int[] { 1 });
	mv.visitIntInsn(INSN_THROW, 0);

	// add delegated returns
	for (DelegatedReturn dr: drList) {
	    dr.visit();
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
	    int callsite = 0;

	    mv.visitLabel(label);

	    switch (opcode) {
	    case INSN_RETURN_VOID:
		mv.visitStringInsn(INSN_CONST_STRING_JUMBO, callsite, methodDesc.toString());
		mv.visitMethodInsn(INSN_INVOKE_STATIC, "Ledu/uoregon/TAU/Profiler;", "stop", "VLjava/lang/String;", new int[] { callsite });
		mv.visitInsn(opcode);
		break;
	    case INSN_RETURN:
	    case INSN_RETURN_WIDE:
	    case INSN_RETURN_OBJECT:
		if (register == 0) {
		    callsite = 2;
		}
		mv.visitStringInsn(INSN_CONST_STRING_JUMBO, callsite, methodDesc.toString());
		mv.visitMethodInsn(INSN_INVOKE_STATIC, "Ledu/uoregon/TAU/Profiler;", "stop", "VLjava/lang/String;", new int[] { callsite });
		mv.visitIntInsn(opcode, register);
		break;
	    /* no default action */
	    }
	}
    }

    public void visitInsn(int opcode) {
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

    public void visitStringInsn(int opcode, int destinationRegister, String string) {
	/*
	 * We pass the callsite as a string constant to Profiler.start()/stop().
	 * As a result, we are increasing strings in dex constant pool. Chances
	 * are that index of some strings in constant pool will be bigger than
	 * 0xffff, i.e. can not be hold in 2 bytes.
	 *
	 * INSN_CONST_STRING is using 2 bytes to hold the string index, thus
	 * it should be rewritten to INSN_CONST_STRING_JUMBO if the string index
	 * it is referring does not fit in 2 bytes.
	 *
	 * ASMDEX hides the string index from us so this fix should be done by
	 * ASMDEX. The fix is not trivial so instead we workaround this in our
	 * code by blindly rewriting all INSN_CONST_STRING to its JUMBO version.
	 */
	mv.visitStringInsn(INSN_CONST_STRING_JUMBO, destinationRegister, string);
    }

    public void visitLocalVariable(String name, String desc, String signature,
				   Label start, List<Label> ends, List<Label> restarts,
				   int index) {
	/*
	 * Name and desc of method parameters are not included in debug
	 * info as they can be extracted from method descriptor. As we
	 * have extended regester space of the method, we shall fix the
	 * register allocation for method parameters recorded in debug
	 * info.
	 *
	 * If you run "dexdump -d" on injected dex file without this fix,
	 * most likely you will see dexdump complains like this:
	 *
	 *   E/dalvik: Invalid debug info stream.....
	 */
	if (name == null) {
	    index += EXTRA_REGS;
	}

	mv.visitLocalVariable(name, desc, signature, start, ends, restarts, index);
    }
}
