package edu.uoregon.TAU.dexInjector;

import java.util.*;

public class MethodDescriptor {
    private String className;
    private String methodName;
    private String rawMethodDesc;

    private List<TypeDescriptor> typeList;

    private int index = 0;

    public static String parseClassName(String className) {
	if (className.startsWith("L") && className.endsWith(";")) {
	    return className.substring(1, className.length()-1).replace('/','.');
	} else {
	    /* malformed class name, return unchanged for safety */
	    return className;
	}
    }
    private TypeDescriptor parseTypeDescriptor(String typeDesc) {
	int    type;
	String name;

	switch(typeDesc.charAt(0)) {
	case 'V':
	    index++;
	    type = TypeDescriptor.VOID;
	    name = "void";
	    break;
	case 'Z':
	    index++;
	    type = TypeDescriptor.BOOL;
	    name = "boolean";
	    break;
	case 'B':
	    index++;
	    type = TypeDescriptor.BYTE;
	    name = "byte";
	    break;
	case 'S':
	    index++;
	    type = TypeDescriptor.SHORT;
	    name = "short";
	    break;
	case 'C':
	    index++;
	    type = TypeDescriptor.CHAR;
	    name = "char";
	    break;
	case 'I':
	    index++;
	    type = TypeDescriptor.INT;
	    name = "int";
	    break;
	case 'J':
	    index++;
	    type = TypeDescriptor.LONG;
	    name = "long";
	    break;
	case 'F':
	    index++;
	    type = TypeDescriptor.FLOAT;
	    name = "float";
	    break;
	case 'D':
	    index++;
	    type = TypeDescriptor.DOUBLE;
	    name = "double";
	    break;
	case 'L':
	    type = TypeDescriptor.CLASS;
	    name = parseClassName(typeDesc.substring(0, typeDesc.indexOf(';')+1));
	    index += typeDesc.indexOf(';') + 1;
	    break;
	case '[':
	    index++;
	    type = TypeDescriptor.ARRAY;
	    name = parseTypeDescriptor(typeDesc.substring(1)) + "[]";
	    break;
	default:
	    index++;
	    type = TypeDescriptor.UNKNOWN;
	    name = "<unknownType>";
	}

	return new TypeDescriptor(type, name);
    }

    public MethodDescriptor(String className, String methodName, String methodDesc) {
	this.className     = parseClassName(className);
	this.methodName    = methodName;
	this.rawMethodDesc = methodDesc;

	typeList = new ArrayList<TypeDescriptor>();

	while(index < rawMethodDesc.length()) {
	    TypeDescriptor desc = parseTypeDescriptor(rawMethodDesc.substring(index));
	    typeList.add(desc);
	}
    }

    public String toString() {
	StringBuilder desc = new StringBuilder();

	/* return type */
	desc.append(typeList.get(0));
	/* className:methodName( */
	desc.append(" " + className + ":" + methodName + "(");
	/* argument list */
	if (typeList.size() >= 2) {
	    desc.append(typeList.get(1));
	}
	for (int i=2; i<typeList.size(); i++) {
	    desc.append(", " + typeList.get(i));
	}
	/* ) */
	desc.append(")");

	return desc.toString();
    }

    public String getClassName() {
	return className;
    }

    public String getMethodName() {
	return methodName;
    }

    public List<TypeDescriptor> getTypeList() {
	return typeList;
    }
    

    /*
    public TypeDescriptor returnType() {
	return typeList.get(0);
    }

    public TypeDescriptor argsList() {
	String args = "";

	if (typeList.size() >= 2) {
	    args += typeList.get(1);
	}

	for (int i = 2; i < typeList.size(); i++) {
	    args += ", " + typeList.get(i);
	}

	return args;
    }

    public void print() {
	System.out.println(returnType() + " (" + argsList() + ")");
    }
    */
}
