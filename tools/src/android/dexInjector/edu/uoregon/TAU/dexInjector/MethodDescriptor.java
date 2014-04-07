package edu.uoregon.TAU.dexInjector;

import java.util.*;

public class MethodDescriptor {
    private String rawDesc;
    private List<String> typeList;

    private int index = 0;

    public static String parseClassName(String className) {
	if (className.startsWith("L") && className.endsWith(";")) {
	    return className.substring(1, className.length()-1).replace('/','.');
	} else {
	    /* malformed class name, return unchanged for safety */
	    return className;
	}
    }
    private String parseTypeDescriptor(String typeDesc) {
	String name;

	switch(typeDesc.charAt(0)) {
	case 'V':
	    index++;
	    name = "void";
	    break;
	case 'Z':
	    index++;
	    name = "boolean";
	    break;
	case 'B':
	    index++;
	    name = "byte";
	    break;
	case 'S':
	    index++;
	    name = "short";
	    break;
	case 'C':
	    index++;
	    name = "char";
	    break;
	case 'I':
	    index++;
	    name = "int";
	    break;
	case 'J':
	    index++;
	    name = "long";
	    break;
	case 'F':
	    index++;
	    name = "float";
	    break;
	case 'D':
	    index++;
	    name = "double";
	    break;
	case 'L':
	    name = parseClassName(typeDesc.substring(0, typeDesc.indexOf(';')+1));
	    index += typeDesc.indexOf(';') + 1;
	    break;
	case '[':
	    index++;
	    name = parseTypeDescriptor(typeDesc.substring(1))+"[]";
	    break;
	default:
	    index++;
	    name = "<unknownType>";
	}

	return name;
    }

    public MethodDescriptor(String desc) {
	typeList = new ArrayList<String>();
	rawDesc = desc;

	while(index < desc.length()) {
	    String name = parseTypeDescriptor(desc.substring(index));
	    typeList.add(name);
	}
    }

    public String returnType() {
	return typeList.get(0);
    }

    public String argsList() {
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
}
