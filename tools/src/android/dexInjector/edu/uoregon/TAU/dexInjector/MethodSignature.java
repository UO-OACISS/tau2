package edu.uoregon.TAU.dexInjector;

public class MethodSignature {
    public static String popType(String desc) {
	return desc.substring(nextTypePosition(desc, 0));
    }

    public static int nextTypePosition(String desc, int pos) {
	while(desc.charAt(pos) == '[') {
	    pos++;
	}

	if (desc.charAt(pos) == 'L') {
	    pos = desc.indexOf(';', pos);
	}

	pos++;

	return pos;
    }
}
