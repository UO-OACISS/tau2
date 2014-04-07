package edu.uoregon.TAU.dexInjector;

import java.util.*;

public class Filter {
    public static String className;   // Full qualified class name
    public static String[] classSignature;  // no use
    public static String classSuperName; // Full qualified super class name

    public static String methodName;  // method name
    public static String[] methodSignature;  // no use
    public static String methodDesc;  // method descriptor

    public static boolean accept() {
	/* ignore constructors */
	if (methodName.equals("<init>") || methodName.equals("<clinit>")) {
	    return false;
	}

	/*
	if (className.equals("Lcom/example/stepstone/MainActivity;") && methodName.equals("onCreate")) {
	    return true;
	}

	if (className.matches("Landroid/support/v4/(accessibilityservice|app|database|graphics|hardware|media|net|os|print|text|util|view|widget).*")) {
	    return true;
	}
	*/

	if (className.matches("Landroid/support/v4/content/FileProvider;")) {
	    /* g */
	    if (methodName.matches("getPath.*")) {
		//return false;
	    }
	}

	return true;
    }
}
