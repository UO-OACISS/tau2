package edu.uoregon.TAU;

import java.io.*;
import java.util.*;

public class Profiler {
    static Map<String, Profile> profiles;

    static HashMap<String,ArrayList<ArrayList<String>>> methodMap;

    static {
	profiles = new HashMap<String, Profile>();
	System.out.println("TAU: init Java Profiler");

	try {
	    FileInputStream fis = new FileInputStream("/mnt/obb/map.ser");
	    ObjectInputStream ois = new ObjectInputStream(fis);
	    methodMap =  (HashMap<String,ArrayList<ArrayList<String>>>) ois.readObject();
	    ois.close();
	} catch (Exception e) {
	    e.printStackTrace();
	}
    }

    static String getCallSiteSignature() {
	StackTraceElement[] traces = Thread.currentThread().getStackTrace();

	/*
	 * 0: dalvik.system.VMStack.getThreadStackTrace
	 * 1: java.lang.Thread.getStackTrace
	 * 2: getCallSiteSignature
	 * 3: start / stop
	 * 4: callSite
	 */
	StackTraceElement callSite = traces[4];

	int line = callSite.getLineNumber();
	String key = callSite.getClassName() + ":" + callSite.getMethodName();
	ArrayList<ArrayList<String>> sigList = methodMap.get(key);

	for (ArrayList<String> sigRec: sigList) {
	    int lineBegin = Integer.parseInt(sigRec.get(0));
	    int lineEnd   = Integer.parseInt(sigRec.get(1));

	    if ((line >= lineBegin) && (line <= lineEnd)) {
		return sigRec.get(2);
	    }
	}

	return null;
    }

    /*
     * Start profile. If it doesn't exist, create one
     */
    public static void start() {
	Profile profile;
	String signature = getCallSiteSignature();

	if (signature == null) {
	    return;
	}

	if (profiles.containsKey(signature)) {
	    profile = profiles.get(signature);
	} else {
	    profile = new Profile(signature, "myType", "myGroup", Profile.TAU_DEFAULT);
	    profiles.put(signature, profile);
	}

	profile.Start();
    }

    /*
     * Stop profile.
     * Our injected code makes sure that the profile is always exist for stop()
     */
    public static void stop() {
	String signature = getCallSiteSignature();
	Profile profile = profiles.get(signature);

	profile.Stop();
    }
}
