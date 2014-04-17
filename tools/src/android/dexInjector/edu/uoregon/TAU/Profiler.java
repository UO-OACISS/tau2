package edu.uoregon.TAU;

import java.io.*;
import java.util.*;

public class Profiler {
    static Map<String, Profile> profiles;

    static HashMap<String,ArrayList<ArrayList<String>>> methodMap;

    static {
	profiles = new HashMap<String, Profile>();
	System.out.println("TAU: init Java Profiler");
    }

    /*
     * Start profile. If it doesn't exist, create one
     */
    public static void start(String signature) {
	Profile profile;

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
    public static void stop(String signature) {
	Profile profile = profiles.get(signature);

	profile.Stop();
    }
}
