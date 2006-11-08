package edu.uoregon.tau.common;

import java.net.URL;

public class Utility {

    public static URL getResource(String name) {
        URL url = null;
        url = Utility.class.getResource(name);
        if (url == null) {
            url = Utility.class.getResource("/" + name);
        }
        if (url == null) {
            url = Utility.class.getResource("resources/" + name);
        }
        return url;
    }

}
