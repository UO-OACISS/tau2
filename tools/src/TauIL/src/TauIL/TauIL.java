/***********************************************************************
 *
 * File        : TauIL.java
 * Author      : Tyrel Datwyler
 *
 * Description : Class containing the TauIL version information.
 *
 ***********************************************************************/

package TauIL;

public class TauIL {
    public static final char [] version = { '0', '3' };

    public static void version() {
	System.err.println("TauIL v" + version[0] + "." + version[1]);
    }
}
