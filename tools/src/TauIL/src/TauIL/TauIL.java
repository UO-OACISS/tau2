package TauIL;

public class TauIL {
    public static final char [] version = { '0', '2', 'a' };

    public static void version() {
	System.err.println("TauIL v" + version[0] + "." + version[1] + version[2]);
    }
}
