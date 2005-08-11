package common;

/**
 * Wrapper class for supporting console output.
 *
 * <P>CVS $Id: PerfExplorerOutput.java,v 1.1 2005/08/11 17:58:58 khuck Exp $</P>
 * @author khuck
 * @version 0.1
 * @since   0.1
 *
 */
public class PerfExplorerOutput {
	private boolean quiet = false;
	private static PerfExplorerOutput instance = null;

	/**
	 * Constructor. 
	 * 
	 */
	private PerfExplorerOutput (boolean quiet) {
		this.quiet = quiet;
	}

	/**
	 * Static accessor method.
	 * 
	 */
	public static void initialize(boolean quiet)  {
		instance = new PerfExplorerOutput(quiet);
	}

	/**
	 * public println method.
	 * 
	 * @param message
	 */
	public static void println (String message) {
		instance.printIt(message);
	}

	/**
	 * private printIt method.
	 * 
	 * @param message
	 */
	public void printIt (String message) {
		if (!quiet)
			System.out.println(message);
	}
}

