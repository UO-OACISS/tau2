package edu.uoregon.tau.perfexplorer.server;

/**
 * This class is a simple implementation of a semaphore lock.
 * In this code it is primarily used as a lock around the database connection.
 *
 * <P>CVS $Id: Semaphore.java,v 1.2 2009/02/24 00:53:45 khuck Exp $</P>
 * @author  Kevin Huck
 * @version 0.1
 * @since   0.1
 */
class Semaphore {
	private boolean available;
	
	/**
	 * Constructor.
	 *
	 */
	public Semaphore() {
		this.available = true;
	}

	/**
	 * The WAIT method is called when a thred wants to use the 
	 * controlled resource.
	 * 
	 * @param procedure
	 */
	public synchronized void WAIT(String procedure) {
		//System.out.println("WAIT: " + procedure);

		// if another thread has control, wait for the notification
		while(available == false) {
			try {
			   wait();
			} catch (InterruptedException e) {
			   //keep trying
			}
		}

		// it's ours - let's go!
		available = false;
	}
       
	/**
	 * The SIGNAL method is called when a thread wants to release
	 * the controlled resource.
	 * 
	 * @param procedure
	 */
	public synchronized void SIGNAL(String procedure) {
		//System.out.println("SIGNAL:" + procedure);

		// we are done with the control
		available = true;

		//alert the first thread that's blocking on this semaphore
		notify(); 
	}
}
