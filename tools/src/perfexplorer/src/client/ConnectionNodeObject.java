package edu.uoregon.tau.perfexplorer.client;

import java.io.Serializable;

public class ConnectionNodeObject implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = -913455767552337894L;
	public String string = null;
	public int index = 0;
	
	public ConnectionNodeObject(String string, int index) {
		this.string = string;
		this.index = index;
	}

	public String toString() {
		return string;
	}
}