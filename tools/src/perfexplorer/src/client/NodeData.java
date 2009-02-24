package edu.uoregon.tau.perfexplorer.client;

public class NodeData {

	private String name = null;
	private int ID = -1;
	private Object object = null;

	public NodeData (String name, int ID, Object object) {
		this.name = name;
		this.ID = ID;
		this.object = object;
	}

	public int getID () {
		return ID;
	}

	public String toString () {
		return name;
	}

	public Object getObject () {
		return object;
	}

}
