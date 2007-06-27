package client;

public class ConnectionNodeObject {
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