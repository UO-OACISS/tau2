package TauIL.error;

abstract public class Message {
    private String message;

    public Message(String message) {
	this.message = message;
    }

    public String toString() {
	return message;
    }
}
