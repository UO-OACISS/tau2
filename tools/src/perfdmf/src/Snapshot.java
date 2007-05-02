package edu.uoregon.tau.perfdmf;


/**
 * Snapshot object representing a Snapshot
 *
 * <P>CVS $Id: Snapshot.java,v 1.5 2007/05/02 19:43:28 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.5 $
 */
public class Snapshot {

    private String name;
    private int id;
    
    private long timestamp;

    public Snapshot(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public String toString() {
        return name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getID() {
        return id;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
    
}
