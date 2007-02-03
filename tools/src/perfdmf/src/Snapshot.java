package edu.uoregon.tau.perfdmf;

import java.util.Date;

/**
 * Snapshot object representing a Snapshot
 *
 * <P>CVS $Id: Snapshot.java,v 1.3 2007/02/03 01:38:51 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.3 $
 */
public class Snapshot {

    private String name;
    private int id;
    
    private Date timestamp;

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

    public Date getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Date timestamp) {
        this.timestamp = timestamp;
    }
    
}
