package edu.uoregon.tau.perfdmf;

/**
 * Snapshot object representing a Snapshot
 *
 * <P>CVS $Id: Snapshot.java,v 1.2 2007/01/04 01:34:36 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.2 $
 */
public class Snapshot {

    private String name;
    private int id;

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
    
}
