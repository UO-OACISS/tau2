/*
 * DatabaseException.java
 *
 * Copyright 2005-2007                                
 * Performance Research Laboratory, University of Oregon
 */

package edu.uoregon.tau.perfdmf;

import java.sql.SQLException;

/**
 * Encapsulates an SQLException as a RuntimeException with an additional message
 *
 * <P>CVS $Id: DatabaseException.java,v 1.4 2007/05/11 21:40:57 amorris Exp $</P>
 * @author  Alan Morris
 * @version $Revision: 1.4 $
 */
public class DatabaseException extends RuntimeException {
    private String message;
    private SQLException exception;

    public DatabaseException(String s, SQLException sqlException) {
        this.message = s;
        this.exception = sqlException;
    }

    public String getMessage() {
        return message;
    }

    public Exception getException() {
        return exception;
    }
}
