/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * JFlex Anttask                                                           *
 * Copyright (C) 2001       Rafal Mantiuk <Rafal.Mantiuk@bellstream.pl>    *
 * All rights reserved.                                                    *
 *                                                                         *
 * This program is free software; you can redistribute it and/or modify    *
 * it under the terms of the GNU General Public License. See the file      *
 * COPYRIGHT for more information.                                         *
 *                                                                         *
 * This program is distributed in the hope that it will be useful,         *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           *
 * GNU General Public License for more details.                            *
 *                                                                         *
 * You should have received a copy of the GNU General Public License along *
 * with this program; if not, write to the Free Software Foundation, Inc., *
 * 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA                 *
 *                                                                         *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

package JFlex.anttask;

/**
 * The class is a replacement for org.apache...ant.BuildException. Exception
 * from apache package can no be used due to license restrictions.
 *
 * @author Rafal Mantiuk
 * @version JFlex 1.3.5, $Revision: 1.1.1.1 $, $Date: 2003/12/08 05:38:41 $
 */
class BuildException extends java.lang.Exception {

    /**
     * Creates new <code>BuildException</code> without detail message.
     */
    public BuildException( ) {
    }

    /**
     * Constructs an <code>BuildException</code> with the specified detail message.
     * @param msg the detail message.
     */
    public BuildException(String msg) {
        super(msg);
    }
}


