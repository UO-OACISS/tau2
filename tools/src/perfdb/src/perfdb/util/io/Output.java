package com.perfdb.util.io;

import java.io.*;
import java.sql.*;
import com.perfdb.analysisutil.*;

/*** Handles output of a query. ***/

public class Output {
    ResultSet result;
 
    public Output(ResultSet result) {
	super();
	this.result = result;
    }

    /*** Display results of first xxx columns ***/

    public void displayResult(int column){
	try{
	    String dataRow = "";
	    String columnName = ""; 
	    for (int i=1; i<=column; i++){
		columnName += result.getMetaData().getColumnName(i);
		if (i<column) columnName += " | ";
	    }
	    System.out.println(columnName);
	    while (result.next()){
		for (int i=1; i<=column; i++){
		    dataRow += result.getString(i);	
		    if (i<column) dataRow += " | "; 
		}	
		System.out.println(dataRow);
		dataRow = "";
	    }
	}catch (Exception e){
	    System.out.println("Cannot display the result");
	    System.out.println(e.getMessage());
	    return;
	}
    }

    /*** Display the whole result. ***/

    public void displayResult(){
	try{
	    int column = result.getMetaData().getColumnCount();
	    displayResult(column);
	}catch (Exception e){
	    System.out.println("Cannot display the result");
	    System.out.println(e.getMessage());
	    return;
        }
    }

    /*** Write some columns of the result to a file. ***/

    public void writeToFile(String filename, int column){
	File writeFile = new File(filename.trim());
	if (!writeFile.exists()){
            try {
                if (writeFile.createNewFile()){
                    System.out.println("Create "+filename+ " !");
                }
            } catch(Exception ex){
		ex.printStackTrace();
            }
        }

	try{
	    BufferedWriter xwriter = new BufferedWriter(new FileWriter(writeFile));
	    writeColumnName(xwriter, column);
	    writeBody(xwriter, column);
	    xwriter.close();     
   	}catch(Exception ex){
	    ex.printStackTrace();
	}
    }

    /*** Write the whole result to a file. ***/

    public void writeToFile(String filename){
	try {
	    int column = result.getMetaData().getColumnCount();
	    writeToFile(filename, column);
	}catch (Exception ex){
	    ex.printStackTrace();	    
        }
    }

    public void writeColumnName(BufferedWriter xwriter, int column){
	try{
	    	    
	    String columnName = "";
	    for (int i=1; i<=column; i++){
		columnName += result.getMetaData().getColumnName(i);
		if (i<column) columnName += "\t";
	    }
	    xwriter.write(columnName);
	    xwriter.newLine();
	} catch(Exception ex){
	    ex.printStackTrace();
        }
    }

    public void writeBody(BufferedWriter xwriter, int column){
	try{
	    String dataRow = "";
	    while (result.next()){
		for (int i=1; i<=column; i++){
		    dataRow += result.getString(i);
		    if (i<column) dataRow += "\t";
		}
		xwriter.write(dataRow);
		xwriter.newLine();
		dataRow = "";
	    }
	} catch(Exception ex){
	    ex.printStackTrace(); 
        }
    }

    /*** Store the query result back to DB. Before storing, please make sure the data structure conforms to
         the table schema.  ***/
    public void appendToTable(com.perfdb.ConnectionManager connector, String tblName){
	
	File tempFile = new File("save.tmp");
	if (!tempFile.exists()){
            try {
                if (tempFile.createNewFile()){
                    System.out.println("Create save.tmp !");
                }
            }
            catch(Exception ex){
		ex.printStackTrace();
            }
        }
	try{
	    BufferedWriter xwriter = new BufferedWriter(new FileWriter(tempFile));
	    writeBody(xwriter, result.getMetaData().getColumnCount());
	    xwriter.close();
	}catch (Exception ex){
	    ex.printStackTrace();
	}

	StringBuffer buf = new StringBuffer();
	buf.append("copy ");
        buf.append(tblName);
	buf.append(" from ");
	buf.append("'" + tempFile.getAbsolutePath() + "';");

	try{
	    connector.getDB().executeUpdate(buf.toString());
	} catch (SQLException ex){
                ex.printStackTrace();
	}
	tempFile.delete();
    }
}


