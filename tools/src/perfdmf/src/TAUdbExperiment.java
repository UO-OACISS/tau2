package edu.uoregon.tau.perfdmf;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collections;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.database.DB;

public class TAUdbExperiment {

	public static Vector<Experiment> getExperimentList(String appname, DB db) {
		 try {
             Experiment.getMetaData(db);
             String buf = "SELECT DISTINCT A.value AS app, E.value AS exp    " +
             		" FROM " + db.getSchemaPrefix() +
             		"primary_metadata A, " + db.getSchemaPrefix() +
             		"primary_metadata E     " +
             		"WHERE A.trial=E.trial AND A.name='Application' AND E.name='Experiment' AND A.value='" + appname.toString() +"'";
             // get the results
             
             ResultSet resultSet = db.executeQuery(buf);
             
             int id = 0;
             Vector<Experiment> experiments = new Vector<Experiment>();

             while (resultSet.next()) {
                 Experiment exp = new Experiment();
                 exp.setDatabase(db.getDatabase());
                exp.setID(id);
                 id++;
                 //TODO Figure out what to do about application ids
//                 exp.setApplicationID(resultSet.getInt(2));
                 String app = resultSet.getString(1);
                 exp.setName(resultSet.getString(2));
                 experiments.addElement(exp);
             }
             resultSet.close();

             Collections.sort(experiments);

             return experiments;

         } catch (SQLException e) {
             throw new DatabaseException("Error getting experiment list", e);
         }
	}
	


public static Vector<Experiment> getExperimentList( DB db) {
	 try {
        Experiment.getMetaData(db);
        String buf = "SELECT DISTINCT A.value AS app, E.value AS exp    " +
        		" FROM " + db.getSchemaPrefix() +
        		"primary_metadata A, " + db.getSchemaPrefix() +
        		"primary_metadata E     " +
        		"WHERE A.trial=E.trial AND A.name='Application' AND E.name='Experiment'";
        // get the results
        Vector<Experiment> experiments = new Vector<Experiment>();

        ResultSet resultSet = db.executeQuery(buf);
        int id = 0;
        while (resultSet.next()) {
            Experiment exp = new Experiment();
            exp.setDatabase(db.getDatabase());
           exp.setID(id);
            id++;
            //TODO: Figure out what to do about application ids
//            exp.setApplicationID(resultSet.getInt(2));
            String app = resultSet.getString(1);
            exp.setName(resultSet.getString(2));
            experiments.addElement(exp);
        }
        resultSet.close();

        Collections.sort(experiments);

        return experiments;

    } catch (SQLException e) {
        throw new DatabaseException("Error getting experiment list", e);
    }
}





}
