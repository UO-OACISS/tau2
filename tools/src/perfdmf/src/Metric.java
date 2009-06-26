package edu.uoregon.tau.perfdmf;

import java.io.Serializable;
import java.sql.DatabaseMetaData;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import edu.uoregon.tau.perfdmf.database.DB;
import edu.uoregon.tau.perfdmf.database.DBConnector;

/**
 * Holds all the data for a metric in the database.
 *
 * <P>CVS $Id: Metric.java,v 1.6 2009/06/26 00:43:10 amorris Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 */
public class Metric implements Serializable {
    private int metricID;
    private int trialID;
    private String name;
    
    private boolean derivedMetric = false;

    public void setDerivedMetric(boolean derivedMetric) {
        this.derivedMetric = derivedMetric;
    }

    public boolean getDerivedMetric() {
        return derivedMetric;
    }

    public boolean equals(Metric inMetric) {
        return (this.name.equals(inMetric.getName())) ? true : false;
    }

    public boolean equals(Object inObject) {
        Metric inMetric = (Metric) inObject;
        return equals(inMetric);
    }

    /**
     * Gets the unique identifier of the current metric object.
     *
     * @return	the unique identifier of the metric
     */
    public int getID() {
        return metricID;
    }

    /**
     * Gets the unique trial identifier of the current metric object.
     *
     * @return	the unique trial identifier of the metric
     */
    public int getTrialID() {
        return trialID;
    }

    /**
     * Gets the name of the current metric object.
     *
     * @return	the name of the metric
     */
    public String getName() {
        return name;
    }

    public String toString() {
        return name;
    }

    /**
     * Sets the unique ID associated with this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	id unique ID associated with this trial
     */
    public void setID(int id) {
        this.metricID = id;
    }

    /**
     * Sets the unique trial ID associated with this trial.
     * <i> NOTE: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	trial unique trial ID associated with this trial
     */
    public void setTrialID(int trial) {
        this.trialID = trial;
    }

    /**
     * Sets the name of the current metric object.
     * <i>Note: This method is used by the DataSession object to initialize
     * the object.  Not currently intended for use by any other code.</i>
     *
     * @param	name the metric name
     */
    public void setName(String name) {
        this.name = name;
    }

    public boolean isTimeMetric() {
        String metricName = name.toUpperCase();
        if (metricName.indexOf("TIME") == -1) {
            return false;
        } else {
            return true;
        }
    }
    
    public boolean isTimeDenominator() {
        String metricName = name.toUpperCase();
        int divIndex = metricName.indexOf("/");
        int timeIndex = metricName.indexOf("TIME");
        if (divIndex != -1 && timeIndex != -1) {
            if (divIndex < timeIndex) {
                return true;
            }
        }
        return false;
    }

    public static void getMetaData(DB db) {
        // see if we've already have them
        // need to load each time in case we are working with a new database. 
        //        if (Trial.fieldNames != null)
        //            return;

        try {
            ResultSet resultSet = null;

            String trialFieldNames[] = null;
            int trialFieldTypes[] = null;

            DatabaseMetaData dbMeta = db.getMetaData();

            if ((db.getDBType().compareTo("oracle") == 0) || (db.getDBType().compareTo("derby") == 0)
                    || (db.getDBType().compareTo("db2") == 0)) {
                resultSet = dbMeta.getColumns(null, null, "METRIC", "%");
            } else {
                resultSet = dbMeta.getColumns(null, null, "metric", "%");
            }

            Vector nameList = new Vector();
            Vector typeList = new Vector();
            List typeNames = new ArrayList();
            List columnSizes = new ArrayList();
            boolean seenID = false;

            ResultSetMetaData md = resultSet.getMetaData();
            for (int i = 0 ; i < md.getColumnCount() ; i++) {
            	//System.out.println(md.getColumnName(i));
            }

            while (resultSet.next() != false) {

                int ctype = resultSet.getInt("DATA_TYPE");
                String cname = resultSet.getString("COLUMN_NAME");
                String typename = resultSet.getString("TYPE_NAME");
                Integer size = new Integer(resultSet.getInt("COLUMN_SIZE"));

                // this code is because of a bug in derby...
                if (cname.equals("ID")) {
                    if (!seenID)
                        seenID = true;
                    else
                        break;
                }

                nameList.add(resultSet.getString("COLUMN_NAME"));
                typeList.add(new Integer(ctype));
                typeNames.add(typename);
                columnSizes.add(size);
            }
            resultSet.close();

            String[] fieldNames = new String[nameList.size()];
            int[] fieldTypes = new int[typeList.size()];
            String[] fieldTypeNames = new String[typeList.size()];
            for (int i = 0; i < typeList.size(); i++) {
                fieldNames[i] = (String) nameList.get(i);
                fieldTypes[i] = ((Integer) typeList.get(i)).intValue();
                if (((Integer)columnSizes.get(i)).intValue() > 255) {
                    fieldTypeNames[i] = (String) typeNames.get(i) + "(" + columnSizes.get(i).toString() + ")";
                } else {
                    fieldTypeNames[i] = (String) typeNames.get(i);
                }
            }

            db.getDatabase().setMetricFieldNames(fieldNames);
            db.getDatabase().setMetricFieldTypeNames(fieldTypeNames);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

    
}
