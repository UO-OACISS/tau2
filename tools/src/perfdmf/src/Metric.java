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
 * <P>CVS $Id: Metric.java,v 1.9 2009/10/27 00:13:17 amorris Exp $</P>
 * @author	Kevin Huck, Robert Bell
 * @version	0.1
 * @since	0.1
 */
public class Metric implements Serializable {
    private int metricID;
    private int trialID;
    private String name;
    private int dbMetricID;
    private boolean derivedMetric = false;

    public int getDbMetricID() {
        return dbMetricID;
    }

    public void setDbMetricID(int dbMetricID) {
        this.dbMetricID = dbMetricID;
    }


    public void setDerivedMetric(boolean derivedMetric) {
        this.derivedMetric = derivedMetric;
    }

    public boolean getDerivedMetric() {
        return derivedMetric;
    }

    public boolean equals(Metric inMetric) {

        if (inMetric == null) {
            return false;
        }

        return (this.name.equals(inMetric.getName())) ? true : false;
    }

    public boolean equals(Object inObject) {

        if (inObject == null) {
            return false;
        }

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
        if (metricName.indexOf("_COUNT") != -1) {
            return false;
        }
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

            Vector<String> nameList = new Vector<String>();
            Vector<Integer> typeList = new Vector<Integer>();
            List<String> typeNames = new ArrayList<String>();
            List<Integer> columnSizes = new ArrayList<Integer>();
            boolean seenID = false;

            ResultSetMetaData md = resultSet.getMetaData();
            for (int i = 0; i < md.getColumnCount(); i++) {
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
                fieldNames[i] = nameList.get(i);
                fieldTypes[i] = typeList.get(i).intValue();
                if (columnSizes.get(i).intValue() > 255) {
                    fieldTypeNames[i] = typeNames.get(i) + "(" + columnSizes.get(i).toString() + ")";
                } else {
                    fieldTypeNames[i] = typeNames.get(i);
                }
            }

            db.getDatabase().setMetricFieldNames(fieldNames);
            db.getDatabase().setMetricFieldTypeNames(fieldTypeNames);
        } catch (SQLException e) {
            e.printStackTrace();
        }
    }

}
