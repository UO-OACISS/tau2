/**
 * 
 */
package edu.uoregon.tau.vis;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * @author khuck
 *
 */
public class HeatMapData implements Iterator {
    /**
     * The whole data structure is:
     * senders: array of references to Map of receivers
     * receivers: Map of references to Map of functions
     * functions: Map of Strings to double[]
     */
    private Map[] senders = null;
    private int size = 0;
    private Map/*<String, double[]>*/maxs = new TreeMap/*<String, double[]>*/();
    private Map/*<String, double[]>*/mins = new TreeMap/*<String, double[]>*/();
    private Set/*<String>*/paths = new TreeSet();
    private static final int COUNT = 0;
    private static final int MAX = 1;
    private static final int MIN = 2;
    private static final int MEAN = 3;
    private static final int STDDEV = 4;
    private static final int VOLUME = 5;

    // for iterating over the values
    private int senderIndex = -1;
    private Iterator receiverIndex = null;

    public HeatMapData(int size) {
        this.size = size;
        this.senders = new Map[this.size];
    }

    public void put(int sender, int receiver, String function, double[] values) {
        // index into the array of senders to get the map of receivers
        Map receivers = senders[sender];
        // if the map doesn't exist yet, create it
        if (receivers == null) {
            receivers = new TreeMap();
            senders[sender] = receivers;
        }
        // get the map of functions for this receiver
        Map functions = (Map) (receivers.get(new Integer(receiver)));
        // if the map doesn't exist yet, create it
        if (functions == null) {
            functions = new TreeMap();
            receivers.put(new Integer(receiver), functions);
        }
        // put the values in the map. 
        functions.put(function, values);
        this.paths.add(function);
    }

    public double[] get(int sender, int receiver, String function) {
        double[] data = null;
        Map receivers = senders[sender];
        if (receivers != null) {
            Map functions = (Map) receivers.get(new Integer(receiver));
            if (functions != null) {
                return (double[]) functions.get(function);
            }
        }
        return data;
    }

    public double get(int sender, int receiver, String function, int index) {
        double[] data = get(sender, receiver, function);
        if (data != null) {
            return data[index];
        } else {
            return 0.0;
        }
    }

    public boolean hasNext() {
        int tmpIndex = senderIndex;
        Iterator tmpIter = receiverIndex;
        while (true) {
            // does the current receiver iterator still have data?
            if (tmpIter != null && tmpIter.hasNext()) {
                return true;
            }
            tmpIndex++;
            if (tmpIndex >= size) {
                return false;
            }
            Map tmpMap = senders[tmpIndex];
            if (tmpMap != null) {
                tmpIter = tmpMap.keySet().iterator();
            }
        }
    }

    public Object next() {
        while (true) {
            // does the current receiver iterator still have data?
            if (receiverIndex != null && receiverIndex.hasNext())
                break;
            senderIndex++;
            if (senderIndex >= size)
                return null;
            Map tmpMap = senders[senderIndex];
            if (tmpMap != null) {
                receiverIndex = tmpMap.keySet().iterator();
            }
        }
        NextValue next = new NextValue();
        next.sender = senderIndex;
        Integer tmp = (Integer) receiverIndex.next();
        next.data = (Map) senders[senderIndex].get(tmp);
        next.receiver = tmp.intValue();
        return next;
    }

    public void remove() {
    // not implemented.
    }

    public class NextValue {
        public int sender;
        public int receiver;
        public Map data;

        public double getValue(String path, int index) {
            double[] tmp = (double[]) data.get(path);
            if (tmp == null) {
                return 0.0;
            } else {
                return tmp[index];
            }
        }
    }

    public static void main(String[] args) {
        double[] dummy = { 1, 2, 3, 4, 5, 6 };
        double[] dummy2 = { 6, 5, 4, 3, 2, 1 };
        int size = 20;
        HeatMapData data = new HeatMapData(size);
        for (int x = 0; x < size; x++) {
            for (int f = 0; f < x % 10 + 1; f++) {
                data.put(x, (x + 1) % size, "test" + f, dummy2);
                data.put(x, (x + size - 1) % size, "test" + f, dummy);
            }
        }

        while (data.hasNext()) {
            NextValue next = (NextValue) data.next();
            System.out.println(next.sender + ", " + next.receiver);
            for (Iterator iter = next.data.keySet().iterator(); iter.hasNext();) {
                String func = (String) iter.next();
                double[] values = (double[]) next.data.get(func);
                System.out.print("\t" + func + ": ");
                for (int i = 0; i < 6; i++) {
                    System.out.print(values[i] + " ");
                }
                System.out.println();
            }
        }

        double[] tmp = data.get(14, 13, "test1");
        System.out.print("14, 13\n\ttest1: ");
        for (int i = 0; i < 6; i++) {
            System.out.print(tmp[i] + " ");
        }
        System.out.println();
        tmp = data.get(14, 15, "test1");
        System.out.print("14, 15\n\ttest1: ");
        for (int i = 0; i < 6; i++) {
            System.out.print(tmp[i] + " ");
        }
        System.out.println();
        return;
    }

    public void massageData() {
        while (this.hasNext()) {
            NextValue next = (HeatMapData.NextValue) this.next();
            // iterate over events
            for (Iterator iter = next.data.keySet().iterator(); iter.hasNext();) {
                String key = (String) iter.next();
                double[] data = { 0, 0, 0, 0, 0, 0 };
                if (next.data.containsKey(key)) {
                    data = (double[]) next.data.get(key);
                }
                double[] max = { 0, 0, 0, 0, 0, 0 };
                if (maxs.keySet().contains(key)) {
                    max = (double[]) maxs.get(key);
                }
                double[] min = { 0, 0, 0, 0, 0, 0 };
                if (mins.keySet().contains(key)) {
                    min = (double[]) mins.get(key);
                }

                // count and volume are fine... we need to re-compute the mean
                if (data[COUNT] > 0) {
                    data[MEAN] = data[VOLUME] / data[COUNT];
                } else {
                    data[MEAN] = 0;
                }

                // compute stddev
                if (data[COUNT] > 0)
                    data[STDDEV] = Math.sqrt(Math.abs((data[STDDEV] / data[COUNT]) - (data[MEAN] * data[MEAN])));
                else
                    data[STDDEV] = 0;

                max[COUNT] = Math.max(max[COUNT], data[COUNT]);
                max[MAX] = Math.max(max[MAX], data[MAX]);
                max[MIN] = Math.max(max[MIN], data[MIN]);
                max[MEAN] = Math.max(max[MEAN], data[MEAN]);
                max[STDDEV] = Math.max(max[STDDEV], data[STDDEV]);
                max[VOLUME] = Math.max(max[VOLUME], data[VOLUME]);

                if (data[COUNT] > 0.0) {
                    min[COUNT] = (min[COUNT] == 0.0) ? data[COUNT] : Math.min(min[COUNT], data[COUNT]);
                    min[MAX] = (min[MAX] == 0.0) ? data[MAX] : Math.min(min[MAX], data[MAX]);
                    min[MIN] = (min[MIN] == 0.0) ? data[MIN] : Math.min(min[MIN], data[MIN]);
                    min[MEAN] = (min[MEAN] == 0.0) ? data[MEAN] : Math.min(min[MEAN], data[MEAN]);
                    min[STDDEV] = (min[STDDEV] == 0.0) ? data[STDDEV] : Math.min(min[STDDEV], data[STDDEV]);
                    min[VOLUME] = (min[VOLUME] == 0.0) ? data[VOLUME] : Math.min(min[VOLUME], data[VOLUME]);
                }
                maxs.put(key, max);
                mins.put(key, min);
                next.data.put(key, data);
            }
        }
    }

    public void reset() {
        this.senderIndex = -1;
        this.receiverIndex = null;
    }

    public int getSize() {
        return size;
    }

    public Map getMaxs() {
        return maxs;
    }

    public Map getMins() {
        return mins;
    }

    public Set getPaths() {
        return paths;
    }

    public double getMin(String path, int index) {
        return ((double[]) mins.get(path))[index];
    }

    public double getMax(String path, int index) {
        return ((double[]) maxs.get(path))[index];
    }

}
