package edu.uoregon.tau.paraprof.util;

import java.util.ArrayList;
import java.util.List;

public class ObjectFilter {
    private Object objects[];
    
    private boolean filter[];

    private int numShown;
    private int numHidden;
    
    public ObjectFilter(List objects) {
        this.objects = objects.toArray();
        filter = new boolean[this.objects.length];
        showAll();
    }
    
    public List getObjects() {
        List list = new ArrayList();
        for (int i=0; i<objects.length; i++) {
            if (filter[i]) {
                list.add(objects[i]);
            }
        }
        return list;
    }

    public void hide(Object object) {
        // maybe use a hash?
        for (int i=0; i<objects.length; i++) {
            if (object.equals(objects[i])) {
                filter[i] = false;
                numShown--;
                numHidden++;
            }
        }
    }

    public void show(Object object) {
        // maybe use a hash?
        for (int i=0; i<objects.length; i++) {
            if (object.equals(objects[i])) {
                filter[i] = true;
                numShown++;
                numHidden--;
            }
        }
    }
    
    public void showAll() {
        for (int i=0; i<filter.length; i++) {
            filter[i] = true;
        }
        numShown = filter.length;
        numHidden = 0;
    }


}
