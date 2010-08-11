package edu.uoregon.tau.paraprof.util;

/*
 * This code is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public 
 * License as published by the Free Software Foundation; either 
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This code is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this program; if not, write to the Free 
 * Software Foundation, Inc., 59 Temple Place - Suite 330, Boston, 
 * MA  02111-1307, USA.
 */

/*
 @author <a href="mailto:jacob.dreyer@geosoft.no">Jacob Dreyer</a>
 */

import java.io.File;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Timer;
import java.util.TimerTask;

public class FileMonitor {
    private Timer timer;
    private HashMap<File, Long> fileMap; // File -> Long
    private Collection<WeakReference<FileMonitorListener>> listeners; // of WeakReference(FileListener)

    public FileMonitor(long pollingInterval) {
        fileMap = new HashMap<File, Long>();
        listeners = new ArrayList<WeakReference<FileMonitorListener>>();

        timer = new Timer(true);
        timer.schedule(new FileMonitorNotifier(), 0, pollingInterval);
    }

    public void stop() {
        timer.cancel();
    }

    public void addFile(File file) {
        file = new File(file.getAbsolutePath());

        if (!fileMap.containsKey(file)) {
            long modifiedTime = file.exists() ? file.lastModified() : -1;
            fileMap.put(file, new Long(modifiedTime));
        }
    }

    public void removeFile(File file) {
        fileMap.remove(file);
    }

    public void addListener(FileMonitorListener fileListener) {
        // Don't add if its already there
        for (Iterator<WeakReference<FileMonitorListener>> i = listeners.iterator(); i.hasNext();) {
            WeakReference<FileMonitorListener> reference = i.next();
            FileMonitorListener listener = reference.get();
            if (listener == fileListener)
                return;
        }

        // Use WeakReference to avoid memory leak if this becomes the
        // sole reference to the object.
        listeners.add(new WeakReference<FileMonitorListener>(fileListener));
    }

    public void removeListener(FileMonitorListener fileListener) {
        for (Iterator<WeakReference<FileMonitorListener>> i = listeners.iterator(); i.hasNext();) {
            WeakReference<FileMonitorListener> reference = i.next();
            FileMonitorListener listener = reference.get();
            if (listener == fileListener) {
                i.remove();
                break;
            }
        }
    }

    /**
     * This is the timer thread which is executed every n milliseconds
     * according to the setting of the file monitor. It investigates the
     * file in question and notify listeners if changed.
     */
    private class FileMonitorNotifier extends TimerTask {
        public void run() {
            // Loop over the registered files and see which have changed.
            // Use a copy of the list in case listener wants to alter the
            // list within its fileChanged method.
            Collection<File> files = new ArrayList<File>(fileMap.keySet());

            for (Iterator<File> i = files.iterator(); i.hasNext();) {
                File file = i.next();

                long lastModifiedTime = fileMap.get(file).longValue();
                long newModifiedTime = file.exists() ? file.lastModified() : -1;

                //                System.out.println("checking : " + file);
                //                System.out.println("lastModifiedTime = " + lastModifiedTime);
                //                System.out.println("newModifiedTime = " + file.lastModified());

                // Check if file has changed
                if (newModifiedTime != lastModifiedTime) {

                    // Register new modified time
                    fileMap.put(file, new Long(newModifiedTime));

                    // Notify listeners
                    for (Iterator<WeakReference<FileMonitorListener>> j = listeners.iterator(); j.hasNext();) {
                        WeakReference<FileMonitorListener> reference = j.next();
                        FileMonitorListener listener = reference.get();

                        // Remove from list if the back-end object has been GC'd
                        if (listener == null) {
                            j.remove();
                        } else {
                            listener.fileChanged(file);
                        }
                    }
                }
            }
        }
    }
}