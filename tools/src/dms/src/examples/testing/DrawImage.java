import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import javax.swing.*;
import java.util.Vector;
import edu.uoregon.tau.dms.dss.*;
import edu.uoregon.tau.dms.analysis.*;
import java.lang.StrictMath;
import java.lang.reflect.Array;

public class DrawImage extends JPanel {
   
   private Image img = null;
   
   public DrawImage(String config, String trialID, String metricID) {

      try {
         // Create a PerfDMFSession object
         DataSession session = new PerfDMFSession();
         session.initialize(config);
         System.out.println ("API loaded...");

         // test out the analysis!
         // Trial trial = session.setTrial(55); // sppm, 256 threads on blue
         // Trial trial = session.setTrial(61); // sppm, 256 threads on frost
         // Trial trial = session.setTrial(67); // sphot, 32 nodes, 2 threads
         // Trial trial = session.setTrial(68); // pprof.dat example
         // Trial trial = session.setTrial(69); // sphot, 4 nodes 2 threads
         // Trial trial = session.setTrial(72); // sppm, openmp, 8 threads
         // Trial trial = session.setTrial(75); // sppm, openmp, 8 threads
         // Trial trial = session.setTrial(77); // sppm, 32 threads on mcr
         Trial trial = session.setTrial(Integer.parseInt(trialID));
         Vector metrics = session.getMetrics();
         // Metric metric = (Metric)(metrics.elementAt(0));
         // Metric metric = (Metric)(metrics.elementAt(7));
         Metric metric = (Metric)(metrics.elementAt(Integer.parseInt(metricID)));

         DistanceAnalysis distance = 
			new ThreadDistance((PerfDMFSession)session, trial, metric);  

         distance.getManhattanDistance();

         int[] pixels = distance.toImage(true, true);
		 int size = (int)(java.lang.StrictMath.sqrt(Array.getLength(pixels)));
         MemoryImageSource purpleMIS = new MemoryImageSource(size, size, pixels, 0, size); 
         // toolkit is an interface to the environment
         Toolkit toolkit = getToolkit();
         // create the image using the toolkit
         img = toolkit.createImage(purpleMIS);
         session.terminate();
      } catch (Exception e) {
         e.printStackTrace();
      }
   }
   
   public void paint(Graphics g) {
      super.paint(g);
      // the size of the component
      Dimension d = getSize();
      // the internal margins of the component
      Insets i = getInsets();
      // draw to fill the entire component
      g.drawImage(img, i.left, i.top, d.width - i.left - i.right, d.height - i.top - i.bottom, this );
      // g.drawImage(img, i.left, i.top, 512, 512, this );
   }
   
   // the entry point for the application
   public static void main(String[] args) {
      try {
         // create the panel
         DrawImage dimg = new DrawImage(args[0], args[1], args[2]);
         // create the window
         final JFrame f = new PlotDrawingWindow("TEST");
         // create and add an event handler for window closing event
         f.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
               System.exit(0);
            }
         });

         //Add the image pane to this frame. 
         f.getContentPane().add(dimg, BorderLayout.CENTER);

         f.setVisible(true);
      } catch (Exception r) {
	     System.out.println("Usage: DrawImage <configfile>");
         System.exit(0);
      }
   }
}

