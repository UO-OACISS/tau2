package edu.uoregon.tau.paraprof.treetable;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Rectangle;
import java.util.Iterator;
import java.util.List;

import javax.swing.JComponent;

import edu.uoregon.tau.paraprof.ColorBar;
import edu.uoregon.tau.paraprof.DataSorter;
import edu.uoregon.tau.paraprof.PPFunctionProfile;
import edu.uoregon.tau.paraprof.ParaProfTrial;
import edu.uoregon.tau.perfdmf.Function;

public class MiniHistogram extends JComponent {

    /**
	 * 
	 */
	private static final long serialVersionUID = 7474671741930948962L;
	//private ParaProfTrial ppTrial;
    //private Function function;

    private int[] bins;
    private int maxInAnyBin;
    private double maxValue;
    private double minValue;
    private double binWidth;
    
    private List<PPFunctionProfile> list;
    
    public MiniHistogram(ParaProfTrial ppTrial, Function function) {
        //this.ppTrial = ppTrial;
        //this.function = function;

    

        DataSorter dataSorter = new DataSorter(ppTrial);
        
        list = dataSorter.getFunctionData(function, false, false);

        processData();
        
        
    }

    private void processData() {

        maxValue = 0;
        minValue = 0;
        PPFunctionProfile ppFunctionProfile = null;

        int numThreads = 0;

        boolean start = true;
        for (Iterator<PPFunctionProfile> it = list.iterator(); it.hasNext();) {
            ppFunctionProfile = it.next();

                numThreads++;
                double tmpValue = ppFunctionProfile.getValue(); 
                if (start) {
                    minValue = tmpValue;
                    start = false;
                }
                maxValue = Math.max(maxValue, tmpValue);
                minValue = Math.min(minValue, tmpValue);
        }

        int numBins = 10;

        //double increment = (double) maxValue / numBins;
        binWidth = ((double) maxValue - minValue) / numBins;

        // allocate and clear the bins
        bins = new int[numBins];
        for (int i = 0; i < numBins; i++) {
            bins[i] = 0;
        }

        int count = 0;

        // fill the bins
        for (Iterator<PPFunctionProfile> it = list.iterator(); it.hasNext(); ) {
            ppFunctionProfile = it.next();
                double tmpDataValue = ppFunctionProfile.getValue();
                for (int j = 0; j < numBins; j++) {
                    if (tmpDataValue <= (minValue + (binWidth * (j + 1)))) {
                        bins[j]++;
                        count++;
                        break;
                    }
                }
        }

        // find the max number of threads in any bin
        maxInAnyBin = 0;
        for (int i = 0; i < numBins; i++) {
            maxInAnyBin = Math.max(maxInAnyBin, bins[i]);
        }

    }
    
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        Rectangle rect = this.getBounds();

        int binWidth = rect.width / 10;
        int height = (int)rect.getHeight()-1;
        
        g.setColor(Color.red);
        for (int i = 0; i < bins.length; i++) {
            if (bins[i] != 0) {

                double ratio = bins[i] / (double)maxInAnyBin;

                int pixelHeight = (int)(ratio*height);
                pixelHeight = Math.max(1, pixelHeight);
                
                
                //Color color = //:TODO Probably no side effects here...
                	ColorBar.getColor((float)ratio);
                //g.setColor(color);

                g.fillRect(binWidth*i,(int)rect.getHeight() - pixelHeight,binWidth, pixelHeight);
                //g.fillRect(binWidth*i,0,binWidth,height);

            }
        }
                
        
    }

}
