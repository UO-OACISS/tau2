package edu.uoregon.tau.perfexplorer.client;

import javax.swing.*;

import java.awt.*;
import java.util.List;
import java.util.Hashtable;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfexplorer.common.*;

public class PerformanceExplorerPane extends JScrollPane implements ActionListener {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2397547783148145635L;

	private static PerformanceExplorerPane thePane = null;

	private JPanel imagePanel = null;
	private Hashtable<String, RMIPerformanceResults> resultsHash = null;
	private RMIPerformanceResults results = null;
	private static final int imagesPerRow = 6;

	public static PerformanceExplorerPane getPane () {
		if (thePane == null) {
			JPanel imagePanel = new JPanel(new BorderLayout());
			//imagePanel.setPreferredScrollableViewportSize(new Dimension(400, 400));
			thePane = new PerformanceExplorerPane(imagePanel);
		}
		thePane.repaint();
		return thePane;
	}

	private PerformanceExplorerPane (JPanel imagePanel) {
		super(imagePanel);
		this.imagePanel = imagePanel;
		this.resultsHash = new Hashtable<String, RMIPerformanceResults>();
		JScrollBar jScrollBar = this.getVerticalScrollBar();
		jScrollBar.setUnitIncrement(35);
	}

	public JPanel getImagePanel () {
		return imagePanel;
	}

	public void updateImagePanel () {
		imagePanel.removeAll();
		PerfExplorerModel model = PerfExplorerModel.getModel();
		if ((model.getCurrentSelection() instanceof Metric) || 
			(model.getCurrentSelection() instanceof Trial)) {
			// check to see if we have these results already
			// results = (RMIPerformanceResults)resultsHash.get(model.toString());
			//if (results == null) {
				PerfExplorerConnection server = PerfExplorerConnection.getConnection();
				results = server.getPerformanceResults(model);
			//}
			if (results.getResultCount() == 0) {
				return;
			}
			int iStart = 0;
			List descriptions = results.getDescriptions();
			List thumbnails = results.getThumbnails();
			int imageCount = descriptions.size();
			resultsHash.put(model.toString(), results);
			JPanel imagePanel2 = null;
			// if we have 4n+1 images, then we have a dendrogram.  Put it at the top.
			if (results.getResultCount() % imagesPerRow == 1) {
				iStart = 1;
				ImageIcon icon = new ImageIcon((byte[])(thumbnails.get(0)));
				String description = (String)(descriptions.get(0));
				PerfExplorerImageButton button = new PerfExplorerImageButton(icon, 0, description);
				button.addActionListener(this);
				imagePanel.add(button, BorderLayout.CENTER);
				imagePanel2 = new JPanel(new GridLayout((results.getResultCount()-1)/imagesPerRow,imagesPerRow));
			// if we have 5n images, then we have clustering of a trial.
			} else if (results.getResultCount() % (imagesPerRow-1) == 0) {
				imagePanel2 = new JPanel(new GridLayout(results.getResultCount()/(imagesPerRow-1),(imagesPerRow-1)));
			// we have clustering of a metric.
			} else {
				imagePanel2 = new JPanel(new GridLayout(results.getResultCount()/imagesPerRow,imagesPerRow));
			}

			for (int i = iStart ; i < imageCount ; i++) {
				ImageIcon icon = new ImageIcon((byte[])(thumbnails.get(i)));
				String description = (String)(descriptions.get(i));
				PerfExplorerImageButton button = new PerfExplorerImageButton(icon, i, description);
				button.addActionListener(this);
				imagePanel2.add(button);
			}
			imagePanel.add(imagePanel2, BorderLayout.SOUTH);
		}
		// this.repaint();
	}

	public void actionPerformed(ActionEvent e) {
		int index = Integer.parseInt(e.getActionCommand());
		// create a new modal dialog with the big image

		String description = (String)(results.getDescriptions().get(index));
		ImageIcon icon = new ImageIcon((byte[])(results.getImages().get(index)));
		//JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), null, description, JOptionPane.PLAIN_MESSAGE, icon);

        // Create and set up the window.
        JFrame frame = new JFrame("TAU/PerfExplorer: " + description);

        //Window Stuff.
        int windowWidth = 500;
        int windowHeight = 500;
        
        //Grab paraProfManager position and size.
        Point parentPosition = PerfExplorerClient.getMainFrame().getLocationOnScreen();
        Dimension parentSize = PerfExplorerClient.getMainFrame().getSize();
        int parentWidth = parentSize.width;
        int parentHeight = parentSize.height;
        
        //Set the window to come up in the center of the screen.
        int xPosition = (parentWidth - windowWidth) / 2;
        int yPosition = (parentHeight - windowHeight) / 2;

        xPosition = (int) parentPosition.getX() + xPosition;
        yPosition = (int) parentPosition.getY() + yPosition;

        frame.setLocation(xPosition, yPosition);
        frame.setSize(new java.awt.Dimension(windowWidth, windowHeight));
 
        // Make the table vertically scrollable
        //JLabel label = new JLabel(icon);
        
/*        ScrollPane pane = new ScrollPane();
        pane.add(new ImageView(icon.getImage()));
*/        
        JPanel pane = new ImagePanel(icon.getImage());
        pane.setPreferredSize(new java.awt.Dimension(windowWidth, windowHeight));
        pane.setSize(windowWidth,windowHeight);
        frame.getContentPane().add(pane);
        frame.pack();
        frame.setVisible(true);
		

	}
}
