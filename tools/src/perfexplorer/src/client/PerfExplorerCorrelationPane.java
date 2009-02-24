package edu.uoregon.tau.perfexplorer.client;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import java.util.Hashtable;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import edu.uoregon.tau.perfdmf.*;
import edu.uoregon.tau.perfexplorer.common.*;

public class PerfExplorerCorrelationPane extends JScrollPane implements ActionListener {

	private static PerfExplorerCorrelationPane thePane = null;

	private JPanel imagePanel = null;
	private Hashtable resultsHash = null;
	private RMIPerformanceResults results = null;

	public static PerfExplorerCorrelationPane getPane () {
		if (thePane == null) {
			JPanel imagePanel = new JPanel(new BorderLayout());
			//imagePanel.setPreferredScrollableViewportSize(new Dimension(400, 400));
			thePane = new PerfExplorerCorrelationPane(imagePanel);
		}
		thePane.repaint();
		return thePane;
	}

	private PerfExplorerCorrelationPane (JPanel imagePanel) {
		super(imagePanel);
		this.imagePanel = imagePanel;
		this.resultsHash = new Hashtable();
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
			//results = (RMIPerformanceResults)resultsHash.get(model.toString());
			//if (results == null) {
				PerfExplorerConnection server = PerfExplorerConnection.getConnection();
				results = server.getCorrelationResults(model);
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
			int sqrt = (int) java.lang.Math.sqrt(results.getResultCount());
			imagePanel2 = new JPanel(new GridLayout(results.getResultCount()/sqrt, sqrt));

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
		JOptionPane.showMessageDialog(PerfExplorerClient.getMainFrame(), null, description, JOptionPane.PLAIN_MESSAGE, icon);

	}
}
