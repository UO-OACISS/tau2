package edu.uoregon.tau.paraprof;

import java.awt.*;

import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.apache.batik.ext.swing.GridBagConstants;

import edu.uoregon.tau.perfdmf.DataSource;
import edu.uoregon.tau.perfdmf.Snapshot;

public class SnapshotControlWindow extends JFrame {

    private ParaProfTrial ppTrial;
    private DataSource dataSource;

    private JSlider slider;
    private JLabel indexLabel = new JLabel("");
    private JLabel timeLabel = new JLabel("");
    private JLabel nameLabel = new JLabel("");

    private int numSnapshots;
    private int selectedSnapshot = -1;

    public SnapshotControlWindow(final ParaProfTrial ppTrial) {
        this.ppTrial = ppTrial;
        dataSource = ppTrial.getDataSource();

        setTitle("Snapshot Controller");

        setSize(new Dimension(300, 130));

        numSnapshots = dataSource.getMeanData().getNumSnapshots();
        selectedSnapshot = numSnapshots - 1;
        slider = new JSlider(0, selectedSnapshot);
        slider.setSnapToTicks(true);
        slider.setPaintTicks(true);
        slider.setValue(selectedSnapshot);
        slider.setBackground(Color.WHITE);

        slider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                selectedSnapshot = slider.getValue();
                setLabels();
                SnapshotControlWindow.this.ppTrial.setSelectedSnapshot(selectedSnapshot);
            }
        });
        
        setLabels();
        JPanel panel = new JPanel();

        panel.setBackground(Color.WHITE);
        panel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstants.NONE;
        ParaProfUtils.addCompItem(panel, indexLabel, gbc,0,0,1,1);
        gbc.fill = GridBagConstants.HORIZONTAL;
        ParaProfUtils.addCompItem(panel, slider, gbc,0,1,1,1);
        gbc.fill = GridBagConstants.NONE;
        ParaProfUtils.addCompItem(panel, nameLabel, gbc,0,2,1,1);
        ParaProfUtils.addCompItem(panel, timeLabel, gbc,0,3,1,1);
        getContentPane().add(panel);
        
        ParaProfUtils.setFrameIcon(this);
    }

    private void setLabels() {
        indexLabel.setText("Snapshot " + selectedSnapshot);

        Snapshot snapshot = (Snapshot) dataSource.getMeanData().getSnapshots().get(selectedSnapshot);

        long position = snapshot.getTimestamp() - dataSource.getMeanData().getStartTime();
        position /= 1e6;
        timeLabel.setText("Time Position: " + position + " Seconds");

        nameLabel.setText("Name: " + snapshot.getName());
    }

}
