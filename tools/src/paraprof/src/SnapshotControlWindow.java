package edu.uoregon.tau.paraprof;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

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

    private JCheckBox animateCheckbox = new JCheckBox("Replay");
    private JSlider animateSlider = new JSlider(0, 100);

    // auto-rotation capability
    private Animator animator;
    private volatile float rotateSpeed = 0.5f;

    private long lastTime;
    private class Animator extends Thread {

        public void run() {
            stop = false;
            while (!stop) {
                try {
                    if (rotateSpeed == 0) {
                        Thread.sleep(250);
                    } else {
                        Runnable runner = new Runnable() {
                            public void run() {
                                if (slider.getValue() >= slider.getMaximum()) {
                                    slider.setValue(0);
                                    long time = System.currentTimeMillis();
                                    double duration = ((double)time - lastTime) / 1000;
                                    lastTime = time;
                                    System.out.println("Duration: " + duration);
                                } else {
                                    slider.setValue(slider.getValue() + 2);
                                }
                            }
                        };
                        
                        SwingUtilities.invokeAndWait(runner);
                        //Thread.sleep(50);

                    }
                } catch (Exception e) {
                    // Who cares if we were interrupted
                }
            }
        }

        private volatile boolean stop = false;

        public void end() {
            stop = true;
        }

    }

    public SnapshotControlWindow(final ParaProfTrial ppTrial) {
        this.ppTrial = ppTrial;
        dataSource = ppTrial.getDataSource();

        setTitle("TAU: ParaProf: Snapshot Controller: " + ppTrial.getTrialIdentifier(ParaProf.preferences.getShowPathTitleInReverse()));

        setSize(new Dimension(300, 180));

        numSnapshots = dataSource.getMeanData().getNumSnapshots();
        selectedSnapshot = numSnapshots - 1;
        slider = new JSlider(0, selectedSnapshot);
        slider.setSnapToTicks(true);
        slider.setPaintTicks(true);
        slider.setValue(selectedSnapshot);
        slider.setBackground(Color.white);

        slider.addChangeListener(new ChangeListener() {
            public void stateChanged(ChangeEvent e) {
                selectedSnapshot = slider.getValue();
                setLabels();
                SnapshotControlWindow.this.ppTrial.setSelectedSnapshot(selectedSnapshot);
            }
        });

        animateSlider.setBackground(Color.white);
        animateCheckbox.setBackground(Color.white);

        animateCheckbox.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent evt) {
                try {
                    if (animateCheckbox.isSelected()) {
                        animator = new Animator();
                        animator.start();
                    } else {
                        animator.end();
                        animator = null;
                    }

                } catch (Exception e) {
                    ParaProfUtils.handleException(e);
                }
            }
        });

        setLabels();

        JPanel panel = new JPanel();

        panel.setBackground(Color.white);
        panel.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstants.NONE;
        ParaProfUtils.addCompItem(panel, indexLabel, gbc, 0, 0, 2, 1);
        gbc.fill = GridBagConstants.HORIZONTAL;
        ParaProfUtils.addCompItem(panel, slider, gbc, 0, 1, 2, 1);
        gbc.fill = GridBagConstants.NONE;
        ParaProfUtils.addCompItem(panel, nameLabel, gbc, 0, 2, 2, 1);
        ParaProfUtils.addCompItem(panel, timeLabel, gbc, 0, 3, 2, 1);
        gbc.anchor = GridBagConstants.SOUTH;
        ParaProfUtils.addCompItem(panel, animateCheckbox, gbc, 0, 4, 1, 1);
        ParaProfUtils.addCompItem(panel, animateSlider, gbc, 1, 4, 1, 1);
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
