package clustering;

import java.util.List;
import java.util.ArrayList;

public class DendrogramTree {

	private DendrogramTree left = null;
	private DendrogramTree right = null;
	private int id = 0;
	private double height = 0.0;
	private int maxDepth = 0;

	// constructor
	public DendrogramTree (int id, double height) {
		this.id = id;
		this.height = height;
	}

	public void setLeft(DendrogramTree left) {
		this.left = left;
		if ((this.right == null) || (left.getDepth() > right.getDepth()))
			maxDepth = left.getDepth() + 1;
		else
			maxDepth = right.getDepth() + 1;
	}

	public void setRight(DendrogramTree right) {
		this.right = right;
		if ((this.left == null) || (right.getDepth() > left.getDepth()))
			maxDepth = right.getDepth() + 1;
		else
			maxDepth = left.getDepth() + 1;
	}

	public void setLeftAndRight(DendrogramTree left, DendrogramTree right) {
		this.left = left;
		this.right = right;
		if (right.getDepth() > left.getDepth())
			maxDepth = right.getDepth() + 1;
		else
			maxDepth = left.getDepth() + 1;
	}

	public boolean isLeaf() {
		return (left == null && right == null) ? true : false;
	}

	public DendrogramTree getLeft() {
		return left;
	}

	public DendrogramTree getRight() {
		return right;
	}

	public double getHeight() {
		return height;
	}

	public int[] findCenters(int count) {
		int[] centers = new int[count];
		if (count == 1) {
			centers[0] = getRepresentative();
		}else if (count == 2) {
			centers[0] = left.getRepresentative();
			centers[1] = right.getRepresentative();
		}else {
			List clusters = new ArrayList(count);
			clusters.add(left);
			clusters.add(right);
			DendrogramTree current = null;
			while (clusters.size() < count) {
				DendrogramTree winner = (DendrogramTree)clusters.get(0);
				for (int i = 0 ; i < clusters.size() ; i++) {
					current = (DendrogramTree)clusters.get(i);
					if (current.getHeight() > winner.getHeight())
						winner = current;
				}
				clusters.remove(winner);
				clusters.add(winner.getLeft());
				clusters.add(winner.getRight());
			}
			for (int i = 0 ; i < clusters.size() ; i++) {
				current = (DendrogramTree)clusters.get(i);
				centers[i] = current.getRepresentative();
			}
		}
		return centers;
	}

	public int getRepresentative() {
		if (isLeaf())
			return java.lang.Math.abs(id);
		else if (left.getDepth() == right.getDepth())
			if (left.getHeight() < right.getHeight())
				return left.getRepresentative();
			else
				return right.getRepresentative();
		else if (left.getDepth() > right.getDepth())
			return left.getRepresentative();
		else
			return right.getRepresentative();
	}

	public int getDepth() {
		return maxDepth;
	}
	
	public String toString() {
		StringBuffer buf = new StringBuffer();
		if (!isLeaf()) {
			buf.append(left.toString());
			buf.append(right.toString());
			buf.append("[" + id + "]: " );
			buf.append(left.id + ", " + right.id + ": " + height + "\n");
		}
		return buf.toString();
	}
}

