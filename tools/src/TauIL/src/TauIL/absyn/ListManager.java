package TauIL.absyn;

public class ListManager {
	public static final int LIST = 0, STACK = 1, QUEUE = 2;	

	private SyntaxList head, tail, cursor, temp;
	private int list_type = LIST;
        private boolean empty = true;

	public ListManager(SyntaxList list) {
		head = tail = cursor = list;
		if (list != null)
			empty = false;
		
		if (!empty) {	
			while (tail.tail != null)
				tail = tail.tail;
		}
	}

	public ListManager(SyntaxList list, int type_flag) {
		this(list);
		list_type = type_flag;
		
		if (list_type < LIST || list_type > QUEUE)
			throw new IllegalArgumentException(list_type + " : is not a legal ListManager type argument.");
	}

    	public boolean hasNext() {
		return ((cursor == null) ? false : true);
    	}

    	public AbstractSyntax next() {
		if (cursor == null)
		    return null;

		temp = cursor;
		cursor = cursor.tail;

		return temp.head;
	}

        public void reset() {
	        cursor = head;
        }

	public void insert(SyntaxList list) {
		temp = list;

		if (list != null) {
			while (temp.tail != null)
				temp = temp.tail;

			switch(list_type) {
			case STACK :
				temp.tail = head;
				head = list;
				break;
			case QUEUE :
			case LIST :
				tail.tail = list;
				tail = temp;				
			}
		}
	}

	public SyntaxList remove() {
		if (empty || cursor == null)
			return null;
		
		SyntaxList result = temp = head;

		switch (list_type) {
		case LIST :
			if (cursor != head) {
				while (temp.tail != cursor)
					temp = temp.tail;

				temp.tail = cursor.tail;
				result = cursor;
				cursor = temp.tail;
				if (cursor == null)
					tail = temp; 
				break;
			}
		case STACK :
		case QUEUE :
			head = head.tail;
			cursor = head;
			if (head == null) {
				tail = null;
				empty = true;
			}
		}
			
		temp = null;

		return result;
	}

	public SyntaxList retrieve() {
		SyntaxList result = head;

		head = tail = cursor = temp = null;
		empty = true;

		return result;
	}

	public boolean isEmpty() {
		return empty;
	}	
}			

